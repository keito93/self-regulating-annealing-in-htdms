"""Train K_phi(x,sigma) = E[eps**2 | x=x0+sigma*eps], eps~StudentT(nu).

A separate, directly supervised conditional second-moment estimator for the
1D heavy-tailed diffusion model. This does NOT train the denoiser and does
NOT require the true conditional moment (oracle) for supervision.

The population-correct conditional mean loss is
    E[log(K_phi(x,sigma)) + eps**2 / K_phi(x,sigma)].
Instead of exp(MLP), use a quadratic tail envelope:
    u = sqrt(alpha) * x / sqrt(1 + alpha*sigma**2), alpha=nu/(nu-2)
    z = u / sqrt(1 + u**2)
    v_phi = softplus(MLP([z, log(sigma)/4])) + epsilon
    K_phi = (1 + u**2) * v_phi.
We optimize the equivalent normalized loss
    E[log(v_phi) + (eps**2/(1+u**2))/v_phi].

Place this file next to run.py in the existing GitHub repository. Examples:

  python run_second_moment.py --train-seed 0 --moment-steps 100000 \
      --batch-size 1000 --sigma-mode loguniform \
      --sigma-min 0.002 --sigma-max 20 --ckpt-every 10000

  python run_second_moment.py --train-seed 0 \
      --denoiser-ckpt runs/YOUR_RUN/checkpoints/step0100000.pt \
      --moment-steps 100000

It saves in runs/second_moment/ so existing sample.py/run.py auto-discovery
cannot mistake a moment-only run for an ordinary denoiser training run.

Use MomentNet(nu, hidden).forward(x, sigma) for K directly in a new sampler.
The checkpoint contains both the frozen denoiser state in 'model' and the
new estimator in 'moment_model'. The estimator is NOT a learned variance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import StudentT
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_TYPE = "normalized_quadratic_softplus_v1"


class MomentNet(nn.Module):
    """Positive estimate of E[eps**2 | x_sigma=x] with quadratic tail scaling.

    All tensors x and sigma should have shape (batch,1), and sigma > 0.
    The network is dimension-specific (1D), but the loss is a proper mean score.
    """

    def __init__(self, nu: float = 3.0, hidden: int = 128,
                 min_positive: float = 1e-6):
        super().__init__()
        if nu <= 2.0:
            raise ValueError("nu must be > 2 for a finite Student-t second moment")
        self.register_buffer("alpha", torch.tensor(float(nu / (nu - 2.0))))
        self.register_buffer("init_logit", torch.tensor(
            float(self._inverse_softplus(nu / (nu - 2.0)))
        ))
        self.min_positive = float(min_positive)
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        if value > 20:
            return value  # stable approximation to softplus^{-1}
        return math.log(math.expm1(value))

    def normalized_prediction(self, x: torch.Tensor, sigma: torch.Tensor):
        if x.shape != sigma.shape or x.ndim != 2 or x.shape[1] != 1:
            raise ValueError("x and sigma must both have shape (batch, 1)")
        # The caller is responsible for positive sigma (no silent clipping).
        if torch.any(sigma <= 0).item():
            raise ValueError("sigma must be strictly positive")
        alpha = self.alpha
        u = (torch.sqrt(alpha) * x) / torch.sqrt(1.0 + alpha * sigma.square())
        envelope = 1.0 + u.square()
        z = u / torch.sqrt(envelope)
        tau = 0.25 * torch.log(sigma)
        features = torch.cat([z, tau], dim=1)
        raw = self.net(features) + self.init_logit
        positive = F.softplus(raw) + self.min_positive
        return positive, envelope

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        positive, envelope = self.normalized_prediction(x, sigma)
        return envelope * positive


def proper_mean_loss(predicted_normalized: torch.Tensor,
                     envelope: torch.Tensor,
                     eps: torch.Tensor) -> torch.Tensor:
    """Population optimum K(x,sigma)=E[eps^2|x,sigma], no fourth moment needed."""
    normalized_target = eps.square() / envelope
    return (predicted_normalized.log()
            + normalized_target / predicted_normalized).mean()


class Array1DDataset(Dataset):
    def __init__(self, array):
        if array.ndim != 2 or array.shape[1] != 1:
            raise ValueError(f"Expected (N,1), got {array.shape}")
        self.data = torch.from_numpy(array.astype(np.float32, copy=False))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Directly train conditional second moment K")
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for moment training, separate from data seed")
    p.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <runs-dir>/second_moment")
    p.add_argument("--denoiser-ckpt", type=Path, default=None,
                   help="Original Student-t denoiser checkpoint. Default: auto-find latest")
    p.add_argument("--data-pt", type=Path, default=None,
                   help="Override dataset path from denoiser checkpoint")
    p.add_argument("--moment-steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--sigma-mode", choices=("loguniform", "lognormal"),
                   default="loguniform")
    p.add_argument("--sigma-min", type=float, default=0.002)
    p.add_argument("--sigma-max", type=float, default=20.0)
    p.add_argument("--pi-mean", type=float, default=-1.2)
    p.add_argument("--pi-std", type=float, default=1.2)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-samples", type=int, default=8192)
    p.add_argument("--ckpt-every", type=int, default=10_000)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return p.parse_args()


def select_latest_denoiser_ckpt(runs_dir: Path, train_seed: int) -> Path:
    candidates = []
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            if int(cfg.get("train_seed", -1)) != train_seed:
                continue
            if cfg.get("noise_mode") != "student_t":
                continue
            if cfg.get("moment_only", False) or cfg.get("variance_only", False):
                continue
            checkpoints = []
            for path in (run_dir / "checkpoints").glob("step*.pt"):
                try:
                    step = int(path.stem[4:])
                except ValueError:
                    continue
                if step > 0:
                    checkpoints.append((step, path))
            if checkpoints:
                ckpt_step, ckpt_path = max(checkpoints)
                candidates.append((run_dir.name, ckpt_step, ckpt_path))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    if not candidates:
        raise FileNotFoundError(
            f"No original Student-t checkpoint for training seed {train_seed} "
            f"under {runs_dir}. Supply --denoiser-ckpt explicitly."
        )
    return max(candidates)[2]


def resolve_data_pt(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    # GitHub config may store relative paths resolved from project root.
    alt = PROJECT_ROOT / path
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(f"Cannot find normalized dataset: {path}")


def sample_sigmas(n: int, args, device: torch.device) -> torch.Tensor:
    if args.sigma_mode == "loguniform":
        v = torch.rand(n, 1, device=device)
        return torch.exp(math.log(args.sigma_min)
                         + v * (math.log(args.sigma_max) - math.log(args.sigma_min)))
    return torch.exp(args.pi_mean + args.pi_std * torch.randn(n, 1, device=device))


def save_checkpoint(path: Path, model: MomentNet,
                    optimizer: torch.optim.Optimizer,
                    frozen_denoiser: dict, denoiser_step: int,
                    moment_step: int, config: dict) -> None:
    # 'step' corresponds to frozen denoiser training and 'moment_step' to K learning.
    torch.save({
        "model": frozen_denoiser,
        "moment_model": model.state_dict(),
        "moment_optimizer": optimizer.state_dict(),
        "step": denoiser_step,
        "moment_step": moment_step,
        "cfg": config,
        "moment_model_type": MODEL_TYPE,
    }, path)


def main():
    args = parse_args()
    if args.moment_steps < 1 or args.batch_size < 1 or args.hidden < 1:
        raise ValueError("moment-steps, batch-size and hidden must be positive")
    if not (0 < args.sigma_min < args.sigma_max):
        raise ValueError("Expected 0 < sigma-min < sigma-max")
    if args.val_samples < 1 or args.log_every < 1 or args.val_every < 1:
        raise ValueError("val-samples, log-every and val-every must be positive")
    if args.ckpt_every < 0 or args.grad_clip < 0:
        raise ValueError("ckpt-every and grad-clip must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    set_seed(args.seed)

    denoiser_path = (args.denoiser_ckpt if args.denoiser_ckpt is not None
                     else select_latest_denoiser_ckpt(args.runs_dir, args.train_seed))
    denoiser_path = denoiser_path.resolve()
    denoiser_ckpt = torch.load(denoiser_path, map_location="cpu")
    original_cfg = dict(denoiser_ckpt.get("cfg") or {})
    if not original_cfg:
        cfg_file = denoiser_path.parent.parent / "config.json"
        if cfg_file.exists():
            original_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    if "model" not in denoiser_ckpt:
        raise KeyError(f"Original checkpoint has no 'model': {denoiser_path}")
    if original_cfg.get("noise_mode") != "student_t":
        raise ValueError("Direct K estimator currently supports Student-t forward noise only")
    if int(original_cfg.get("train_seed", -1)) != args.train_seed:
        raise ValueError("--train-seed does not match denoiser checkpoint train_seed")
    denoiser_step = int(denoiser_ckpt.get("step", 0))
    if denoiser_step <= 0:
        raise ValueError("The denoiser checkpoint was not trained (step<=0)")
    nu = float(original_cfg.get("nu", 3.0))
    if nu <= 2.0:
        raise ValueError("Require nu>2 to learn a finite second moment")
    if args.data_pt is None:
        if not original_cfg.get("data_pt_path"):
            raise ValueError("Denoiser checkpoint has no data_pt_path. Supply --data-pt")
        data_path = Path(original_cfg["data_pt_path"])
    else:
        data_path = args.data_pt
    data_path = resolve_data_pt(data_path)
    pack = torch.load(data_path, map_location="cpu")
    if int(pack.get("train_seed", -1)) != args.train_seed:
        raise ValueError("Dataset train_seed does not match --train-seed")
    if abs(float(pack.get("nu", nu)) - nu) > 1e-6:
        raise ValueError("Dataset Student-t degrees of freedom mismatch")
    train_all = np.asarray(pack["train"], dtype=np.float32)
    if train_all.ndim != 2 or train_all.shape[1] != 1:
        raise ValueError("Training data must have shape (N,1)")
    if len(train_all) <= args.batch_size:
        raise ValueError("Training data must have N > batch-size for a validation holdout")

    # Hold validation examples out from moment training. Never use test data
    # during development: the repository test split remains final evaluation.
    n_val = min(args.val_samples, len(train_all) - args.batch_size)
    val_indices = np.random.default_rng(args.seed + 1000).choice(
        len(train_all), size=n_val, replace=False
    )
    val_mask = np.ones(len(train_all), dtype=bool)
    val_mask[val_indices] = False
    val_clean = train_all[val_indices].copy()
    train = train_all[val_mask].copy()

    model = MomentNet(nu=nu, hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    student = StudentT(df=torch.tensor(nu, device=device))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        Array1DDataset(train), batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
        pin_memory=(device.type == "cuda"), generator=generator,
    )

    # Fixed validation corruptions from a held-out TRAIN subset.
    val_x0 = torch.from_numpy(val_clean).to(device)
    val_sigma = sample_sigmas(len(val_x0), args, device)
    val_eps = student.sample((len(val_x0), 1))
    val_x = val_x0 + val_sigma * val_eps

    output_root = args.output_dir if args.output_dir is not None else args.runs_dir / "second_moment"
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    run_dir = output_root / f"{timestamp}_direct_k_tr{args.train_seed}_dstep{denoiser_step}"
    if run_dir.exists():
        raise FileExistsError(f"Run output directory already exists: {run_dir}")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    cfg = dict(original_cfg)
    cfg.update({
        "train_seed": args.train_seed,
        "data_pt_path": str(data_path),
        "moment_only": True,
        "moment_model_type": MODEL_TYPE,
        "moment_target": "K(x,sigma)=E[eps^2|x_sigma=x]",
        "moment_hidden": args.hidden,
        "moment_steps": args.moment_steps,
        "moment_seed": args.seed,
        "moment_sigma_mode": args.sigma_mode,
        "moment_sigma_min": args.sigma_min,
        "moment_sigma_max": args.sigma_max,
        "moment_pi_mean": args.pi_mean,
        "moment_pi_std": args.pi_std,
        "moment_lr": args.lr,
        "moment_grad_clip": args.grad_clip,
        "moment_batch_size": args.batch_size,
        "moment_val_samples": len(val_x0),
        "moment_validation_source": "train_holdout",
        "denoiser_ckpt_path": str(denoiser_path),
        "denoiser_step": denoiser_step,
        "device": str(device),
    })
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print("device:", device)
    print("train seed:", args.train_seed)
    print("denoiser checkpoint:", denoiser_path)
    print("denoiser step:", denoiser_step, "(frozen, not trained here)")
    print("normalized dataset:", data_path)
    print("nu:", nu)
    print("moment training output:", run_dir)

    records = []
    iterator = iter(loader)
    running_sum = 0.0
    running_count = 0
    last_val = float("nan")
    progress = tqdm(range(1, args.moment_steps + 1), desc="train direct K")
    model.train()

    for step in progress:
        try:
            x0 = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x0 = next(iterator)
        x0 = x0.to(device, non_blocking=True)
        sigma = sample_sigmas(len(x0), args, device)
        eps = student.sample((len(x0), 1))
        x = x0 + sigma * eps

        normalized, envelope = model.normalized_prediction(x, sigma)
        loss = proper_mean_loss(normalized, envelope, eps)
        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite moment loss at step={step}. "
                f"Inspect sigma, eps and normalized prediction."
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm).item():
                raise RuntimeError(f"Non-finite gradient norm at step={step}")
        optimizer.step()

        running_sum += float(loss.detach().item())
        running_count += 1
        do_log = (step % args.log_every == 0) or (step == args.moment_steps)
        do_val = (step % args.val_every == 0) or (step == args.moment_steps)
        if do_val:
            model.eval()
            with torch.no_grad():
                val_norm, val_env = model.normalized_prediction(val_x, val_sigma)
                last_val = float(proper_mean_loss(val_norm, val_env, val_eps).item())
            model.train()
            if not math.isfinite(last_val):
                raise RuntimeError(f"Validation loss is not finite at step={step}")
        if do_log or do_val:
            mean_loss = running_sum / max(1, running_count) if running_count else float("nan")
            records.append((step, mean_loss, last_val))
            if do_log:
                progress.set_postfix(loss=f"{mean_loss:.4g}", val=f"{last_val:.4g}")
                running_sum, running_count = 0.0, 0

        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            save_checkpoint(
                ckpt_dir / f"moment_step{step:07d}.pt",
                model, optimizer, denoiser_ckpt["model"], denoiser_step,
                step, cfg,
            )

    final_path = ckpt_dir / f"step{denoiser_step:07d}.pt"
    model.eval()
    save_checkpoint(final_path, model, optimizer,
                    denoiser_ckpt["model"], denoiser_step,
                    args.moment_steps, cfg)
    with (run_dir / "moment_loss.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["moment_step", "mean_train_loss", "fixed_validation_loss"])
        writer.writerows(records)
    if records:
        steps = np.array([p[0] for p in records])
        train_losses = np.array([p[1] for p in records], dtype=np.float64)
        val_losses = np.array([p[2] for p in records], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, train_losses, label="Training (window average)")
        ax.plot(steps, val_losses, label="Validation (fixed corruptions)")
        ax.set(xlabel="Moment optimizer step", ylabel="Normalized proper mean loss",
               title="Direct conditional second-moment learning")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "moment_loss.png", dpi=150)
        plt.close(fig)

    summary = {"moment_steps": args.moment_steps,
               "denoiser_step": denoiser_step,
               "final_val_loss": last_val,
               "checkpoint": str(final_path)}
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2),
                                            encoding="utf-8")
    print("\nSaved direct-K training run:", run_dir)
    print("Final checkpoint:", final_path)
    print("Use checkpoint['moment_model'] with MomentNet to obtain K directly.")
    print("Do NOT exponentiate the output or add an approximate moment term.")


if __name__ == "__main__":
    main()
