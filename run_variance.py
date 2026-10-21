"""Train conditional residual variance for a frozen 1D HTDM denoiser.

This script does NOT retrain the denoiser. It reads a pretrained checkpoint
from run.py, reuses its normalized training data, and trains V_phi(x,sigma)>0
with the loss E[log V_phi + ((x0-D_theta(x,sigma))/sigma)^2 / V_phi].

The population minimizer for a fixed denoiser is
  V_phi^*(x,sigma) = Var[x0 | x,sigma] / sigma^2
                    + (E[x0 | x,sigma]-D_theta(x,sigma))^2 / sigma^2.
It equals the normalized conditional variance when D_theta is exact.

The variance network uses softplus + a bounded quadratic term rather than
exp(unconstrained MLP) to avoid the exponential extrapolation found earlier.
This parameterization does not guarantee accurate extrapolation or SDE stability.

Example (pretrained in the user's notebook):
  python run_variance.py \\
    --denoiser-ckpt runs/YOUR_RUN/checkpoints/step0100000.pt \\
    --denoiser-variant notebook --variance-steps 100000

Example (checkpoint trained with the GitHub run.py):
  python run_variance.py \\
    --denoiser-ckpt runs/YOUR_RUN/checkpoints/step0100000.pt \\
    --denoiser-variant github --variance-steps 100000
"""

from __future__ import annotations

import argparse
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

SIGMA_DATA = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def c_noise(sigma: torch.Tensor) -> torch.Tensor:
    return 0.25 * torch.log(sigma)


def c_skip(sigma: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return SIGMA_DATA**2 / (alpha * sigma.square() + SIGMA_DATA**2)


def c_out(sigma: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return (sigma * SIGMA_DATA * torch.sqrt(alpha)
            / torch.sqrt(alpha * sigma.square() + SIGMA_DATA**2))


def c_in(sigma: torch.Tensor, alpha: torch.Tensor, variant: str) -> torch.Tensor:
    denominator = torch.sqrt(alpha * sigma.square() + SIGMA_DATA**2)
    if variant == "notebook":
        # This factor is present in the notebook code shared in the conversation.
        return torch.sqrt(alpha) / denominator
    if variant == "github":
        # This matches the publicly available repository's run.py.
        return 1.0 / denominator
    raise ValueError(f"Unknown denoiser variant: {variant}")


class MLPDenoiser(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, c_noise(sigma)], dim=-1)
        return self.net(inputs)


class EDM(nn.Module):
    def __init__(self, alpha: float, hidden: int = 128,
                 variant: str = "notebook"):
        super().__init__()
        self.f = MLPDenoiser(hidden=hidden)
        self.variant = variant
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        x_in = c_in(sigma, a, self.variant) * x
        return c_skip(sigma, a) * x + c_out(sigma, a) * self.f(x_in, sigma)


class VarianceNet(nn.Module):
    """Predict V_phi(x,sigma) ~= Var[x0|x,sigma]/sigma**2, dimension d=1.

    An explicit quadratic envelope prevents exponential growth from exp(MLP).
    Both outputs are nonnegative: softplus(base) + x_scaled**2 * sigmoid(quad).
    The quadratic coefficient is bounded (here by 1); check expressivity
    against an oracle before treating this as a final architecture.
    """

    def __init__(self, nu: float, hidden: int = 128):
        super().__init__()
        if not nu > 2.0:
            raise ValueError("nu must be > 2")
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        k_init = nu / (nu - 2.0)
        self.register_buffer(
            "base_bias", torch.tensor(math.log(math.expm1(k_init)))
        )
        self.register_buffer("quad_bias", torch.tensor(-4.0))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Explicitly define the variance input scaling, independent of the
        # denoiser's preconditioning variant (notebook vs GitHub).
        x_scaled = x / torch.sqrt(SIGMA_DATA**2 + sigma.square())
        h = torch.cat([x_scaled, c_noise(sigma)], dim=-1)
        raw = self.net(h)
        base = F.softplus(raw[:, :1] + self.base_bias)
        quadratic = x_scaled.square() * torch.sigmoid(
            raw[:, 1:2] + self.quad_bias
        )
        return base + quadratic + 1e-6


class Array1DDataset(Dataset):
    def __init__(self, array: np.ndarray):
        if array.ndim != 2 or array.shape[1] != 1:
            raise ValueError(f"Expected (N,1), got {array.shape}")
        self.x = torch.from_numpy(array.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train conditional variance using a frozen HTDM denoiser"
    )
    p.add_argument("--denoiser-ckpt", type=Path, required=True)
    p.add_argument("--denoiser-variant", choices=["notebook", "github"],
                   default="notebook")
    p.add_argument("--data-pt", type=Path, default=None,
                   help="Override the training .pt file stored in the checkpoint")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--variance-steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--hidden", type=int, default=128,
                   help="Width of the variance model (denoiser width comes from checkpoint)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sigma-mode", choices=["loguniform", "lognormal"],
                   default="loguniform",
                   help="loguniform covers the full sampling sigma range")
    p.add_argument("--sigma-min", type=float, default=0.002)
    p.add_argument("--sigma-max", type=float, default=20.0)
    p.add_argument("--pi-mean", type=float, default=-1.2)
    p.add_argument("--pi-std", type=float, default=1.2)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="Intermediate variance checkpoint interval; 0 disables")
    return p.parse_args()


def save_checkpoint(path: Path, denoiser: EDM, variance_model: VarianceNet,
                    optimizer: torch.optim.Optimizer, denoiser_step: int,
                    variance_step: int, cfg: dict) -> None:
    torch.save({
        "model": denoiser.state_dict(),
        "variance_model": variance_model.state_dict(),
        "variance_optimizer": optimizer.state_dict(),
        "step": denoiser_step,
        "variance_step": variance_step,
        "cfg": cfg,
        "variance_model_type": "softplus_quadratic_v1",
        "rng_state_cpu": torch.get_rng_state(),
        "rng_state_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
    }, path)


def main():
    args = parse_args()
    if args.variance_steps < 1 or args.batch_size < 1:
        raise ValueError("variance-steps and batch-size must be positive")
    if args.sigma_min <= 0 or args.sigma_max <= args.sigma_min:
        raise ValueError("Require 0 < sigma_min < sigma_max")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be nonnegative")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_ckpt = torch.load(args.denoiser_ckpt, map_location="cpu")
    if "model" not in raw_ckpt:
        raise KeyError("No 'model' found in the pretrained denoiser checkpoint")
    denoiser_cfg = dict(raw_ckpt.get("cfg") or {})
    nu = float(denoiser_cfg.get("nu", 3.0))
    if not nu > 2:
        raise ValueError("Student-t degrees of freedom must be > 2")
    alpha = float(denoiser_cfg.get("alpha_resolved", nu / (nu - 2.0)))
    hidden = int(denoiser_cfg.get("hidden", 128))
    denoiser_step = int(raw_ckpt.get("step", 0))
    if denoiser_step <= 0:
        raise ValueError("Checkpoint appears to contain an untrained denoiser (step=0)")

    stored_data_path = denoiser_cfg.get("data_pt_path", "")
    if args.data_pt is None and not stored_data_path:
        raise ValueError("Checkpoint lacks data_pt_path; supply --data-pt")
    pt_path = args.data_pt if args.data_pt is not None else Path(stored_data_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"Dataset not found: {pt_path}")
    pack = torch.load(pt_path, map_location="cpu")
    data = pack["train"].numpy().astype(np.float32)
    if data.ndim != 2 or data.shape[1] != 1:
        raise ValueError(f"Expected normalized train data (N,1), got {data.shape}")

    dataset_seed = pack.get("train_seed", None)
    run_seed = denoiser_cfg.get("train_seed", None)
    if dataset_seed is not None and run_seed is not None:
        if int(dataset_seed) != int(run_seed):
            raise ValueError(
                f"Train-seed mismatch: dataset {dataset_seed} vs checkpoint {run_seed}"
            )

    denoiser = EDM(alpha, hidden, args.denoiser_variant).to(device)
    denoiser.load_state_dict(raw_ckpt["model"], strict=True)
    denoiser.eval()
    denoiser.requires_grad_(False)

    variance_model = VarianceNet(nu=nu, hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(variance_model.parameters(), lr=args.lr)

    loader_gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        Array1DDataset(data), batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=0, generator=loader_gen,
        pin_memory=torch.cuda.is_available(),
    )

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    tag = str(denoiser_cfg.get("exp_name", "student_t")) + "_variance"
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    run_dir = args.runs_dir / (timestamp + "_" + tag + "_tr" +
                               str(denoiser_cfg.get("train_seed", "unknown")))
    run_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()

    out_cfg = dict(denoiser_cfg)
    out_cfg.update({
        "exp_name": tag,
        "data_pt_path": str(pt_path),
        "alpha_resolved": alpha,
        "nu": nu,
        "hidden": hidden,
        "train_variance": True,
        "variance_only": True,
        "variance_model_type": "softplus_quadratic_v1",
        "variance_hidden": args.hidden,
        "variance_steps": args.variance_steps,
        "variance_sigma_mode": args.sigma_mode,
        "variance_sigma_min": args.sigma_min,
        "variance_sigma_max": args.sigma_max,
        "denoiser_variant": args.denoiser_variant,
        "denoiser_ckpt_path": str(args.denoiser_ckpt),
        "variance_seed": args.seed,
    })
    with open(run_dir / "config.json", "w", encoding="utf-8") as fp:
        json.dump(out_cfg, fp, ensure_ascii=False, indent=2)

    progress = tqdm(total=args.variance_steps, desc="train conditional variance")
    log_records = []
    iterator = iter(loader)
    student = StudentT(df=torch.tensor(nu, device=device))
    variance_model.train()

    for step in range(1, args.variance_steps + 1):
        try:
            x0 = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x0 = next(iterator)
        x0 = x0.to(device, non_blocking=True)
        b = x0.size(0)

        if args.sigma_mode == "loguniform":
            u = torch.rand(b, 1, device=device)
            sigma = torch.exp(
                math.log(args.sigma_min)
                + (math.log(args.sigma_max) - math.log(args.sigma_min)) * u
            )
        else:
            sigma = torch.exp(
                torch.randn(b, 1, device=device) * args.pi_std + args.pi_mean
            )

        eps = student.sample((b, 1))
        x = x0 + eps * sigma
        with torch.no_grad():
            d = denoiser(x, sigma)
            squared_residual = ((x0 - d) / sigma).square()

        v_hat = variance_model(x, sigma)
        loss = (torch.log(v_hat) + squared_residual / v_hat).mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Nonfinite loss at step {step}. "
                "Inspect sigma, residuals and variance predictions."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(variance_model.parameters(), args.grad_clip)
        optimizer.step()
        loss_value = float(loss.detach().item())
        progress.update()
        if step % args.log_every == 0 or step == args.variance_steps:
            log_records.append((step, loss_value))
            progress.set_postfix(variance_loss=loss_value)

        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            save_checkpoint(
                ckpt_dir / f"variance_step{step:07d}.pt",
                denoiser, variance_model, optimizer,
                denoiser_step, step, out_cfg,
            )

    progress.close()
    variance_model.eval()

    # Preserve compatibility with existing sampler's pick_latest_ckpt(),
    # which searches for files named step*.pt.
    final_ckpt = ckpt_dir / f"step{denoiser_step:07d}.pt"
    save_checkpoint(
        final_ckpt, denoiser, variance_model, optimizer,
        denoiser_step, args.variance_steps, out_cfg,
    )

    with open(run_dir / "variance_loss.csv", "w", encoding="utf-8") as fp:
        fp.write("step,loss\n")
        for step, value in log_records:
            fp.write(f"{step},{value}\n")
    if log_records:
        steps, values = zip(*log_records)
        plt.figure(figsize=(6, 4))
        plt.plot(steps, values)
        plt.xlabel("Variance training step")
        plt.ylabel("log V + residual^2 / V")
        plt.title("Conditional residual-variance learning")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(run_dir / "variance_loss.png", dpi=150)
        plt.close()
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump({"final_variance_loss": log_records[-1][1],
                   "variance_steps": args.variance_steps}, fp, indent=2)
    print("Output:", run_dir)
    print("Final checkpoint:", final_ckpt)
    print("Use checkpoint['variance_model'] in the new sampler.")


if __name__ == "__main__":
    main()
