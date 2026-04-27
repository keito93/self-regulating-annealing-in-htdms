"""Train EDM denoisers on the prepared one-dimensional Student-t datasets.

This script assumes that `data.py` has already generated normalized `.pt` files
under `data/normalized_pt/`. Each `.pt` file should contain at least the key
`"train"`, which stores z-score-normalized training samples of shape (N, 1).

Default behavior:
    - train seeds: 0, 1, 2, 3, 4
    - two experiment variants:
        1. Student-t noise + log-normal sigma + Student-t alpha
        2. Gaussian noise + log-normal sigma + standard alpha

Example:
    python run.py

    python run.py --total-steps 200000 --train-seeds 0 1 2 3 4

    python run.py --experiments student_t gaussian --batch-size 1000
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import StudentT
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


SIGMA_DATA = 1.0


@dataclass
class TrainCfg:
    """Configuration for one training run."""

    exp_name: str

    # Reproducibility.
    seed: int = 42

    # Data.
    train_seed: int = 0
    test_seed: int = 123
    n_train: int = 1_000_000
    n_test: int = 1_000_000
    data_pt_path: str = ""

    # Model.
    hidden: int = 128
    alpha_mode: str = "standard"
    nu: float = 3.0

    # Sigma sampling.
    sigma_mode: str = "lognormal"
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    pi_mean: float = -1.2
    pi_std: float = 1.2

    # Noise sampling.
    noise_mode: str = "gaussian"

    # Training.
    total_steps: int = 10_000
    batch_size: int = 1000
    num_workers: int = 0
    lr: float = 2e-4
    weight_decay: float = 0.0
    ckpt_interval_steps: int = 10_000
    grad_clip: float = 0.0
    log_interval_steps: int | None = None


class Array1DDataset(Dataset):
    """Dataset wrapper for arrays of shape (N, 1)."""

    def __init__(self, array: np.ndarray) -> None:
        if array.ndim != 2 or array.shape[1] != 1:
            raise ValueError(f"Expected array of shape (N, 1), got {array.shape}")
        self.x = torch.from_numpy(array.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.x[index]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train EDM denoisers on Student-t data.")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Dataset directory. Default: <project_root>/data.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Directory for saving run outputs. Default: <project_root>/runs.",
    )
    parser.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Training-data seeds to run.",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=123,
        help="Test-data seed used in the prepared dataset filename.",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=1_000_000,
        help="Number of training samples used in the prepared dataset filename.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1_000_000,
        help="Number of test samples used in the prepared dataset filename.",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=3.0,
        help="Degrees of freedom used for Student-t noise and alpha.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training random seed.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10_000,
        help="Number of optimization steps per run.",
    )
    parser.add_argument(
        "--ckpt-interval-steps",
        type=int,
        default=None,
        help="Checkpoint interval. Default: same as total_steps.",
    )
    parser.add_argument(
        "--log-interval-steps",
        type=int,
        default=None,
        help="Loss logging interval. Default: total_steps // 1000.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="Hidden width of the MLP denoiser.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Gradient clipping norm. Use 0 to disable clipping.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["student_t", "gaussian", "all"],
        default=["all"],
        help="Experiment variants to run.",
    )

    return parser.parse_args()


def get_project_root() -> Path:
    """Return the project root directory."""
    project_root = Path.cwd()
    if project_root.name == "notebooks":
        project_root = project_root.parent
    return project_root


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def c_noise(sigma: torch.Tensor) -> torch.Tensor:
    """Noise embedding used in EDM."""
    return 0.25 * torch.log(sigma)


def c_skip(sigma: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """EDM skip coefficient with alpha-controlled preconditioning."""
    return (SIGMA_DATA**2) / (alpha * sigma**2 + SIGMA_DATA**2)


def c_out(sigma: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """EDM output coefficient with alpha-controlled preconditioning."""
    return sigma * SIGMA_DATA * torch.sqrt(alpha) / torch.sqrt(
        alpha * sigma**2 + SIGMA_DATA**2
    )


def c_in(sigma: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """EDM input coefficient with alpha-controlled preconditioning."""
    return torch.sqrt(alpha) / torch.sqrt(alpha * sigma**2 + SIGMA_DATA**2)


def lambda_weight(sigma: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Loss weight corresponding to 1 / c_out(sigma, alpha)^2."""
    return 1.0 / (c_out(sigma, alpha) ** 2)


class MLPDenoiser(nn.Module):
    """Small MLP denoiser for one-dimensional inputs."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Start from an approximately identity-like EDM mapping through the skip term.
        output_layer = self.net[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, c_noise(sigma)], dim=-1)
        return self.net(h)


class EDM(nn.Module):
    """EDM wrapper with alpha-controlled preconditioning."""

    def __init__(self, alpha: float, hidden: int = 128) -> None:
        super().__init__()
        self.f = MLPDenoiser(hidden=hidden)
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        return c_skip(sigma, alpha) * x + c_out(sigma, alpha) * self.f(
            c_in(sigma, alpha) * x,
            sigma,
        )


def sample_sigma_loguniform(
    batch_size: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample sigma from a log-uniform distribution."""
    log_sigma = torch.rand(batch_size, 1, device=device) * (
        math.log(sigma_max) - math.log(sigma_min)
    ) + math.log(sigma_min)
    return log_sigma.exp()


def sample_sigma_lognormal(
    batch_size: int,
    pi_mean: float,
    pi_std: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample sigma from a log-normal distribution."""
    return torch.exp(torch.randn(batch_size, 1, device=device) * pi_std + pi_mean)


def sample_noise_student_t(batch_size: int, nu: float, device: torch.device) -> torch.Tensor:
    """Sample Student-t noise."""
    distribution = StudentT(df=torch.tensor(float(nu), device=device))
    return distribution.sample((batch_size, 1))


def sample_noise_gaussian_like(x: torch.Tensor) -> torch.Tensor:
    """Sample Gaussian noise with the same shape as x."""
    return torch.randn_like(x)


def alpha_from_cfg(cfg: TrainCfg) -> float:
    """Resolve the alpha value from the training configuration."""
    if cfg.alpha_mode == "standard":
        return 1.0
    if cfg.alpha_mode == "student_t":
        if cfg.nu <= 2.0:
            raise ValueError("student_t alpha requires nu > 2.")
        return float(cfg.nu) / (float(cfg.nu) - 2.0)
    raise ValueError(f"Unknown alpha_mode: {cfg.alpha_mode}")


def prepared_pt_path(
    data_dir: Path,
    train_seed: int,
    test_seed: int,
    n_train: int,
    n_test: int,
) -> Path:
    """Return the normalized dataset path created by data.py."""
    return data_dir / "normalized_pt" / (
        f"student_t1d_norm_trainseed{train_seed}_testseed{test_seed}"
        f"_Ntrain{n_train}_Ntest{n_test}.pt"
    )


def load_train_array(pt_path: Path) -> np.ndarray:
    """Load normalized training data from a prepared .pt file."""
    if not pt_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {pt_path}")

    pack = torch.load(pt_path, map_location="cpu")
    train = pack["train"].numpy().astype(np.float32)

    if train.ndim != 2 or train.shape[1] != 1:
        raise ValueError(f"Expected train array of shape (N, 1), got {train.shape}")

    return train


def make_train_loader(
    train: np.ndarray,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Create a deterministic training DataLoader."""
    dataset = Array1DDataset(train)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def make_run_dir(runs_dir: Path, cfg: TrainCfg) -> Path:
    """Create and return a run directory."""
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    tag = f"{cfg.exp_name}_tr{cfg.train_seed}_{cfg.alpha_mode}_{cfg.noise_mode}"
    run_dir = runs_dir / f"{timestamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, obj: object) -> None:
    """Save an object as a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_loss_csv(path: Path, loss_log: list[tuple[int, float]]) -> None:
    """Save sparse training losses as a CSV file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("step,loss\n")
        for step, loss in loss_log:
            f.write(f"{step},{loss}\n")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg_dict: dict,
) -> None:
    """Save a training checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "cfg": cfg_dict,
        "rng_state_cpu": torch.get_rng_state(),
        "rng_state_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    torch.save(checkpoint, path)


def train_one_run(
    cfg: TrainCfg,
    runs_dir: Path,
    device: torch.device,
) -> Path:
    """Train one EDM model and save outputs."""
    set_seed(cfg.seed)

    train_array = load_train_array(Path(cfg.data_pt_path))
    train_loader = make_train_loader(
        train=train_array,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    run_dir = make_run_dir(runs_dir, cfg)
    ckpt_dir = run_dir / "checkpoints"
    plot_dir = run_dir / "plots"
    ckpt_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    alpha = alpha_from_cfg(cfg)
    model = EDM(alpha=alpha, hidden=cfg.hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay,
    )

    cfg_dict = asdict(cfg)
    cfg_dict["alpha_resolved"] = alpha
    cfg_dict["device"] = str(device)
    save_json(run_dir / "config.json", cfg_dict)

    save_checkpoint(
        ckpt_dir / "step0000000.pt",
        model=model,
        optimizer=optimizer,
        step=0,
        cfg_dict=cfg_dict,
    )

    log_interval = cfg.log_interval_steps
    if log_interval is None:
        log_interval = max(1, cfg.total_steps // 1000)

    loss_log: list[tuple[int, float]] = []
    min_loss = float("inf")
    final_loss = None

    model.train()
    data_iter = iter(train_loader)
    progress = tqdm(total=cfg.total_steps, desc=f"train {cfg.exp_name} tr{cfg.train_seed}")

    for step in range(1, cfg.total_steps + 1):
        try:
            x0 = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x0 = next(data_iter)

        x0 = x0.to(device, non_blocking=True)
        batch_size = x0.size(0)

        if cfg.sigma_mode == "loguniform":
            sigma = sample_sigma_loguniform(
                batch_size=batch_size,
                sigma_min=cfg.sigma_min,
                sigma_max=cfg.sigma_max,
                device=device,
            )
        elif cfg.sigma_mode == "lognormal":
            sigma = sample_sigma_lognormal(
                batch_size=batch_size,
                pi_mean=cfg.pi_mean,
                pi_std=cfg.pi_std,
                device=device,
            )
        else:
            raise ValueError(f"Unknown sigma_mode: {cfg.sigma_mode}")

        if cfg.noise_mode == "gaussian":
            eps = sample_noise_gaussian_like(x0)
        elif cfg.noise_mode == "student_t":
            eps = sample_noise_student_t(batch_size, cfg.nu, device)
        else:
            raise ValueError(f"Unknown noise_mode: {cfg.noise_mode}")

        x = x0 + eps * sigma
        pred = model(x, sigma)
        loss = (lambda_weight(sigma, model.alpha) * (pred - x0).pow(2)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        loss_value = float(loss.item())
        final_loss = loss_value
        min_loss = min(min_loss, loss_value)

        progress.update(1)
        progress.set_postfix(loss=loss_value)

        is_checkpoint_step = (step % cfg.ckpt_interval_steps == 0) or (
            step == cfg.total_steps
        )
        if (step % log_interval == 0) or is_checkpoint_step:
            loss_log.append((step, loss_value))

        if is_checkpoint_step:
            save_checkpoint(
                ckpt_dir / f"step{step:07d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                cfg_dict=cfg_dict,
            )

    progress.close()
    model.eval()

    save_loss_csv(run_dir / "loss.csv", loss_log)

    metrics = {
        "final_loss": float(final_loss) if final_loss is not None else None,
        "min_loss": float(min_loss) if min_loss < float("inf") else None,
        "steps": int(cfg.total_steps),
        "logged_points": int(len(loss_log)),
        "log_interval_steps": int(log_interval),
    }
    save_json(run_dir / "metrics.json", metrics)
    save_loss_plot(run_dir / "plots" / "loss.png", loss_log, cfg.exp_name)

    print(f"Saved run to: {run_dir}")
    return run_dir


def save_loss_plot(path: Path, loss_log: list[tuple[int, float]], title: str) -> None:
    """Save a loss curve plot."""
    plt.figure(figsize=(6, 4))
    if len(loss_log) > 0:
        steps, losses = zip(*loss_log)
        plt.plot(steps, losses, label="train loss")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_base_experiments(args: argparse.Namespace) -> list[TrainCfg]:
    """Build the base experiment configurations."""
    selected = set(args.experiments)
    if "all" in selected:
        selected = {"student_t", "gaussian"}

    ckpt_interval = args.ckpt_interval_steps
    if ckpt_interval is None:
        ckpt_interval = args.total_steps

    common = dict(
        seed=args.seed,
        test_seed=args.test_seed,
        n_train=args.n_train,
        n_test=args.n_test,
        hidden=args.hidden,
        nu=args.nu,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ckpt_interval_steps=ckpt_interval,
        grad_clip=args.grad_clip,
        log_interval_steps=args.log_interval_steps,
    )

    experiments: list[TrainCfg] = []

    if "student_t" in selected:
        experiments.append(
            TrainCfg(
                exp_name="student_t_noise__lognormal_sigma__student_t_alpha",
                alpha_mode="student_t",
                sigma_mode="lognormal",
                noise_mode="student_t",
                **common,
            )
        )

    if "gaussian" in selected:
        experiments.append(
            TrainCfg(
                exp_name="gaussian_noise__lognormal_sigma__standard_alpha",
                alpha_mode="standard",
                sigma_mode="lognormal",
                noise_mode="gaussian",
                **common,
            )
        )

    return experiments


def main() -> None:
    """Run all requested training experiments."""
    args = parse_args()

    project_root = get_project_root()
    data_dir = args.data_dir if args.data_dir is not None else project_root / "data"
    runs_dir = args.runs_dir if args.runs_dir is not None else project_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    base_experiments = build_base_experiments(args)

    print(f"PROJECT_ROOT: {project_root.resolve()}")
    print(f"DATA_DIR    : {data_dir.resolve()}")
    print(f"RUNS_DIR    : {runs_dir.resolve()}")
    print(f"device      : {device}")
    print(f"train seeds : {args.train_seeds}")
    print("experiments :")
    for cfg in base_experiments:
        print(f"  - {cfg.exp_name}")

    run_dirs: list[str] = []

    for train_seed in args.train_seeds:
        pt_path = prepared_pt_path(
            data_dir=data_dir,
            train_seed=train_seed,
            test_seed=args.test_seed,
            n_train=args.n_train,
            n_test=args.n_test,
        )
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Missing prepared dataset: {pt_path}\n"
                "Run `python data.py` before running this script."
            )

        for base_cfg in base_experiments:
            cfg = copy.deepcopy(base_cfg)
            cfg.train_seed = int(train_seed)
            cfg.data_pt_path = str(pt_path)

            run_dir = train_one_run(cfg, runs_dir=runs_dir, device=device)
            run_dirs.append(str(run_dir))

    print("\nAll runs saved:")
    for run_dir in run_dirs:
        print(f" - {run_dir}")


if __name__ == "__main__":
    main()
