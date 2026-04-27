"""Generate samples from trained EDM models.

This script loads trained checkpoints created by `run.py`, generates samples in
z-space, and saves them as `.npy` files for later evaluation and plotting.

The generated samples are in the same normalized z-space as the training and
test data stored by `data.py`. Evaluation scripts can compare these samples
against `pack["test"]` from the corresponding prepared `.pt` file.

Default behavior:
    - use train seed 4
    - find the latest Student-t-noise run and Gaussian-noise run in `runs/`
    - generate four sample sets:
        1. t_ode
        2. t_sde
        3. t_sde_coeff1
        4. g_sde
    - save outputs under `samples/`

Example:
    python sample.py --train-seed 4

    python sample.py --train-seed 4 --n-samples 1000000 --sample-batch-size 50000

    python sample.py --t-run-dir runs/<t_run_name> --g-run-dir runs/<g_run_name>
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import StudentT
from tqdm.auto import tqdm


SIGMA_DATA = 1.0


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate samples from trained EDM models.")

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Directory containing outputs from run.py. Default: <project_root>/runs.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Directory for saving generated samples. Default: <project_root>/samples.",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=4,
        help="Training-data seed used to select run directories.",
    )
    parser.add_argument(
        "--t-run-dir",
        type=Path,
        default=None,
        help="Student-t-noise run directory. If omitted, the latest matching run is used.",
    )
    parser.add_argument(
        "--g-run-dir",
        type=Path,
        default=None,
        help="Gaussian-noise run directory. If omitted, the latest matching run is used.",
    )
    parser.add_argument(
        "--t-ckpt",
        type=Path,
        default=None,
        help="Checkpoint for the Student-t-noise model. If omitted, the latest checkpoint is used.",
    )
    parser.add_argument(
        "--g-ckpt",
        type=Path,
        default=None,
        help="Checkpoint for the Gaussian-noise model. If omitted, the latest checkpoint is used.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1_000_000,
        help="Number of samples to generate for each sampler.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=50_000,
        help="Number of samples generated at once. Reduce this if GPU memory is limited.",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=80.0,
        help="Maximum sigma used during sampling.",
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=0.002,
        help="Minimum sigma used during sampling.",
    )
    parser.add_argument(
        "--n-steps-ode",
        type=int,
        default=64,
        help="Number of sampling steps for ODE samplers.",
    )
    parser.add_argument(
        "--n-steps-sde",
        type=int,
        default=128,
        help="Number of sampling steps for SDE samplers.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=7.0,
        help="Karras schedule rho parameter.",
    )
    parser.add_argument(
        "--nu-init",
        type=float,
        default=None,
        help="Student-t degrees of freedom for t-ODE initialization. Default: value from config.",
    )
    parser.add_argument(
        "--nu-init-sde",
        type=float,
        default=3.0,
        help="Student-t degrees of freedom for t-SDE initialization.",
    )
    parser.add_argument(
        "--nu-coeff",
        type=float,
        default=2.5,
        help="Degrees of freedom used in the state-dependent t-SDE coefficient.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing sample files in the output directory.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def get_project_root() -> Path:
    """Return the project root directory."""
    project_root = Path.cwd()
    if project_root.name == "notebooks":
        project_root = project_root.parent
    return project_root


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj: object) -> None:
    """Save an object as a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Run and checkpoint discovery
# -----------------------------------------------------------------------------


def list_run_dirs(runs_dir: Path) -> list[Path]:
    """List run directories under runs_dir."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    return sorted([path for path in runs_dir.iterdir() if path.is_dir()])


def pick_latest_ckpt(run_dir: Path) -> Path:
    """Return the latest checkpoint in a run directory."""
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("step*.pt"))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return ckpts[-1]


def find_run_dir(runs_dir: Path, train_seed: int, exp_key: str) -> Path:
    """Find the latest run directory matching a train seed and experiment key."""
    candidates: list[Path] = []

    for run_dir in list_run_dirs(runs_dir):
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue

        cfg = load_json(cfg_path)
        if int(cfg.get("train_seed", -1)) != int(train_seed):
            continue

        exp_name = str(cfg.get("exp_name", ""))
        if exp_key not in exp_name:
            continue

        candidates.append(run_dir)

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No run found for train_seed={train_seed}, exp_key='{exp_key}' in {runs_dir}"
        )

    # Run directory names start with a timestamp in run.py, so sorting picks the latest.
    return sorted(candidates)[-1]


# -----------------------------------------------------------------------------
# Model definition. This must match run.py.
# -----------------------------------------------------------------------------


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


def load_model_from_ckpt(
    ckpt_path: Path,
    alpha: float,
    hidden: int,
    device: torch.device,
) -> EDM:
    """Load an EDM model from a checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = EDM(alpha=alpha, hidden=hidden).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Sigma schedule and samplers
# -----------------------------------------------------------------------------


def sigma_schedule_karras(
    n_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    """Return the Karras sigma schedule from sigma_max to sigma_min."""
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2.")

    i = torch.arange(n_steps, device=device, dtype=torch.float32)
    ramp = i / (n_steps - 1)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


@torch.no_grad()
def sample_tnet_ode_heun(
    model: EDM,
    nu_init: float,
    n_samples: int,
    sigma_max: float,
    sigma_min: float,
    n_steps: int,
    rho: float,
    device: torch.device,
) -> np.ndarray:
    """Sample with a Student-t initialization and Heun ODE solver."""
    sigmas = sigma_schedule_karras(n_steps, sigma_min, sigma_max, rho, device)

    student = StudentT(
        df=torch.tensor(float(nu_init), device=device),
        loc=torch.tensor(0.0, device=device),
        scale=torch.tensor(1.0, device=device),
    )
    x = student.sample((n_samples, 1)) * sigmas[0]

    for i in range(n_steps - 1):
        sigma_t = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma_t

        def drift(x_cur: torch.Tensor, sigma_value: torch.Tensor) -> torch.Tensor:
            sigma_mat = sigma_value.expand_as(x_cur[:, :1])
            return (x_cur - model(x_cur, sigma_mat)) / sigma_value

        k1 = drift(x, sigma_t) * dt
        x_euler = x + k1
        k2 = drift(x_euler, sigma_next) * dt
        x = x + 0.5 * (k1 + k2)

    sigma_final = sigmas[-1]
    x0 = model(x, sigma_final.expand_as(x[:, :1]))
    return x0.detach().cpu().numpy()[:, 0].astype(np.float32)


@torch.no_grad()
def sample_tnet_state_sde(
    model: EDM,
    nu_init: float,
    nu_coeff: float,
    n_samples: int,
    sigma_max: float,
    sigma_min: float,
    n_steps: int,
    rho: float,
    coeff_dep_on: bool,
    device: torch.device,
) -> np.ndarray:
    """Sample with the state-dependent Student-t SDE sampler.

    If coeff_dep_on is False, the state-dependent coefficient is replaced by 1,
    giving the ablated t-SDE sampler.
    """
    if nu_coeff <= 1.0:
        raise ValueError("nu_coeff must be greater than 1.0.")

    sigmas = sigma_schedule_karras(n_steps, sigma_min, sigma_max, rho, device)

    student = StudentT(
        df=torch.tensor(float(nu_init), device=device),
        loc=torch.tensor(0.0, device=device),
        scale=torch.tensor(1.0, device=device),
    )
    x = student.sample((n_samples, 1)) * sigmas[0]
    nu = float(nu_coeff)

    for i in range(n_steps - 1):
        sigma_t = sigmas[i]
        sigma_next = sigmas[i + 1]
        d_sigma = sigma_next - sigma_t

        sigma_mat = sigma_t.expand_as(x[:, :1])
        denoised = model(x, sigma_mat)

        # Reverse-SDE drift used in this implementation. The factor 2 differs
        # from the probability-flow ODE drift.
        drift = 2.0 * (x - denoised) / sigma_t

        if coeff_dep_on:
            delta2 = (x - denoised).pow(2) / (sigma_t**2)
            coeff_dep = torch.sqrt((nu + delta2) / (nu - 1.0))
        else:
            coeff_dep = 1.0

        noise_var = max(float(sigma_t**2 - sigma_next**2), 1e-12)
        noise_std = math.sqrt(noise_var)
        noise = torch.randn(n_samples, 1, device=device) * noise_std

        x = x + drift * d_sigma + noise * coeff_dep

        if torch.isnan(x).any():
            raise RuntimeError(f"NaN detected in t-SDE sampler at step {i}")

    sigma_final = sigmas[-1]
    x0 = model(x, sigma_final.expand_as(x[:, :1]))
    return x0.detach().cpu().numpy()[:, 0].astype(np.float32)


@torch.no_grad()
def sample_gnet_sde_euler(
    model: EDM,
    n_samples: int,
    sigma_max: float,
    sigma_min: float,
    n_steps: int,
    rho: float,
    device: torch.device,
) -> np.ndarray:
    """Sample with the Gaussian VE-SDE Euler sampler."""
    sigmas = sigma_schedule_karras(n_steps, sigma_min, sigma_max, rho, device)
    x = torch.randn(n_samples, 1, device=device) * sigmas[0]

    for i in range(n_steps - 1):
        sigma_t = sigmas[i]
        sigma_next = sigmas[i + 1]
        d_sigma = sigma_next - sigma_t

        sigma_mat = sigma_t.expand_as(x[:, :1])
        denoised = model(x, sigma_mat)

        # Reverse-SDE drift used in this implementation. The factor 2 differs
        # from the probability-flow ODE drift.
        drift = 2.0 * (x - denoised) / sigma_t

        noise_var = max(float(sigma_t**2 - sigma_next**2), 1e-12)
        noise_std = math.sqrt(noise_var)
        noise = torch.randn(n_samples, 1, device=device) * noise_std

        x = x + drift * d_sigma + noise

        if torch.isnan(x).any():
            raise RuntimeError(f"NaN detected in Gaussian SDE sampler at step {i}")

    sigma_final = sigmas[-1]
    x0 = model(x, sigma_final.expand_as(x[:, :1]))
    return x0.detach().cpu().numpy()[:, 0].astype(np.float32)


# -----------------------------------------------------------------------------
# Chunked sampling and saving
# -----------------------------------------------------------------------------


def generate_samples_chunked(
    sampler_name: str,
    sampler_fn: Callable[[int], np.ndarray],
    n_samples: int,
    sample_batch_size: int,
) -> np.ndarray:
    """Generate samples in chunks and return a one-dimensional array."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if sample_batch_size <= 0:
        raise ValueError("sample_batch_size must be positive.")

    chunks: list[np.ndarray] = []
    remaining = int(n_samples)

    with tqdm(total=n_samples, desc=f"sample {sampler_name}") as progress:
        while remaining > 0:
            current_batch = min(sample_batch_size, remaining)
            chunk = sampler_fn(current_batch).reshape(-1).astype(np.float32)
            chunks.append(chunk)
            remaining -= current_batch
            progress.update(current_batch)

    return np.concatenate(chunks, axis=0)


def save_samples(path: Path, samples: np.ndarray, force: bool) -> None:
    """Save generated samples as a .npy file."""
    if path.exists() and not force:
        raise FileExistsError(f"Sample file already exists: {path}. Use --force to overwrite.")
    np.save(path, samples.astype(np.float32))
    print(f"Saved samples: {path} shape={samples.shape}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    """Generate and save samples from trained models."""
    args = parse_args()

    project_root = get_project_root()
    runs_dir = args.runs_dir if args.runs_dir is not None else project_root / "runs"
    samples_root = args.samples_dir if args.samples_dir is not None else project_root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    device = get_device()
    set_seed(args.seed)

    t_run_dir = args.t_run_dir
    if t_run_dir is None:
        t_run_dir = find_run_dir(runs_dir, args.train_seed, "student_t_noise")

    g_run_dir = args.g_run_dir
    if g_run_dir is None:
        g_run_dir = find_run_dir(runs_dir, args.train_seed, "gaussian_noise")

    t_ckpt = args.t_ckpt if args.t_ckpt is not None else pick_latest_ckpt(t_run_dir)
    g_ckpt = args.g_ckpt if args.g_ckpt is not None else pick_latest_ckpt(g_run_dir)

    cfg_t = load_json(t_run_dir / "config.json")
    cfg_g = load_json(g_run_dir / "config.json")

    if int(cfg_t.get("train_seed", -1)) != int(cfg_g.get("train_seed", -2)):
        raise ValueError("Student-t and Gaussian runs have different train_seed values.")

    if str(cfg_t.get("data_pt_path", "")) != str(cfg_g.get("data_pt_path", "")):
        print("Warning: Student-t and Gaussian runs point to different data_pt_path values.")

    nu_train = float(cfg_t.get("nu", 3.0))
    nu_init_ode = float(args.nu_init) if args.nu_init is not None else nu_train

    alpha_t = float(cfg_t["alpha_resolved"])
    alpha_g = float(cfg_g.get("alpha_resolved", 1.0))
    hidden_t = int(cfg_t.get("hidden", 128))
    hidden_g = int(cfg_g.get("hidden", 128))

    model_t = load_model_from_ckpt(t_ckpt, alpha=alpha_t, hidden=hidden_t, device=device)
    model_g = load_model_from_ckpt(g_ckpt, alpha=alpha_g, hidden=hidden_g, device=device)

    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    output_dir = samples_root / f"{timestamp}_trainseed{args.train_seed}"
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"PROJECT_ROOT: {project_root.resolve()}")
    print(f"RUNS_DIR    : {runs_dir.resolve()}")
    print(f"SAMPLES_DIR : {output_dir.resolve()}")
    print(f"device      : {device}")
    print(f"T_RUN_DIR   : {t_run_dir.name}")
    print(f"T_CKPT      : {t_ckpt.name}")
    print(f"G_RUN_DIR   : {g_run_dir.name}")
    print(f"G_CKPT      : {g_ckpt.name}")
    print(f"n_samples   : {args.n_samples}")

    metadata = {
        "train_seed": int(args.train_seed),
        "seed": int(args.seed),
        "n_samples": int(args.n_samples),
        "sample_batch_size": int(args.sample_batch_size),
        "sigma_max": float(args.sigma_max),
        "sigma_min": float(args.sigma_min),
        "n_steps_ode": int(args.n_steps_ode),
        "n_steps_sde": int(args.n_steps_sde),
        "rho": float(args.rho),
        "nu_train": float(nu_train),
        "nu_init_ode": float(nu_init_ode),
        "nu_init_sde": float(args.nu_init_sde),
        "nu_coeff": float(args.nu_coeff),
        "space": "z",
        "t_run_dir": str(t_run_dir),
        "g_run_dir": str(g_run_dir),
        "t_ckpt": str(t_ckpt),
        "g_ckpt": str(g_ckpt),
        "data_pt_path": str(cfg_t.get("data_pt_path", "")),
        "sample_files": {},
    }

    sample_specs: list[tuple[str, Callable[[int], np.ndarray], int]] = [
        (
            "t_ode",
            lambda n: sample_tnet_ode_heun(
                model=model_t,
                nu_init=nu_init_ode,
                n_samples=n,
                sigma_max=args.sigma_max,
                sigma_min=args.sigma_min,
                n_steps=args.n_steps_ode,
                rho=args.rho,
                device=device,
            ),
            args.seed + 101,
        ),
        (
            "t_sde",
            lambda n: sample_tnet_state_sde(
                model=model_t,
                nu_init=args.nu_init_sde,
                nu_coeff=args.nu_coeff,
                n_samples=n,
                sigma_max=args.sigma_max,
                sigma_min=args.sigma_min,
                n_steps=args.n_steps_sde,
                rho=args.rho,
                coeff_dep_on=True,
                device=device,
            ),
            args.seed + 202,
        ),
        (
            "t_sde_coeff1",
            lambda n: sample_tnet_state_sde(
                model=model_t,
                nu_init=args.nu_init_sde,
                nu_coeff=args.nu_coeff,
                n_samples=n,
                sigma_max=args.sigma_max,
                sigma_min=args.sigma_min,
                n_steps=args.n_steps_sde,
                rho=args.rho,
                coeff_dep_on=False,
                device=device,
            ),
            args.seed + 303,
        ),
        (
            "g_sde",
            lambda n: sample_gnet_sde_euler(
                model=model_g,
                n_samples=n,
                sigma_max=args.sigma_max,
                sigma_min=args.sigma_min,
                n_steps=args.n_steps_sde,
                rho=args.rho,
                device=device,
            ),
            args.seed + 404,
        ),
    ]

    for sampler_name, sampler_fn, sampler_seed in sample_specs:
        set_seed(sampler_seed)
        samples = generate_samples_chunked(
            sampler_name=sampler_name,
            sampler_fn=sampler_fn,
            n_samples=args.n_samples,
            sample_batch_size=args.sample_batch_size,
        )
        sample_path = output_dir / f"{sampler_name}.npy"
        save_samples(sample_path, samples, force=args.force)
        metadata["sample_files"][sampler_name] = str(sample_path)

    save_json(output_dir / "metadata.json", metadata)
    print(f"Saved metadata: {output_dir / 'metadata.json'}")
    print("Sampling complete.")


if __name__ == "__main__":
    main()
