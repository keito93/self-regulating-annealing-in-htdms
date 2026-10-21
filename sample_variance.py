"""Generate 1D HTDM samples with an additionally learned conditional variance.

This file is intentionally separate from the repository's sample.py and
reuses its original four samplers without modifying sample.py.

Required neighboring files:
    sample.py          (original GitHub sampler; left unchanged)
    run_variance.py    (conditional-variance trainer and VarianceNet class)

Five output files (normalized z space):
    t_ode.npy
    t_sde.npy               original approximate state-dependent t-SDE
    t_sde_variance.npy      NEW: corrected t-SDE using conditional variance
    t_sde_coeff1.npy
    g_sde.npy

The Student-t denoiser is loaded from the *variance training checkpoint*,
which stores an unchanged copy of the pretrained denoiser. This ensures that
both t-SDE samplers use exactly the same denoiser.

Example:
    python3 sample_variance.py --train-seed 0 --n-samples 10000 \\
        --sample-batch-size 1000 --nu-coeff 3 --samples-dir samples_qq

For a quick diagnostic test, run only the two t-SDE samplers:
    python3 sample_variance.py --train-seed 0 --samplers t_sde t_sde_variance \\
        --n-samples 1000 --sample-batch-size 1000 --n-steps-sde 128

By default sigma_min and sigma_max are taken from the variance-training
checkpoint to avoid inadvertently sampling outside its trained sigma range.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.distributions import StudentT

# Reuse the original repository's code; this does not modify sample.py.
from sample import (
    find_run_dir,
    generate_samples_chunked,
    get_device,
    get_project_root,
    list_run_dirs,
    load_json,
    pick_latest_ckpt,
    sample_gnet_sde_euler,
    sample_tnet_ode_heun,
    sample_tnet_state_sde,
    save_json,
    save_samples,
    set_seed,
    sigma_schedule_karras,
)
# EXACTLY the same model definitions used by run_variance.py.
from run_variance import EDM, VarianceNet

ALL_SAMPLERS = (
    "t_ode", "t_sde", "t_sde_variance", "t_sde_coeff1", "g_sde"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample HTDMs, including the learned-variance t-SDE, "
                    "without modifying the original sample.py."
    )
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--samples-dir", type=Path, default=None)
    parser.add_argument("--train-seed", type=int, default=4)
    parser.add_argument(
        "--variance-run-dir", type=Path, default=None,
        help="Run directory from run_variance.py (auto-discovered if omitted)."
    )
    parser.add_argument(
        "--variance-ckpt", type=Path, default=None,
        help="Explicit checkpoint from run_variance.py; overrides auto-discovery."
    )
    parser.add_argument("--g-run-dir", type=Path, default=None)
    parser.add_argument("--g-ckpt", type=Path, default=None)
    parser.add_argument(
        "--g-denoiser-variant", choices=("github", "notebook"),
        default=None, help="Default: read from Gaussian config or 'github'."
    )
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--sample-batch-size", type=int, default=50_000)
    parser.add_argument(
        "--samplers", nargs="+", choices=ALL_SAMPLERS,
        default=list(ALL_SAMPLERS),
        help="Run any subset of the five samplers."
    )
    parser.add_argument(
        "--sigma-max", type=float, default=None,
        help="Default: variance_sigma_max from variance checkpoint."
    )
    parser.add_argument(
        "--sigma-min", type=float, default=None,
        help="Default: variance_sigma_min from variance checkpoint."
    )
    parser.add_argument("--n-steps-ode", type=int, default=64)
    parser.add_argument("--n-steps-sde", type=int, default=128)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument(
        "--nu-init", type=float, default=None,
        help="t-ODE initial Student-t degrees of freedom; default: training nu."
    )
    parser.add_argument(
        "--nu-init-sde", type=float, default=None,
        help="t-SDE initial Student-t df; default: --nu-coeff."
    )
    parser.add_argument(
        "--nu-coeff", type=float, default=None,
        help="t-SDE coefficient df; default: variance checkpoint's training nu."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-sigma-extrapolation", action="store_true",
        help="Permit explicit sigma range beyond variance training range."
    )
    return parser.parse_args()


def find_variance_run_dir(runs_dir: Path, train_seed: int) -> Path:
    """Avoid matching ordinary student_t runs by checking metadata flags."""
    candidates = []
    for run_dir in list_run_dirs(runs_dir):
        cfg_file = run_dir / "config.json"
        if not cfg_file.exists():
            continue
        cfg = load_json(cfg_file)
        if (int(cfg.get("train_seed", -1)) == train_seed
                and bool(cfg.get("variance_only", False))
                and cfg.get("variance_model_type") == "softplus_quadratic_v1"):
            candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(
            "No compatible variance run found for train_seed="
            f"{train_seed} in {runs_dir}. Train run_variance.py first, "
            "or pass --variance-ckpt explicitly."
        )
    return sorted(candidates)[-1]


def load_variance_models(
    checkpoint_path: Path, device: torch.device, train_seed: int
) -> tuple[EDM, VarianceNet, dict, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError(f"Variance checkpoint has no cfg dict: {checkpoint_path}")
    if int(cfg.get("train_seed", -1)) != train_seed:
        raise ValueError(
            f"Checkpoint train_seed={cfg.get('train_seed')} differs from "
            f"--train-seed={train_seed}."
        )
    model_type = checkpoint.get("variance_model_type", cfg.get("variance_model_type"))
    if model_type != "softplus_quadratic_v1":
        raise ValueError(
            "Unsupported variance_model_type: "
            f"{model_type!r}. Expected 'softplus_quadratic_v1'."
        )
    if "model" not in checkpoint or "variance_model" not in checkpoint:
        raise KeyError(
            "Expected BOTH 'model' and 'variance_model' in the variance "
            "checkpoint produced by run_variance.py."
        )
    if int(checkpoint.get("step", 0)) <= 0:
        raise ValueError("The denoiser in this checkpoint was not trained (step <= 0).")
    if int(checkpoint.get("variance_step", 0)) <= 0:
        raise ValueError("The variance network was not trained (variance_step <= 0).")

    nu = float(cfg["nu"])
    if nu <= 2:
        raise ValueError("Training nu must be > 2 for a finite second moment.")
    alpha = float(cfg.get("alpha_resolved", nu / (nu - 2.0)))
    hidden = int(cfg.get("hidden", 128))
    variance_hidden = int(cfg.get("variance_hidden", 128))
    variant = str(cfg.get("denoiser_variant", "github"))
    if variant not in ("github", "notebook"):
        raise ValueError(f"Unsupported denoiser_variant: {variant}")

    denoiser = EDM(alpha=alpha, hidden=hidden, variant=variant).to(device)
    denoiser.load_state_dict(checkpoint["model"], strict=True)
    denoiser.eval().requires_grad_(False)
    variance = VarianceNet(nu=nu, hidden=variance_hidden).to(device)
    variance.load_state_dict(checkpoint["variance_model"], strict=True)
    variance.eval().requires_grad_(False)
    return denoiser, variance, cfg, checkpoint


def load_gaussian_model(
    ckpt: Path, cfg: dict, device: torch.device, variant_override: str | None
) -> EDM:
    variant = variant_override or str(cfg.get("denoiser_variant", "github"))
    alpha = float(cfg.get("alpha_resolved", 1.0))
    hidden = int(cfg.get("hidden", 128))
    checkpoint = torch.load(ckpt, map_location="cpu")
    model = EDM(alpha=alpha, hidden=hidden, variant=variant).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval().requires_grad_(False)


def _check_finite(name: str, value: torch.Tensor, step: int,
                  sigma: torch.Tensor, x: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        bad = (~torch.isfinite(value)).sum().item()
        finite = value[torch.isfinite(value)]
        bounds = (f"[{finite.min().item():.5g}, {finite.max().item():.5g}]"
                  if finite.numel() else "none")
        finite_x = x[torch.isfinite(x)]
        x_max = finite_x.abs().max().item() if finite_x.numel() else float("nan")
        raise RuntimeError(
            f"Non-finite {name} at step={step}, sigma={sigma.item():.6g}; "
            f"bad_values={bad}; finite_{name}_range={bounds}; "
            f"max_finite_abs_x={x_max:.6g}. No silent clipping was applied."
        )


@torch.no_grad()
def sample_tnet_state_sde_variance(
    model: EDM,
    variance_model: VarianceNet,
    nu_init: float,
    nu_coeff: float,
    n_samples: int,
    sigma_max: float,
    sigma_min: float,
    n_steps: int,
    rho: float,
    device: torch.device,
) -> np.ndarray:
    """Original Euler t-SDE, replacing delta^2 by delta^2 + V_phi.

    V_phi is normalized conditional residual variance from run_variance.py:
        V_phi(x,sigma) ~ E[((x0 - D_theta(x,sigma))/sigma)^2 | x,sigma].

    Accordingly the added coefficient term is
        K_hat(x,sigma) = ((x - D_theta)/sigma)^2 + V_phi(x,sigma).
    It is exact when the denoiser equals the conditional mean and V_phi is
    calibrated. The same time stepping and drift as sample.py are used.
    """
    if nu_init <= 0.0 or nu_coeff <= 2.0:
        raise ValueError("Require nu_init > 0 and nu_coeff > 2.")
    sigmas = sigma_schedule_karras(
        n_steps, sigma_min, sigma_max, rho, device
    )
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
        _check_finite("denoised", denoised, i, sigma_t, x)
        drift = 2.0 * (x - denoised) / sigma_t
        _check_finite("drift", drift, i, sigma_t, x)

        delta2 = ((x - denoised) / sigma_t).square()
        variance = variance_model(x, sigma_mat)
        _check_finite("variance", variance, i, sigma_t, x)
        if (variance <= 0).any().item():
            raise RuntimeError(f"Variance model returned non-positive values at step {i}")
        k_hat = delta2 + variance
        _check_finite("conditional_second_moment", k_hat, i, sigma_t, x)
        coeff = torch.sqrt((nu + k_hat) / (nu - 1.0))
        _check_finite("diffusion_coefficient", coeff, i, sigma_t, x)

        # Identical discretization to the original sample.py t_sde sampler.
        noise_var = max(float(sigma_t**2 - sigma_next**2), 1e-12)
        noise_std = math.sqrt(noise_var)
        noise = torch.randn(n_samples, 1, device=device) * noise_std
        x = x + drift * d_sigma + noise * coeff
        _check_finite("state", x, i, sigma_t, x)

    sigma_final = sigmas[-1]
    x0 = model(x, sigma_final.expand_as(x[:, :1]))
    _check_finite("final_denoised", x0, n_steps - 1, sigma_final, x)
    return x0.detach().cpu().numpy()[:, 0].astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0 or args.sample_batch_size <= 0:
        raise ValueError("n-samples and sample-batch-size must be positive.")
    if args.n_steps_ode < 2 or args.n_steps_sde < 2 or args.rho <= 0:
        raise ValueError("Require n-steps >= 2 and rho > 0.")
    if len(set(args.samplers)) != len(args.samplers):
        raise ValueError("Duplicate sampler names in --samplers.")

    root = get_project_root()
    runs_dir = args.runs_dir if args.runs_dir is not None else root / "runs"
    samples_root = (args.samples_dir if args.samples_dir is not None
                    else root / "samples_variance")
    device = get_device()
    set_seed(args.seed)

    if args.variance_ckpt is not None:
        variance_ckpt = args.variance_ckpt
        variance_run_dir = variance_ckpt.parent.parent
    else:
        variance_run_dir = (args.variance_run_dir
                            if args.variance_run_dir is not None
                            else find_variance_run_dir(runs_dir, args.train_seed))
        variance_ckpt = pick_latest_ckpt(variance_run_dir)

    model_t, variance_model, cfg_t, variance_pack = load_variance_models(
        variance_ckpt, device, args.train_seed
    )
    sigma_min_train = float(cfg_t.get("variance_sigma_min", 0.002))
    sigma_max_train = float(cfg_t.get("variance_sigma_max", 20.0))
    sigma_min = (float(args.sigma_min) if args.sigma_min is not None
                 else sigma_min_train)
    sigma_max = (float(args.sigma_max) if args.sigma_max is not None
                 else sigma_max_train)
    if sigma_min <= 0 or sigma_max <= sigma_min:
        raise ValueError("Require 0 < sigma_min < sigma_max.")
    if ((sigma_min < sigma_min_train or sigma_max > sigma_max_train)
            and "t_sde_variance" in args.samplers):
        if not args.allow_sigma_extrapolation:
            raise ValueError(
                f"Requested sigma range [{sigma_min}, {sigma_max}] exceeds "
                f"variance training range [{sigma_min_train}, {sigma_max_train}]. "
                "Use a compatible range or add --allow-sigma-extrapolation "
                "(at your own risk)."
            )
        print("WARNING: extrapolating the learned variance beyond training sigma range.")

    nu_train = float(cfg_t["nu"])
    nu_coeff = float(args.nu_coeff) if args.nu_coeff is not None else nu_train
    nu_init_ode = float(args.nu_init) if args.nu_init is not None else nu_train
    nu_init_sde = (float(args.nu_init_sde) if args.nu_init_sde is not None
                   else nu_coeff)
    if nu_coeff <= 2 or nu_init_ode <= 0 or nu_init_sde <= 0:
        raise ValueError("Require nu_coeff > 2 and nu_init > 0.")
    if abs(nu_coeff - nu_train) > 1e-8:
        print(f"WARNING: nu_coeff={nu_coeff} differs from training nu={nu_train}. "
              "For the cleanest comparison, use the same value.")

    model_g = None
    g_run_dir = None
    g_ckpt = None
    if "g_sde" in args.samplers:
        g_run_dir = (args.g_run_dir if args.g_run_dir is not None
                     else find_run_dir(runs_dir, args.train_seed, "gaussian_noise"))
        g_ckpt = args.g_ckpt if args.g_ckpt is not None else pick_latest_ckpt(g_run_dir)
        cfg_g = load_json(g_run_dir / "config.json")
        if int(cfg_g.get("train_seed", -1)) != args.train_seed:
            raise ValueError("Gaussian run uses a different train_seed.")
        model_g = load_gaussian_model(
            g_ckpt, cfg_g, device, args.g_denoiser_variant
        )
        if (str(cfg_g.get("data_pt_path", ""))
                != str(cfg_t.get("data_pt_path", ""))):
            print("WARNING: Gaussian and Student-t training data paths differ.")

    samples_root.mkdir(parents=True, exist_ok=True)
    output_dir = (samples_root /
                  f"{time.strftime('%Y-%m-%d_%H%M%S')}_trainseed{args.train_seed}")
    output_dir.mkdir(parents=True, exist_ok=False)
    print("device             :", device)
    print("variance checkpoint:", variance_ckpt)
    print("denoiser variant   :", cfg_t.get("denoiser_variant"))
    print("variance steps     :", variance_pack["variance_step"])
    print("training nu        :", nu_train)
    print("sampling nu_coeff  :", nu_coeff)
    print("sampling sigmas    :", (sigma_min, sigma_max))
    print("output directory   :", output_dir)

    metadata: dict = {
        "train_seed": args.train_seed,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "sample_batch_size": args.sample_batch_size,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "n_steps_ode": args.n_steps_ode,
        "n_steps_sde": args.n_steps_sde,
        "rho": args.rho,
        "nu_train": nu_train,
        "nu_init_ode": nu_init_ode,
        "nu_init_sde": nu_init_sde,
        "nu_coeff": nu_coeff,
        "space": "z",
        "variance_run_dir": str(variance_run_dir),
        "variance_ckpt": str(variance_ckpt),
        "variance_step": int(variance_pack["variance_step"]),
        "variance_model_type": cfg_t["variance_model_type"],
        "denoiser_variant": cfg_t.get("denoiser_variant"),
        "t_ckpt": str(variance_ckpt),
        "g_run_dir": str(g_run_dir) if g_run_dir is not None else None,
        "g_ckpt": str(g_ckpt) if g_ckpt is not None else None,
        "data_pt_path": str(cfg_t.get("data_pt_path", "")),
        "sample_files": {},
    }

    common_t = dict(
        model=model_t,
        nu_init=nu_init_sde,
        nu_coeff=nu_coeff,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        n_steps=args.n_steps_sde,
        rho=args.rho,
        device=device,
    )
    methods: dict[str, Callable[[int], np.ndarray]] = {
        "t_ode": lambda n: sample_tnet_ode_heun(
            model=model_t, nu_init=nu_init_ode, n_samples=n,
            sigma_max=sigma_max, sigma_min=sigma_min,
            n_steps=args.n_steps_ode, rho=args.rho, device=device,
        ),
        "t_sde": lambda n: sample_tnet_state_sde(
            **common_t, n_samples=n, coeff_dep_on=True,
        ),
        "t_sde_variance": lambda n: sample_tnet_state_sde_variance(
            **common_t, n_samples=n, variance_model=variance_model,
        ),
        "t_sde_coeff1": lambda n: sample_tnet_state_sde(
            **common_t, n_samples=n, coeff_dep_on=False,
        ),
    }
    if model_g is not None:
        methods["g_sde"] = lambda n: sample_gnet_sde_euler(
            model=model_g, n_samples=n, sigma_max=sigma_max,
            sigma_min=sigma_min, n_steps=args.n_steps_sde,
            rho=args.rho, device=device,
        )

    # Identical seeds for the two t-SDEs = common random numbers at equal batch size.
    seeds = {
        "t_ode": args.seed + 101,
        "t_sde": args.seed + 202,
        "t_sde_variance": args.seed + 202,
        "t_sde_coeff1": args.seed + 303,
        "g_sde": args.seed + 404,
    }
    for name in args.samplers:
        set_seed(seeds[name])
        print(f"Sampling {name}, random seed {seeds[name]}")
        samples = generate_samples_chunked(
            sampler_name=name, sampler_fn=methods[name],
            n_samples=args.n_samples,
            sample_batch_size=args.sample_batch_size,
        )
        if not np.isfinite(samples).all():
            raise RuntimeError(f"Sampler {name} returned nonfinite output.")
        output_path = output_dir / f"{name}.npy"
        save_samples(output_path, samples, args.force)
        metadata["sample_files"][name] = str(output_path)
        save_json(output_dir / "metadata.json", metadata)

    print(f"Saved metadata: {output_dir / 'metadata.json'}")
    print("Sampling complete.")


if __name__ == "__main__":
    main()
