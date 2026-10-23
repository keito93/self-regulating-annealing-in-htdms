"""Oracle conditional-moment sampling for the 1D Student-t(3) HTDM experiment.

Separate file: does not edit sample.py, run.py, or sample_variance.py.

The data .pt file contains z=(raw - train_raw_mean)/train_raw_std.  Therefore
p_data(z) is StudentT(df=3, loc=-mean/std, scale=1/std), NOT StudentT(3,0,1).
The forward noise is StudentT(df=3, scale=sigma).

Oracle targets:
    m(x,sigma) = E[z0 | x_sigma=x]
    K(x,sigma) = E[((x-z0)/sigma)^2 | x_sigma=x]

Both have exact rational closed forms when BOTH Student-t degrees of freedom
are 3. This script also implements direct scipy quadrature to verify the
closed forms, and a very slow --backend quadrature sampling mode for tiny tests.
Use --backend closed_form for production (posterior moments are evaluated
without learning or Monte Carlo integration).

Example (Google Colab, in project directory):
    python3 sample_oracle.py --train-seed 0 --n-samples 1000000 \\
        --sample-batch-size 50000 --n-steps-sde 128 --samples-dir samples_oracle

For full oracle with exact forward marginal at sigma_max:
    python3 sample_oracle.py --train-seed 0 --samplers full_oracle \\
        --init-mode exact --n-samples 100000 --n-steps-sde 256

Direct adaptive quadrature sanity test (slow; keep very small):
    python3 sample_oracle.py --train-seed 0 --samplers full_oracle \\
        --backend quadrature --init-mode exact --n-samples 12 \\
        --sample-batch-size 12 --n-steps-sde 20
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import StudentT
from tqdm.auto import tqdm

# Exact schedule and denoiser architecture of the existing repository.
from sample import load_model_from_ckpt, sigma_schedule_karras

MODES = (
    "approx",             # learned D and approximated K
    "oracle_k",           # learned D and exact K
    "oracle_mean_approx", # exact m and approximated K
    "full_oracle",        # exact m and exact K
)


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--data-pt", type=Path, default=None)
    p.add_argument("--t-run-dir", type=Path, default=None)
    p.add_argument("--t-ckpt", type=Path, default=None)
    p.add_argument("--samples-dir", type=Path, default=Path("samples_oracle"))
    p.add_argument("--samplers", nargs="+", choices=MODES,
                   default=["approx", "oracle_k", "oracle_mean_approx", "full_oracle"])
    p.add_argument("--n-samples", type=int, default=100_000)
    p.add_argument("--sample-batch-size", type=int, default=20_000)
    p.add_argument("--n-steps-sde", type=int, default=128)
    p.add_argument("--sigma-max", type=float, default=20.0)
    p.add_argument("--sigma-min", type=float, default=0.002)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--init-mode", choices=("approx", "exact"), default="approx",
                   help="approx: sigma_max * t3; exact: z0 + sigma_max * t3")
    p.add_argument("--backend", choices=("closed_form", "quadrature"),
                   default="closed_form",
                   help="Quadrature evaluates integrals separately for every particle and time step; extremely slow")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-quadrature-check", action="store_true")
    p.add_argument("--force", action="store_true")
    return p


def exact_t3_moments_torch(x: torch.Tensor, sigma: torch.Tensor,
                           loc: float, scale: float):
    """Closed-form evaluation of the exact 1D posterior mean and noise second moment.

    Let a=sqrt(3)*scale, b=sqrt(3)*sigma, A=a/(a+b), B=b/(a+b),
    r=(x-loc)/(a+b), H=(1+AB)+(1-3AB)*r**2. Then
      m=loc+(x-loc)*A**2*(A+4*B+A*r**2)/H
      K=3*(A**2+(1+3AB)*r**2+B*r**4)/H
    This is the exact evaluation of the posterior integrals for t3/t3.
    """
    a = math.sqrt(3.0) * scale
    b = math.sqrt(3.0) * sigma
    c = a + b
    A = a / c
    B = b / c
    r = (x - loc) / c
    r2 = r.square()
    ab = A * B
    H = (1.0 + ab) + (1.0 - 3.0 * ab) * r2
    m = loc + (x - loc) * A.square() * (A + 4.0 * B + A * r2) / H
    K = 3.0 * (A.square() + (1.0 + 3.0 * ab) * r2 + B * r2.square()) / H
    return m, K


def t3_pdf_scalar(x: float, loc: float, scale: float) -> float:
    """Normalized Student-t(df=3) density, with arbitrary location and scale."""
    a = math.sqrt(3.0) * scale
    return 2.0 * a ** 3 / (math.pi * ((x - loc) ** 2 + a * a) ** 2)


def integral_moments_quad(x: float, sigma: float, loc: float, scale: float):
    """Direct adaptive numerical integrals over z0 in (-inf,inf).

    The integration interval is split near BOTH posterior peaks: z0=loc
    and z0=x. This avoids missing a narrow likelihood peak for small sigma.
    Suitable as a trusted numerical check or tiny sampling experiments only.
    """
    from scipy.integrate import quad

    x = float(x)
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    breaks = set()
    for center, width in ((loc, scale), (x, sigma)):
        for v in (-25.0, -5.0, -1.0, 0.0, 1.0, 5.0, 25.0):
            breaks.add(float(center + v * width))
    breaks = sorted(breaks)
    intervals = list(zip([-math.inf] + breaks, breaks + [math.inf]))

    def p_joint(z):
        return (t3_pdf_scalar(z, loc, scale)
                * t3_pdf_scalar(x - z, 0.0, sigma))

    def piecewise_quad(f):
        total = 0.0
        for low, high in intervals:
            val, _ = quad(f, low, high, epsabs=1e-15, epsrel=1e-10, limit=120)
            total += val
        return total

    p = piecewise_quad(p_joint)
    if not (math.isfinite(p) and p > 0):
        raise FloatingPointError(f"Invalid posterior normalizer at x={x}, sigma={sigma}: {p}")
    mean = piecewise_quad(lambda z: z * p_joint(z)) / p
    K = piecewise_quad(lambda z: ((x - z) / sigma) ** 2 * p_joint(z)) / p
    if not (math.isfinite(mean) and math.isfinite(K) and K >= 0):
        raise FloatingPointError(f"Invalid numerical posterior moment at x={x}, sigma={sigma}")
    return mean, K


def evaluate_oracle(x: torch.Tensor, sigma: torch.Tensor, loc: float,
                    scale: float, backend: str):
    if backend == "closed_form":
        return exact_t3_moments_torch(x, sigma, loc, scale)
    if backend == "quadrature":
        # The only purpose of this branch is literal quadrature sampling for
        # a *tiny* number of samples. It is not used for production.
        sigma_value = float(sigma.item())
        arr = x.detach().cpu().numpy().reshape(-1)
        result = [integral_moments_quad(z, sigma_value, loc, scale) for z in arr]
        mean = torch.tensor([r[0] for r in result], device=x.device, dtype=x.dtype)
        K = torch.tensor([r[1] for r in result], device=x.device, dtype=x.dtype)
        return mean.reshape_as(x), K.reshape_as(x)
    raise ValueError(backend)


def verify_quadrature(loc: float, scale: float) -> None:
    """Compare analytic evaluation against independent adaptive quadrature."""
    print("\nValidating closed forms with scipy.integrate.quad...")
    test_points = [
        (0.0, 0.002), (1.0, 0.002), (10.0, 0.002),
        (0.0, 0.1), (3.0, 0.1), (10.0, 1.0),
        (30.0, 1.0), (0.0, 20.0), (50.0, 20.0),
    ]
    max_mean_error, max_k_rel_error = 0.0, 0.0
    for x, s in test_points:
        X = torch.tensor([[x]], dtype=torch.float64)
        S = torch.tensor(float(s), dtype=torch.float64)
        m, k = exact_t3_moments_torch(X, S, loc, scale)
        mq, kq = integral_moments_quad(x, s, loc, scale)
        max_mean_error = max(max_mean_error, abs(float(m.item()) - mq))
        max_k_rel_error = max(max_k_rel_error, abs(float(k.item()) - kq) / max(1e-14, abs(kq)))
    print(f"  max abs mean error: {max_mean_error:.3e}")
    print(f"  max rel K error  : {max_k_rel_error:.3e}")
    if max_mean_error > 1e-5 or max_k_rel_error > 1e-5:
        raise RuntimeError("Closed-form / quadrature verification failed")


def find_original_t_run(runs_dir: Path, train_seed: int) -> Path:
    """Select original t-denoiser runs; never accidentally select variance runs."""
    candidates = []
    for directory in runs_dir.iterdir():
        if not directory.is_dir() or not (directory / "config.json").exists():
            continue
        cfg = json.loads((directory / "config.json").read_text())
        if (int(cfg.get("train_seed", -1)) == train_seed
                and cfg.get("noise_mode") == "student_t"
                and not cfg.get("variance_only", False)
                and list((directory / "checkpoints").glob("step*.pt"))):
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(f"No original trained Student-t run for seed {train_seed} in {runs_dir}")
    return max(candidates, key=lambda p: p.name)


def load_test_pack(path: Path, train_seed: int):
    pack = torch.load(path, map_location="cpu", weights_only=False)
    if int(pack.get("train_seed", -1)) != train_seed:
        raise ValueError(f"Train seed mismatch in data file {path}")
    if abs(float(pack.get("nu", -1)) - 3.0) > 1e-8:
        raise ValueError("Oracle formula requires Student-t data with nu=3")
    raw_mu = float(torch.as_tensor(pack["train_raw_mean"]).reshape(-1)[0])
    raw_std = float(torch.as_tensor(pack["train_raw_std"]).reshape(-1)[0])
    if raw_std <= 0:
        raise ValueError("Invalid train_raw_std")
    # data.py uses raw StudentT(3,loc=0,scale=1)
    loc = -raw_mu / raw_std
    scale = 1.0 / raw_std
    return pack, loc, scale


def check_finite(value: torch.Tensor, name: str, step: int, sigma: torch.Tensor):
    if not torch.isfinite(value).all().item():
        mask = torch.isfinite(value)
        print(f"Nonfinite {name}: step={step} sigma={sigma.item():.5g} "
              f"bad={int((~mask).sum().item())}/{value.numel()}")
        raise RuntimeError(f"Nonfinite {name} at step {step}")


@torch.no_grad()
def sample_sde(mode: str, denoiser, n_samples: int, sigmas: torch.Tensor,
               init_mode: str, loc: float, scale: float, backend: str,
               device: torch.device) -> np.ndarray:
    t3 = StudentT(df=torch.tensor(3.0, device=device))
    if init_mode == "approx":
        x = t3.sample((n_samples, 1)) * sigmas[0]
    elif init_mode == "exact":
        x = loc + scale * t3.sample((n_samples, 1)) + sigmas[0] * t3.sample((n_samples, 1))
    else:
        raise ValueError(init_mode)

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        next_sigma = sigmas[i + 1]
        ds = next_sigma - sigma

        if mode in ("oracle_k", "full_oracle", "oracle_mean_approx"):
            m_exact, k_exact = evaluate_oracle(x, sigma, loc, scale, backend)
        else:
            m_exact = k_exact = None

        if mode in ("approx", "oracle_k"):
            if denoiser is None:
                raise RuntimeError(f"Mode {mode} needs a trained denoiser")
            D = denoiser(x, sigma.expand_as(x))
        else:
            D = m_exact
        check_finite(D, "denoiser_or_oracle_mean", i, sigma)

        if mode in ("oracle_k", "full_oracle"):
            K = k_exact
        else:
            K = ((x - D) / sigma).square()
        check_finite(K, "conditional_second_moment", i, sigma)
        if not torch.all(K >= 0).item():
            raise RuntimeError(f"Negative K at step {i}")

        drift = 2.0 * (x - D) / sigma
        # Exactly matches the discretization in sample.py.
        coeff = torch.sqrt((3.0 + K) / 2.0)
        noise_std = (sigma.square() - next_sigma.square()).clamp_min(0).sqrt()
        x = x + drift * ds + noise_std * coeff * torch.randn_like(x)
        check_finite(x, "state", i, sigma)

    sigma_final = sigmas[-1]
    if mode in ("approx", "oracle_k"):
        result = denoiser(x, sigma_final.expand_as(x))
    else:
        result, _ = evaluate_oracle(x, sigma_final, loc, scale, backend)
    check_finite(result, "final_denoised", len(sigmas) - 1, sigma_final)
    return result.squeeze(1).detach().cpu().numpy().astype(np.float32)


def main():
    args = args_parser().parse_args()
    if not (args.n_samples > 0 and args.sample_batch_size > 0
            and args.n_steps_sde >= 2 and 0 < args.sigma_min < args.sigma_max
            and args.rho > 0):
        raise ValueError("Invalid sampling parameters")
    if len(set(args.samplers)) != len(args.samplers):
        raise ValueError("Duplicate samplers")
    if args.backend == "quadrature" and args.n_samples * args.n_steps_sde > 5000:
        raise ValueError("Direct quadrature is too slow at this scale. "
                         "Use --backend closed_form for production; "
                         "the analytic expression is independently verified by quadrature.")

    root = Path(__file__).resolve().parent
    runs_dir = args.runs_dir or root / "runs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    needs_denoiser = any(m in ("approx", "oracle_k") for m in args.samplers)
    cfg = None
    model = None
    ckpt = None
    run_dir = None
    if needs_denoiser:
        run_dir = args.t_run_dir or find_original_t_run(runs_dir, args.train_seed)
        cfg = json.loads((run_dir / "config.json").read_text())
        if int(cfg.get("train_seed", -1)) != args.train_seed:
            raise ValueError("Train seed mismatch in denoiser config")
        if cfg.get("noise_mode") != "student_t":
            raise ValueError("Expected Student-t denoiser")
        if abs(float(cfg.get("nu", 3.0)) - 3.0) > 1e-8:
            raise ValueError("Oracle experiment requires nu=3 in denoiser training")
        ckpt = args.t_ckpt
        if ckpt is None:
            checkpoints = sorted((run_dir / "checkpoints").glob("step*.pt"))
            ckpt = next((p for p in checkpoints[::-1]
                         if int(p.stem[4:]) > 0), None)
            if ckpt is None:
                raise FileNotFoundError(f"No trained denoiser checkpoint in {run_dir}")
        model = load_model_from_ckpt(
            ckpt, alpha=float(cfg["alpha_resolved"]),
            hidden=int(cfg.get("hidden", 128)), device=device
        ).eval().requires_grad_(False)

    if args.data_pt is not None:
        data_path = args.data_pt
    elif cfg is not None and cfg.get("data_pt_path"):
        data_path = Path(cfg["data_pt_path"])
    else:
        candidates = sorted((root / "data" / "normalized_pt").glob(
            f"student_t1d_nu3_norm_trainseed{args.train_seed}_testseed*_Ntrain*_Ntest*.pt"
        ))
        if not candidates:
            raise FileNotFoundError("No normalized Student-t(3) data .pt found")
        data_path = candidates[-1]
    if not data_path.is_absolute() and not data_path.exists():
        data_path = root / data_path
    pack, loc, scale = load_test_pack(data_path, args.train_seed)

    if not args.skip_quadrature_check:
        verify_quadrature(loc, scale)

    output_root = args.samples_dir
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = (output_root /
                  f"{time.strftime('%Y-%m-%d_%H%M%S')}_trainseed{args.train_seed}_{args.init_mode}")
    output_dir.mkdir(parents=True, exist_ok=args.force)

    metadata = {
        "train_seed": args.train_seed,
        "n_samples": args.n_samples,
        "sample_batch_size": args.sample_batch_size,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "rho": args.rho,
        "n_steps_sde": args.n_steps_sde,
        "nu_data": 3,
        "nu_noise": 3,
        "nu_coeff": 3,
        "nu_init": 3,
        "init_mode": args.init_mode,
        "oracle_backend": args.backend,
        "seed": args.seed,
        "data_pt_path": str(data_path),
        "normalized_data_loc": loc,
        "normalized_data_scale": scale,
        "train_raw_mean": float(torch.as_tensor(pack["train_raw_mean"]).reshape(-1)[0]),
        "train_raw_std": float(torch.as_tensor(pack["train_raw_std"]).reshape(-1)[0]),
        "t_run_dir": str(run_dir) if run_dir else None,
        "t_ckpt": str(ckpt) if ckpt else None,
        "samplers": list(args.samplers),
        "sample_files": {},
    }
    sigmas = sigma_schedule_karras(
        args.n_steps_sde, args.sigma_min, args.sigma_max, args.rho, device
    )
    print("device            :", device)
    print("normalized Student-t(3) parameters (loc, scale):", loc, scale)
    print("denoiser ckpt     :", ckpt)
    print("init              :", args.init_mode)
    print("backend           :", args.backend)
    print("output            :", output_dir)
    print("SDE modes         :", args.samplers)

    for mode in args.samplers:
        # Same seed and same underlying random draws for fair comparisons.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        output = []
        progress = tqdm(total=args.n_samples, desc=f"sample {mode}")
        for start in range(0, args.n_samples, args.sample_batch_size):
            n = min(args.sample_batch_size, args.n_samples - start)
            chunk = sample_sde(mode, model, n, sigmas, args.init_mode,
                               loc, scale, args.backend, device)
            output.append(chunk)
            progress.update(n)
        progress.close()
        array = np.concatenate(output)
        if not np.isfinite(array).all():
            raise RuntimeError(f"Nonfinite samples from {mode}")
        path = output_dir / f"{mode}.npy"
        if path.exists() and not args.force:
            raise FileExistsError(path)
        np.save(path, array)
        metadata["sample_files"][mode] = str(path)
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print("saved", path, array.shape)

    print("All sampling outputs saved in", output_dir)


if __name__ == "__main__":
    main()
