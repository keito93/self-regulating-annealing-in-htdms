"""Numerical demo of self-regulating annealing in HTDMs.

This script reproduces the fixed-point plot and trajectory plot used to
illustrate the self-regulating annealing mechanism for the symmetric two-point
data distribution.

It generates:
    - fixedpoint.pdf
    - traj.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Numerical demo of self-regulating annealing."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--a", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=3.0)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
        }
    )


def exp_gamma(y: np.ndarray, gamma: float) -> np.ndarray:
    """Generalized exponential with domain handling."""
    y = np.asarray(y, dtype=float)
    base = 1.0 + gamma * y

    out = np.full_like(y, np.nan, dtype=float)
    valid = base > 0.0
    out[valid] = np.power(base[valid], 1.0 / gamma)
    return out


def tanh_gamma(y: np.ndarray, gamma: float) -> np.ndarray:
    """Generalized hyperbolic tangent."""
    y = np.asarray(y, dtype=float)

    ey = exp_gamma(y, gamma)
    emy = exp_gamma(-y, gamma)

    out = np.full_like(y, np.nan, dtype=float)
    valid = np.isfinite(ey) & np.isfinite(emy) & ((ey + emy) != 0.0)
    out[valid] = (ey[valid] - emy[valid]) / (ey[valid] + emy[valid])
    return out


def beta_t(x: np.ndarray, t: float, a: float, sigma: float, nu: float, d: int) -> np.ndarray:
    """State-dependent inverse-temperature factor."""
    sigma2 = sigma**2
    return (nu + d) / (nu * sigma2 * t + x**2 + a**2)


def denoiser_target(
    x: np.ndarray,
    t: float,
    a: float,
    sigma: float,
    nu: float,
    d: int,
    gamma: float,
) -> np.ndarray:
    """Ideal denoiser E[x0 | xt] for the symmetric two-point distribution."""
    b = beta_t(x, t, a=a, sigma=sigma, nu=nu, d=d)
    y = b * a * x
    return a * tanh_gamma(y, gamma)


def drift_and_alpha(
    x: float,
    t: float,
    a: float,
    sigma: float,
    nu: float,
    d: int,
    gamma: float,
) -> tuple[float, float]:
    """Return drift and state-dependent coefficient alpha."""
    x_arr = np.array([x], dtype=float)
    target = denoiser_target(x_arr, t, a=a, sigma=sigma, nu=nu, d=d, gamma=gamma)[0]

    drift = (x - target) / t

    # For sigma_t = sigma * sqrt(t), Delta_t^2 = |x - target|^2 / (sigma^2 t).
    delta2 = (x - target) ** 2 / (sigma**2 * t)
    alpha = np.sqrt((nu + delta2) / (nu + d - 2.0))

    return float(drift), float(alpha)


def simulate_trajectory(
    x_init: float,
    dW: np.ndarray,
    t_grid: np.ndarray,
    a: float,
    sigma: float,
    nu: float,
    d: int,
    gamma: float,
    use_state_dependent_alpha: bool,
) -> np.ndarray:
    """Simulate the one-dimensional SDE trajectory."""
    x = np.zeros_like(t_grid)
    x[0] = x_init

    dt = t_grid[1] - t_grid[0]

    for j in range(len(t_grid) - 1):
        t = float(t_grid[j])
        drift, alpha = drift_and_alpha(
            x=float(x[j]),
            t=t,
            a=a,
            sigma=sigma,
            nu=nu,
            d=d,
            gamma=gamma,
        )

        if not use_state_dependent_alpha:
            alpha = 1.0

        x[j + 1] = x[j] + drift * dt + alpha * sigma * dW[j]

    return x


def find_fixed_points(
    x_grid: np.ndarray,
    t: float,
    a: float,
    sigma: float,
    nu: float,
    d: int,
    gamma: float,
) -> np.ndarray:
    """Find fixed points of x = E[x0 | xt] by sign changes."""
    target = denoiser_target(x_grid, t, a=a, sigma=sigma, nu=nu, d=d, gamma=gamma)
    h = x_grid - target

    valid = np.isfinite(h)
    xs = x_grid[valid]
    hs = h[valid]

    roots: list[float] = []

    # Include exact zeros if they appear on the grid.
    exact = np.where(hs == 0.0)[0]
    for idx in exact:
        roots.append(float(xs[idx]))

    # Detect sign changes.
    signs = np.sign(hs)
    idxs = np.where(signs[:-1] * signs[1:] < 0.0)[0]

    for idx in idxs:
        x1, x2 = xs[idx], xs[idx + 1]
        h1, h2 = hs[idx], hs[idx + 1]

        root = x1 - h1 * (x2 - x1) / (h2 - h1)
        roots.append(float(root))

    if len(roots) == 0:
        return np.array([], dtype=float)

    roots = np.array(sorted(roots), dtype=float)

    # Remove near-duplicates.
    unique_roots = [roots[0]]
    for root in roots[1:]:
        if abs(root - unique_roots[-1]) > 1e-4:
            unique_roots.append(root)

    return np.array(unique_roots, dtype=float)


def plot_fixed_points(
    output_path: Path,
    a: float,
    sigma: float,
    nu: float,
    d: int,
    gamma: float,
    t0: float = 1.0,
) -> None:
    """Plot the fixed-point equation."""
    colors = {
        "blue": "#0072B2",
        "orange": "#E69F00",
    }

    x_grid = np.linspace(-1.2 * a, 1.2 * a, 20001)
    lhs = x_grid
    rhs = denoiser_target(x_grid, t0, a=a, sigma=sigma, nu=nu, d=d, gamma=gamma)
    fixed_points = find_fixed_points(
        x_grid, t=t0, a=a, sigma=sigma, nu=nu, d=d, gamma=gamma
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x_grid, lhs, linewidth=2.5, color=colors["blue"], label=r"$y=x$")
    ax.plot(
        x_grid,
        rhs,
        linewidth=2.5,
        color=colors["orange"],
        label=r"$a\,\tanh_\gamma(\beta_t(x) a x)$",
    )

    for i, x_fp in enumerate(fixed_points):
        ax.plot(
            x_fp,
            x_fp,
            "o",
            markersize=9,
            color="black",
            label="fixed points" if i == 0 else None,
        )

    ax.axhline(0.0, linestyle=":", linewidth=1.0, color="0.5")
    ax.axvline(0.0, linestyle=":", linewidth=1.0, color="0.5")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel("value")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(
    output_path: Path,
    a: float,
    sigma: float,
    nu: float,
    d: int,
    gamma: float,
    n_steps: int,
    seed: int,
) -> None:
    """Plot sample trajectories with and without the state-dependent alpha."""
    colors = {
        0.0: "#0072B2",  # blue
        4.0: "#E69F00",  # orange
    }

    t_grid = np.linspace(1.0, 0.0, n_steps + 1)
    dt = t_grid[1] - t_grid[0]

    rng = np.random.default_rng(seed)
    dW = np.sqrt(abs(dt)) * rng.standard_normal(n_steps)

    x_init_list = [0.0, 4.0]

    fig, ax = plt.subplots(figsize=(10, 5))

    for x_init in x_init_list:
        x_with = simulate_trajectory(
            x_init=x_init,
            dW=dW,
            t_grid=t_grid,
            a=a,
            sigma=sigma,
            nu=nu,
            d=d,
            gamma=gamma,
            use_state_dependent_alpha=True,
        )

        x_without = simulate_trajectory(
            x_init=x_init,
            dW=dW,
            t_grid=t_grid,
            a=a,
            sigma=sigma,
            nu=nu,
            d=d,
            gamma=gamma,
            use_state_dependent_alpha=False,
        )

        color = colors[x_init]

        ax.plot(
            t_grid,
            x_with,
            linewidth=2.0,
            color=color,
            linestyle="-",
            label=rf"$x_0={x_init:g}$",
        )

        ax.plot(
            t_grid,
            x_without,
            linewidth=2.0,
            color=color,
            linestyle="--",
            label=rf"$x_0={x_init:g}$ ($\alpha \equiv 1$)",
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x(t)$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_matplotlib()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gamma = -2.0 / (args.nu + args.d)

    plot_fixed_points(
        output_path=args.output_dir / "fixedpoint.pdf",
        a=args.a,
        sigma=args.sigma,
        nu=args.nu,
        d=args.d,
        gamma=gamma,
        t0=1.0,
    )

    plot_trajectories(
        output_path=args.output_dir / "traj.pdf",
        a=args.a,
        sigma=args.sigma,
        nu=args.nu,
        d=args.d,
        gamma=gamma,
        n_steps=args.n_steps,
        seed=args.seed,
    )

    print(f"Saved figures to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()            nu=nu,
            d=d,
            gamma=gamma,
            use_state_dependent_alpha=True,
        )

        x_without = simulate_trajectory(
            x_init=x_init,
            dW=dW,
            t_grid=t_grid,
            a=a,
            sigma=sigma,
            nu=nu,
            d=d,
            gamma=gamma,
            use_state_dependent_alpha=False,
        )

        color = colors[x_init]

        ax.plot(
            t_grid,
            x_with,
            linewidth=2.0,
            color=color,
            linestyle="-",
            label=rf"$x_0={x_init:g}$",
        )

        ax.plot(
            t_grid,
            x_without,
            linewidth=2.0,
            color=color,
            linestyle="--",
            label=rf"$x_0={x_init:g}$ ($\alpha \equiv 1$)",
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x(t)$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(
    output_path: Path,
    a: float,
    sigma: float,
    nu: float,
    d: int,
    gamma: float,
    n_steps: int,
    seed: int,
) -> None:
    """Plot sample trajectories with and without the state-dependent alpha."""
    colors = ["#0072B2", "#E69F00", "#CC79A7", "#56B4E9"]

    t_grid = np.linspace(1.0, 0.0, n_steps + 1)
    dt = t_grid[1] - t_grid[0]

    rng = np.random.default_rng(seed)
    dW = np.sqrt(abs(dt)) * rng.standard_normal(n_steps)

    x_init_list = [0.0, 4.0]

    fig, ax = plt.subplots(figsize=(10, 5))

    color_index = 0

    for x_init in x_init_list:
        x_with = simulate_trajectory(
            x_init=x_init,
            dW=dW,
            t_grid=t_grid,
            a=a,
            sigma=sigma,
            nu=nu,
            d=d,
            gamma=gamma,
            use_state_dependent_alpha=True,
        )

        x_without = simulate_trajectory(
            x_init=x_init,
            dW=dW,
            t_grid=t_grid,
            a=a,
            sigma=sigma,
            nu=nu,
            d=d,
            gamma=gamma,
            use_state_dependent_alpha=False,
        )

        c_with = colors[color_index % len(colors)]
        color_index += 1
        c_without = colors[color_index % len(colors)]
        color_index += 1

        ax.plot(
            t_grid,
            x_with,
            linewidth=2.0,
            color=c_with,
            label=rf"$x_0={x_init:g}$",
        )

        ax.plot(
            t_grid,
            x_without,
            linewidth=2.0,
            linestyle="--",
            color=c_without,
            label=rf"$x_0={x_init:g}$ ($\alpha \equiv 1$)",
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x(t)$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_matplotlib()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gamma = -2.0 / (args.nu + args.d)

    plot_fixed_points(
        output_path=args.output_dir / "fixedpoint.pdf",
        a=args.a,
        sigma=args.sigma,
        nu=args.nu,
        d=args.d,
        gamma=gamma,
        t0=1.0,
    )

    plot_trajectories(
        output_path=args.output_dir / "traj.pdf",
        a=args.a,
        sigma=args.sigma,
        nu=args.nu,
        d=args.d,
        gamma=gamma,
        n_steps=args.n_steps,
        seed=args.seed,
    )

    print(f"Saved figures to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
