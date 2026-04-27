"""Create QQ plots for generated samples in z-space.

This script evaluates generated samples saved by sample.py. It loads
metadata.json, reads the corresponding reference test data from the prepared
.pt file, optionally selects a fixed random subset of test samples, computes
common quantiles, and saves an overlay QQ plot.

The comparison is performed in z-space, i.e., the same normalized space used
for training and sampling.

Example:
    python qq_plot.py --sample-dir samples/2026-04-27_153000_trainseed4

    python qq_plot.py --sample-dir samples/2026-04-27_153000_trainseed4 \
        --n-quantiles 100000 --q-min 1e-4

    # Optional: use a fixed random subset of test samples.
    python qq_plot.py --sample-dir samples/2026-04-27_153000_trainseed4 \
        --n-test-samples 100000 --test-subsample-seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D


# Colorblind-friendly palette used in the paper figures.
COLORS = {
    "t_ode": "#0072B2",         # blue
    "t_sde": "#E69F00",         # orange/yellow
    "t_sde_coeff1": "#D55E00",  # vermillion
    "g_sde": "#009E73",         # green
}

MARKERS = {
    "t_ode": "o",
    "t_sde": "D",
    "t_sde_coeff1": "^",
    "g_sde": "s",
}

LABELS = {
    "g_sde": "VE-SDE",
    "t_ode": r"$t$-ODE",
    "t_sde": r"$t$-SDE",
    "t_sde_coeff1": r"Ablated $t$-SDE",
}

# Plotting order follows the original notebook:
# t-ODE, t-SDE, ablated t-SDE, VE-SDE.
PLOT_ORDER = ["t_ode", "t_sde", "t_sde_coeff1", "g_sde"]

# Legend order follows the original notebook:
# VE-SDE, t-ODE, t-SDE, ablated t-SDE, ideal.
LEGEND_ORDER = ["g_sde", "t_ode", "t_sde", "t_sde_coeff1"]

POINT_ALPHA = 0.8
MARKER_SIZE = 1.2
LINE_WIDTH = 1.2
LEGEND_FONT_SIZE = 10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create z-space QQ plots.")

    parser.add_argument(
        "--sample-dir",
        type=Path,
        required=True,
        help="Directory created by sample.py containing metadata.json and .npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saving figures. Default: <sample_dir>/figures.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="qq_overlay_zspace",
        help="Base filename for the output figure.",
    )
    parser.add_argument(
        "--n-quantiles",
        type=int,
        default=100_000,
        help="Number of quantile points.",
    )
    parser.add_argument(
        "--q-min",
        type=float,
        default=1e-4,
        help="Smallest quantile. The largest quantile is 1 - q_min.",
    )
    parser.add_argument(
        "--n-test-samples",
        type=int,
        default=None,
        help=(
            "Number of test samples used for QQ plots. "
            "Default: use all test samples."
        ),
    )
    parser.add_argument(
        "--test-subsample-seed",
        type=int,
        default=42,
        help=(
            "Random seed for selecting test samples without replacement. "
            "Used only when --n-test-samples is specified and smaller than "
            "the full test set size."
        ),
    )
    parser.add_argument(
        "--plot-step",
        type=int,
        default=1,
        help="Plot every plot_step-th quantile point.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(4.8, 4.8),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster outputs.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        choices=["png", "pdf", "svg"],
        help="Figure formats to save.",
    )
    parser.add_argument(
        "--save-quantiles",
        action="store_true",
        help="Save computed quantiles as a .npz file.",
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path_str: str, sample_dir: Path) -> Path:
    """Resolve paths stored in metadata.json.

    sample.py may store paths as absolute paths, project-root-relative paths,
    or sample-directory-relative paths. This function tries the common cases
    and returns the first existing path.
    """
    path = Path(path_str)

    if path.is_absolute():
        return path

    candidates = [
        (Path.cwd() / path).resolve(),
        (sample_dir / path).resolve(),
        (sample_dir.parent / path).resolve(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Return the project-root interpretation for a clear downstream error.
    return candidates[0]


def load_reference_test_z(metadata: dict, sample_dir: Path) -> np.ndarray:
    """Load the reference test samples in z-space from data_pt_path."""
    data_pt_path = metadata.get("data_pt_path", "")

    if data_pt_path == "":
        raise KeyError("metadata.json must contain a non-empty 'data_pt_path'.")

    pt_path = resolve_path(data_pt_path, sample_dir)

    if not pt_path.exists():
        raise FileNotFoundError(f"Reference .pt file not found: {pt_path}")

    pack = torch.load(pt_path, map_location="cpu")
    return pack["test"].numpy().astype(np.float32).reshape(-1)


def load_generated_samples(metadata: dict, sample_dir: Path) -> dict[str, np.ndarray]:
    """Load generated samples listed in metadata.json."""
    sample_files = metadata.get("sample_files", {})

    if not isinstance(sample_files, dict) or len(sample_files) == 0:
        raise KeyError("metadata.json must contain a non-empty 'sample_files' dictionary.")

    samples: dict[str, np.ndarray] = {}

    for name, path_str in sample_files.items():
        path = resolve_path(path_str, sample_dir)

        if not path.exists():
            raise FileNotFoundError(f"Sample file not found for {name}: {path}")

        samples[name] = np.load(path).astype(np.float32).reshape(-1)

    return samples


def subsample_test_samples(
    values: np.ndarray,
    n_samples: int | None,
    seed: int,
) -> np.ndarray:
    """Select a fixed random subset of test samples without replacement.

    If n_samples is None, all test samples are used.
    """
    if n_samples is None:
        return values

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if values.shape[0] < n_samples:
        raise ValueError(
            f"Requested {n_samples} test samples, "
            f"but only {values.shape[0]} are available."
        )

    if values.shape[0] == n_samples:
        return values

    rng = np.random.default_rng(seed)
    indices = rng.choice(values.shape[0], size=n_samples, replace=False)
    return values[indices]


def check_finite(name: str, values: np.ndarray) -> None:
    """Check that all values are finite."""
    if not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"{name} contains {n_bad} non-finite values.")


def compute_quantiles(
    reference: np.ndarray,
    samples: dict[str, np.ndarray],
    n_quantiles: int,
    q_min: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Compute reference and generated quantiles on a common grid."""
    if not 0.0 < q_min < 0.5:
        raise ValueError("q_min must satisfy 0 < q_min < 0.5.")

    if n_quantiles <= 1:
        raise ValueError("n_quantiles must be greater than 1.")

    q = np.linspace(q_min, 1.0 - q_min, n_quantiles)
    ref_q = np.quantile(reference, q)
    sample_q = {
        name: np.quantile(values, q)
        for name, values in samples.items()
    }

    return q, ref_q, sample_q


def save_quantiles(
    path: Path,
    q: np.ndarray,
    ref_q: np.ndarray,
    sample_q: dict[str, np.ndarray],
) -> None:
    """Save computed quantiles to a .npz file."""
    payload = {
        "q": q,
        "test_q": ref_q,
    }

    for name, values in sample_q.items():
        payload[f"{name}_q"] = values

    np.savez_compressed(path, **payload)
    print(f"Saved quantiles: {path}")


def plot_qq_overlay(
    ref_q: np.ndarray,
    sample_q: dict[str, np.ndarray],
    output_base: Path,
    formats: list[str],
    plot_step: int,
    figsize: tuple[float, float],
    dpi: int,
) -> None:
    """Create and save an overlay QQ plot."""
    if plot_step <= 0:
        raise ValueError("plot_step must be positive.")

    available_names = [name for name in PLOT_ORDER if name in sample_q]

    if len(available_names) == 0:
        raise ValueError("No recognized sample names are available for plotting.")

    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    all_quantiles = [ref_q]

    for name in available_names:
        values = sample_q[name]
        all_quantiles.append(values)

        ax.plot(
            ref_q[::plot_step],
            values[::plot_step],
            MARKERS[name],
            markersize=MARKER_SIZE,
            alpha=POINT_ALPHA,
            color=COLORS[name],
            rasterized=True,
        )

    lo = min(float(values.min()) for values in all_quantiles)
    hi = max(float(values.max()) for values in all_quantiles)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=LINE_WIDTH)

    ax.set_xlabel("Test sample quantiles", fontsize=12)
    ax.set_ylabel("Generated sample quantiles", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)

    legend_handles = []

    for name in LEGEND_ORDER:
        if name not in sample_q:
            continue

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=MARKERS[name],
                linestyle="None",
                color=COLORS[name],
                alpha=1.0,
                markersize=7,
                label=LABELS[name],
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="black",
            linewidth=LINE_WIDTH,
            label="ideal",
        )
    )

    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.2,
        handletextpad=0.5,
    )

    fig.tight_layout()

    for fmt in formats:
        out_path = output_base.with_suffix(f".{fmt}")

        if fmt == "png":
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)

        print(f"Saved figure: {out_path}")

    plt.close(fig)


def main() -> None:
    """Run QQ-plot evaluation."""
    args = parse_args()

    sample_dir = args.sample_dir.resolve()
    metadata_path = sample_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

    output_dir = args.output_dir if args.output_dir is not None else sample_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(metadata_path)

    space = metadata.get("space", "z")
    if space != "z":
        raise ValueError(
            f"This script expects z-space samples, but metadata space is '{space}'."
        )

    reference_all = load_reference_test_z(metadata, sample_dir)
    reference = subsample_test_samples(
        reference_all,
        n_samples=args.n_test_samples,
        seed=args.test_subsample_seed,
    )

    samples = load_generated_samples(metadata, sample_dir)

    check_finite("reference", reference)
    for name, values in samples.items():
        check_finite(name, values)

    print(f"SAMPLE_DIR          : {sample_dir}")
    print(f"OUTPUT_DIR          : {output_dir.resolve()}")
    print(f"reference all       : shape={reference_all.shape}")
    print(f"reference used      : shape={reference.shape}")

    if args.n_test_samples is None or reference_all.shape[0] == reference.shape[0]:
        print("test subset         : all test samples")
    else:
        print(f"test subsample seed : {args.test_subsample_seed}")

    for name, values in samples.items():
        print(f"{name:18s}: shape={values.shape}")

    q, ref_q, sample_q = compute_quantiles(
        reference=reference,
        samples=samples,
        n_quantiles=args.n_quantiles,
        q_min=args.q_min,
    )

    if args.save_quantiles:
        save_quantiles(
            output_dir / f"{args.output_name}_quantiles.npz",
            q,
            ref_q,
            sample_q,
        )

    plot_qq_overlay(
        ref_q=ref_q,
        sample_q=sample_q,
        output_base=output_dir / args.output_name,
        formats=args.formats,
        plot_step=args.plot_step,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )

    print("QQ-plot evaluation complete.")


if __name__ == "__main__":
    main()
