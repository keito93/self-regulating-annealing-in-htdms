"""Prepare one-dimensional Student-t datasets for experiments.

This script generates raw train/test samples from a standard Student-t distribution,
normalizes them using training-set statistics only,
and saves PyTorch .pt files for downstream training and evaluation.

Default settings reproduce the data configuration used in the experiments:
    - Student-t degrees of freedom: nu = 3
    - Train size: 1,000,000
    - Test size: 1,000,000
    - Train seeds: 0, 1, 2, 3, 4
    - Test seed: 123

Example:
    python data.py

    python data.py --nu 3.0 --n-train 1000000 --n-test 1000000 \
        --train-seeds 0 1 2 3 4 --test-seed 123
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.distributions import StudentT
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and normalize one-dimensional Student-t datasets."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory for saving generated datasets. Default: <project_root>/data.",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=3.0,
        help="Degrees of freedom of the Student-t distribution.",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=1_000_000,
        help="Number of training samples per seed.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1_000_000,
        help="Number of test samples.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Number of samples generated at once.",
    )
    parser.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds for generating training datasets.",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=123,
        help="Random seed for generating the fixed test dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset files.",
    )

    return parser.parse_args()


def get_project_root() -> Path:
    """Return the project root directory.

    If this script is executed from a notebooks/ directory, the parent directory
    is treated as the project root.
    """
    project_root = Path.cwd()
    if project_root.name == "notebooks":
        project_root = project_root.parent
    return project_root


def get_device() -> torch.device:
    """Return CPU device for reproducible data generation."""
    return torch.device("cpu")


def format_nu_for_filename(nu: float) -> str:
    """Return a filesystem-friendly string for nu."""
    return f"{nu:g}".replace("-", "m").replace(".", "p")


def sample_student_t_1d_chunked(
    n_samples: int,
    chunk_size: int,
    nu: float,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Sample one-dimensional data from a standard Student-t distribution.

    Samples are generated in chunks to reduce peak memory usage. The output has
    shape (n_samples, 1) and dtype float32.

    Args:
        n_samples: Number of samples to generate.
        chunk_size: Number of samples generated in each chunk.
        nu: Degrees of freedom of the Student-t distribution.
        device: Device used for sampling.
        seed: Random seed for PyTorch sampling.

    Returns:
        A NumPy array of shape (n_samples, 1).
    """
    if n_samples % chunk_size != 0:
        raise ValueError(
            f"n_samples must be divisible by chunk_size: "
            f"n_samples={n_samples}, chunk_size={chunk_size}"
        )

    # StudentT.sample uses the PyTorch random number generator.
    torch.manual_seed(seed)

    distribution = StudentT(
        df=torch.tensor(float(nu), device=device),
        loc=torch.tensor(0.0, device=device),
        scale=torch.tensor(1.0, device=device),
    )

    chunks = []
    n_chunks = n_samples // chunk_size

    with torch.no_grad():
        for _ in tqdm(range(n_chunks), desc=f"sampling seed={seed}"):
            samples = distribution.sample((chunk_size, 1))
            samples_np = samples.detach().cpu().numpy().astype(np.float32)
            chunks.append(samples_np)

    return np.concatenate(chunks, axis=0)


def raw_dataset_path(
    data_dir: Path,
    split: str,
    seed: int,
    n_samples: int,
    nu: float,
) -> Path:
    """Return the path for a raw .npz dataset."""
    nu_tag = format_nu_for_filename(nu)
    return data_dir / f"student_t1d_nu{nu_tag}_{split}_seed{seed}_N{n_samples}.npz"


def normalized_dataset_path(
    out_dir: Path,
    train_seed: int,
    test_seed: int,
    n_train: int,
    n_test: int,
    nu: float,
) -> Path:
    """Return the path for a normalized .pt dataset."""
    nu_tag = format_nu_for_filename(nu)
    return out_dir / (
        f"student_t1d_nu{nu_tag}_norm_trainseed{train_seed}_testseed{test_seed}"
        f"_Ntrain{n_train}_Ntest{n_test}.pt"
    )


def save_raw_dataset(
    path: Path,
    samples: np.ndarray,
    nu: float,
    seed: int,
    force: bool,
) -> None:
    """Save raw samples and metadata as a compressed .npz file."""
    if path.exists() and not force:
        print(f"Skip existing raw dataset: {path}")
        return

    np.savez_compressed(
        path,
        x=samples.astype(np.float32),
        nu=np.float32(nu),
        mu=np.float32(0.0),
        sigma=np.float32(1.0),
        seed=np.int64(seed),
    )
    print(f"Saved raw dataset: {path}")


def generate_raw_datasets(
    data_dir: Path,
    nu: float,
    n_train: int,
    n_test: int,
    chunk_size: int,
    train_seeds: Iterable[int],
    test_seed: int,
    device: torch.device,
    force: bool,
) -> None:
    """Generate and save raw train/test datasets."""
    test_path = raw_dataset_path(data_dir, "test", test_seed, n_test, nu)
    if test_path.exists() and not force:
        print(f"Skip existing test dataset: {test_path}")
    else:
        test_raw = sample_student_t_1d_chunked(
            n_samples=n_test,
            chunk_size=chunk_size,
            nu=nu,
            device=device,
            seed=test_seed,
        )
        save_raw_dataset(test_path, test_raw, nu=nu, seed=test_seed, force=force)

    for train_seed in train_seeds:
        train_path = raw_dataset_path(data_dir, "train", train_seed, n_train, nu)
        if train_path.exists() and not force:
            print(f"Skip existing train dataset: {train_path}")
            continue

        train_raw = sample_student_t_1d_chunked(
            n_samples=n_train,
            chunk_size=chunk_size,
            nu=nu,
            device=device,
            seed=train_seed,
        )
        save_raw_dataset(train_path, train_raw, nu=nu, seed=train_seed, force=force)


def normalize_and_save_datasets(
    data_dir: Path,
    nu: float,
    train_seeds: Iterable[int],
    test_seed: int,
    n_train: int,
    n_test: int,
    force: bool,
    eps: float = 1e-8,
) -> None:
    """Normalize datasets using training statistics and save .pt files.

    For each training seed, the mean and standard deviation are computed from
    the corresponding raw training set only. The same statistics are then
    applied to both the training set and the fixed test set. This avoids using
    test-set statistics during preprocessing.
    """
    out_dir = data_dir / "normalized_pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_npz = raw_dataset_path(data_dir, "test", test_seed, n_test, nu)
    if not test_npz.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_npz}")

    test_pack = np.load(test_npz)
    test_raw = test_pack["x"].astype(np.float32)

    nu_loaded = float(test_pack["nu"])
    mu = float(test_pack["mu"]) if "mu" in test_pack else 0.0
    sigma = float(test_pack["sigma"]) if "sigma" in test_pack else 1.0

    print(f"Loaded test dataset: {test_npz}")
    print(f"test_raw: shape={test_raw.shape}, dtype={test_raw.dtype}")
    print(f"distribution parameters: nu={nu_loaded}, mu={mu}, sigma={sigma}")

    for train_seed in train_seeds:
        train_npz = raw_dataset_path(data_dir, "train", train_seed, n_train, nu)
        if not train_npz.exists():
            raise FileNotFoundError(f"Train dataset not found: {train_npz}")

        out_pt = normalized_dataset_path(
            out_dir=out_dir,
            train_seed=train_seed,
            test_seed=test_seed,
            n_train=n_train,
            n_test=n_test,
            nu=nu_loaded,
        )
        if out_pt.exists() and not force:
            print(f"Skip existing normalized dataset: {out_pt}")
            continue

        train_pack = np.load(train_npz)
        train_raw = train_pack["x"].astype(np.float32)

        # Fit normalization statistics on the training data only.
        mean = train_raw.mean(axis=0, keepdims=True)
        std = train_raw.std(axis=0, keepdims=True)
        std = np.maximum(std, eps)

        train = ((train_raw - mean) / std).astype(np.float32)
        test = ((test_raw - mean) / std).astype(np.float32)

        torch.save(
            {
                "train": torch.from_numpy(train),
                "test": torch.from_numpy(test),
                "train_raw_mean": torch.from_numpy(mean.astype(np.float32)),
                "train_raw_std": torch.from_numpy(std.astype(np.float32)),
                "nu": float(nu_loaded),
                "mu": float(mu),
                "sigma": float(sigma),
                "train_seed": int(train_seed),
                "test_seed": int(test_seed),
                "paths": {
                    "train_npz": str(train_npz),
                    "test_npz": str(test_npz),
                },
            },
            out_pt,
        )

        print(
            f"Saved normalized dataset: {out_pt} "
            f"mean={mean.flatten()[0]:.8f}, std={std.flatten()[0]:.8f}"
        )


def main() -> None:
    """Run data generation and normalization."""
    args = parse_args()

    project_root = get_project_root()
    data_dir = args.data_dir if args.data_dir is not None else project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print(f"PROJECT_ROOT: {project_root.resolve()}")
    print(f"DATA_DIR    : {data_dir.resolve()}")
    print(f"device      : {device}")

    generate_raw_datasets(
        data_dir=data_dir,
        nu=args.nu,
        n_train=args.n_train,
        n_test=args.n_test,
        chunk_size=args.chunk_size,
        train_seeds=args.train_seeds,
        test_seed=args.test_seed,
        device=device,
        force=args.force,
    )

    normalize_and_save_datasets(
        data_dir=data_dir,
        nu=args.nu,
        train_seeds=args.train_seeds,
        test_seed=args.test_seed,
        n_train=args.n_train,
        n_test=args.n_test,
        force=args.force,
    )


if __name__ == "__main__":
    main()
