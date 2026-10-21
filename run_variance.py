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
