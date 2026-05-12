# Self-Regulating Annealing in Heavy-Tailed Diffusion Models

Official implementation of the paper:

**Self-Regulating Annealing in Heavy-Tailed Diffusion Models**  
Keito Wakatsuki and Hideaki Shimazaki

This repository contains code for reproducing the one-dimensional synthetic experiments in the paper. The experiments compare Gaussian VE-SDE sampling, t-ODE sampling, the proposed t-SDE sampler, and an ablated t-SDE sampler without the state-dependent diffusion coefficient.

## Overview

Diffusion models are commonly formulated with Gaussian transitions, but this can be insufficient for heavy-tailed data. Heavy-Tailed Diffusion Models (HTDMs) replace the Gaussian formulation with a Student's t-distribution. In this work, we study the SDE-based sampler for HTDMs and propose a sampler with a state-dependent diffusion coefficient.

The state-dependent coefficient induces a **self-regulating annealing** mechanism: the effective noise scale increases when the current state is far from the denoiser's target estimate and decreases when the state approaches it.

The repository reproduces the synthetic Student's t experiment used to evaluate tail fidelity.

## Repository structure

```text
.
├── data.py                      # Generate and normalize 1D Student-t datasets
├── run.py                       # Train EDM / t-EDM denoisers
├── sample.py                    # Generate samples from trained checkpoints
├── qq_plot.py                   # Create Q-Q plots
├── quantitative_evaluation.py   # Compute W1 and tail-probability metrics
├── requirements.txt             # Python dependencies
├── LICENSE
└── README.md
```

The scripts are organized according to the experimental workflow: data generation, training, sampling, plotting, and quantitative evaluation.

## Usage

The basic workflow consists of four steps:

1. Generate the dataset.
2. Train the denoisers.
3. Generate samples.
4. Create Q-Q plots.
5. Compute quantitative metrics.

### 1. Generate data

```bash
python data.py
```

This creates one-dimensional Student's t datasets and saves the normalized data under `data/normalized_pt/`.

### 2. Train models

```bash
python run.py
```

Training outputs are saved under `runs/`.

### 3. Generate samples

```bash
python sample.py --train-seed 4
```

Generated samples are saved under `samples/`.

### 4. Create Q-Q plots

```bash
python qq_plot.py --sample-dir samples/<sample-directory>
Replace `<sample-directory>` with the directory created by `sample.py`.
```

### 5. Compute quantitative metrics

```bash
python quantitative_evaluation.py --sample-dir samples/<sample-directory>
```

## Samplers

The sampling script generates samples from four samplers:

| File | Description |
|---|---|
| `g_sde.npy` | Gaussian VE-SDE baseline |
| `t_ode.npy` | ODE-based sampler for t-EDM |
| `t_sde.npy` | Proposed t-SDE sampler with the state-dependent coefficient |
| `t_sde_coeff1.npy` | Ablated t-SDE sampler with the coefficient fixed to 1 |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{wakatsuki2026selfregulating,
  title     = {Self-Regulating Annealing in Heavy-Tailed Diffusion Models},
  author    = {Wakatsuki, Keito and Shimazaki, Hideaki},
  booktitle = {Proceedings of the International Joint Conference on Neural Networks},
  year      = {2026}
}
```

## License

This repository is released under the MIT License.
