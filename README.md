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
