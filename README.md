# ECG Denoising + Compression with WWPRD-based Optimization

This repository contains a deep autoencoder baseline for simultaneous ECG denoising and compression, trained directly with a differentiable waveform-weighted PRD (WWPRD) objective.

## Novelty (for TDK)
- We train with **differentiable WWPRD** aligned to evaluation metrics, rather than MSE.
- Single model performs denoising and compression; optional VP projection layer ablation.

## Quick start
1. Create environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Smoke test the differentiable WWPRD optimization on synthetic data:
   ```bash
   python -m ecgdae.losses
   python scripts/train_synthetic.py
   ```

## Repo structure
- `ecgdae/losses.py`: PRD, WWPRD (time-domain) and STFT-weighted WWPRD losses with a unit test.
- `ecgdae/data.py`: Simple dataset, windowing, and Gaussian SNR noise mixer.
- `scripts/train_synthetic.py`: Tiny CAE trained with WWPRD on synthetic ECG-like data to demonstrate optimization.
- `slides.md`: Meeting deck outline aligned with evaluation criteria.

## Two-week plan (high level)
- Week 1: baseline model + WWPRD loss + training/eval scaffolding on MIT-BIH.
- Week 2: ablations (loss, noise, CR), VP-layer prototype, preliminary report.
