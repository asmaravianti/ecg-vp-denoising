### 1. Motivation
- ECG compression is essential for storage/telemetry, but signals are noisy.
- Distortion is not uniform: QRS/clinical bands matter more.

### 2. Prior Work (very short)
- Classical: DWT/DCT, SPIHT, dictionary learning; metrics: PRD, WWPRD.
- Deep: 1D CAE for compression or denoising separately; trained with MSE.

### 3. Gap
- Evaluation uses PRD/WWPRD, but training uses MSE → objective mismatch.

### 4. Proposal (novelty)
- Train end-to-end with differentiable WWPRD as the main loss.
- Single model jointly denoises and compresses via a controllable bottleneck.
- Optional: replace first conv with VP projection layer; ablation for novelty.

### 5. WWPRD loss design
- Time-weighted WWPRD and frequency-weighted WWPRD (STFT-based), both differentiable.
- Soft weights (no hard masks); stable normalization.

### 6. Model
- 1D CAE baseline; bottleneck width controls CR; quantization-aware noise + L1.
- VP-layer variant as ablation.

### 7. Data & Augmentation
- MIT-BIH Arrhythmia; windowing at 2–5 s; noisy inputs (Gaussian + NSTDB).
- Targets are clean segments.

### 8. Experiments (2-week scope)
- Loss ablation: L2 vs PRD vs WWPRD at CR≈8,16.
- Noise vs no-noise training.
- VP-layer vs conv for WWPRD at fixed CR.

### 9. Metrics & Verification
- CR, PRD, WWPRD, SNR; plots and PRD–CR curves; visual overlays.
- Provide code to compute theoretical and practical CR.

### 10. Timeline & Risks
- Week 1: baseline, WWPRD loss, training loop, metrics.
- Week 2: ablations, VP-layer proto, preliminary report.
- Risks: weight design, CR accounting; mitigations included.

### Appendix A — What we did this week (show quickly)
- Implemented differentiable PRD and WWPRD (time- and STFT-frequency-weighted).
- Verified optimization: losses decrease under gradient descent.
- Built tiny 1D CAE and synthetic ECG+noise pipeline; WWPRD trends down during training.
- Repo scaffold + README; meeting deck aligned to evaluation criteria.
- Demo commands:
  - `py -m ecgdae.losses` → prints PRD/WWPRD decreases.
  - `py -m scripts.train_synthetic` → prints training WWPRD and before/after.

### Appendix B — Literature snapshot (preliminary)
- Classical compression: DWT/DCT, SPIHT/SET, wavelet packet codecs; metrics: PRD, WWPRD.
- Deep learning: 1D CAEs for compression or denoising separately; training uses MSE; evaluation still PRD/WWPRD.
- Gap we target: objective mismatch (train with MSE but evaluate with PRD/WWPRD). We propose differentiable WWPRD as the primary loss.
- Next action: structured review table (methods, datasets, metrics, CR/PRD) by next meeting.

### Appendix C — Next steps (2 weeks)
- MIT-BIH + NSTDB pipeline; first PRD/WWPRD–CR curves.
- Ablations: MSE vs PRD vs WWPRD; noise vs no-noise; bottleneck sizes.
- VP-layer variant at fixed CR; compare rate–distortion–denoising.



