# ECG Denoising and Compression with WWPRD-based Optimization
## Progress Report: Week 1 - Week 2

**Project Team:** [Your Names]  
**Date:** progress_report_week1_week2  
**Status:** Week 1 ✅ Complete | Week 2 🔄 In Progress

---

## Executive Summary

This report summarizes our progress on developing a deep learning framework for ECG signal denoising and compression using Waveform-Weighted PRD (WWPRD) loss optimization. 

**Key Achievements:**
- ✅ **Week 1**: Successfully implemented complete data pipeline, trained baseline models with WWPRD loss, achieved significant denoising improvements (SNR improvement: ~6 dB)
- 🔄 **Week 2**: Implemented compression ratio analysis framework, created comprehensive visualization suite for rate-distortion analysis

**Current Status:** 
- Model trained on 10 MIT-BIH records (50 epochs)
- Baseline autoencoder achieving PRD ~28%, WWPRD ~25% on validation set
- Compression framework ready; awaiting final CR sweep results for Week 2 deliverables

---

## 1. Project Overview

### 1.1 Objectives

Our project aims to develop an ECG signal compression algorithm that:
1. **Preserves Diagnostic Quality**: Maintains PRD < 4.33% and WWPRD < 7.4% for "Excellent" clinical usability
2. **Unified Approach**: Single end-to-end model performs both denoising and compression simultaneously
3. **Clinical Alignment**: Optimizes directly with WWPRD loss, emphasizing diagnostically-critical features (QRS complexes)
4. **High Efficiency**: Achieves significant compression without losing critical diagnostic information

### 1.2 Key Innovations

1. **Direct WWPRD Optimization**: Train with differentiable WWPRD loss instead of MSE, aligning training objective with clinical evaluation metrics
2. **Automatic QRS Weighting**: Derivative-based weights automatically emphasize QRS complexes without manual annotation
3. **Real Noise Training**: Integrates NSTDB (MIT-BIH Noise Stress Test Database) for realistic ECG noise patterns
4. **Variable Projection (VP) Layer** (Planned Week 3): Adaptive signal representation for improved rate-distortion trade-offs

### 1.3 Clinical Quality Standards

ECG compression quality is evaluated using standardized metrics:

| Quality Level | PRD Range | WWPRD Range | Diagnostic Usability |
|--------------|-----------|-------------|---------------------|
| **Excellent** | < 4.33% | < 7.4% | Fully diagnostic |
| **Very Good** | 4.33% - 9.00% | 7.4% - 14.8% | Diagnostic |
| **Good** | 9.00% - 15.00% | 14.8% - 24.7% | Limited |
| **Not Good** | ≥ 15.00% | ≥ 24.7% | Not recommended |

---

## 2. Week 1 Progress (✅ COMPLETED)

### 2.1 Data Pipeline Implementation

**✅ MIT-BIH Arrhythmia Database Loader**
- Automatic download from PhysioNet (48 records available)
- Two-channel ECG (MLII and V5/V2), defaulting to MLII channel
- Sampling rate: 360 Hz
- Automatic windowing: 512 samples (~1.4 seconds) with configurable overlap

**✅ NSTDB Noise Integration**
- Three realistic noise types:
  - `muscle_artifact`: High-frequency noise from muscle contractions
  - `baseline_wander`: Low-frequency drift from respiration  
  - `electrode_motion`: Motion artifacts from electrode displacement
- Precise SNR control: Random SNR between 5-15 dB per window
- Power normalization for consistent noise levels

**✅ PyTorch Integration**
- Efficient batched data loading with automatic normalization
- Zero-mean, unit-variance normalization per window
- Train/validation split (15% validation)

### 2.2 Loss Functions

**✅ Differentiable PRD Loss**
```
PRD = 100 × sqrt(Σ(x - x̂)² / Σ(x²))
```

**✅ WWPRD Loss** ⭐ *Primary Innovation*
```
WWPRD = 100 × sqrt(Σ(w·(x - x̂)²) / Σ(w·x²))
w(t) = 1 + α·|dx/dt|/max(|dx/dt|)
```
- Derivative-based weights automatically emphasize QRS complexes
- Parameter α (default 2.0) controls emphasis strength
- No manual R-peak detection required
- Fully differentiable for gradient descent

### 2.3 Model Architecture

**Convolutional Autoencoder (CAE)**
- Input: [Batch, 1, 512] ECG windows
- Encoder: 4 strided convolutions (512 → 256 → 128 → 64 → 32 latent)
- Decoder: 4 transposed convolutions (32 → 64 → 128 → 256 → 512)
- Activation: GELU with BatchNorm
- Parameters: ~147K for default config (latent_dim=32)

### 2.4 Training Configuration

**Training Setup:**
- **Dataset**: 10 MIT-BIH records (split into windows)
- **Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Loss Function**: WWPRD (weight_alpha=2.0)
- **Latent Dimension**: 32 (controls compression ratio)
- **Validation Split**: 15%
- **Early Stopping**: Based on validation loss

### 2.5 Week 1 Results

**Final Metrics (Validation Set):**
- **PRD**: 28.07% (std: 19.89%)
- **WWPRD**: 25.07% (std: 19.84%)
- **SNR Input**: 6.29 dB
- **SNR Output**: 12.26 dB
- **SNR Improvement**: 5.97 dB

**Training Progress:**
- Training loss decreased from ~34.6 to ~23.0 (33% reduction)
- Validation loss decreased from ~28.2 to ~22.6 (20% reduction)
- Model shows consistent improvement over 50 epochs
- No overfitting observed (train/val loss tracking closely)

**Key Findings:**
1. ✅ **Denoising Effective**: SNR improvement of ~6 dB demonstrates successful noise removal
2. ✅ **WWPRD Loss Working**: Model converges successfully with derivative-based weighting
3. ⚠️ **Quality Improvement Needed**: Current PRD (~28%) and WWPRD (~25%) are above "Good" threshold (15%)
4. 📊 **Next Steps**: Need compression ratio analysis to understand rate-distortion trade-offs

**Generated Files:**
- `outputs/week1_presentation/training_curves.png` - Training progress visualization
- `outputs/week1_presentation/reconstruction_examples.png` - Sample denoised outputs
- `outputs/week1_presentation/best_model.pth` - Trained model checkpoint
- `outputs/week1_presentation/config.json` - Training configuration
- `outputs/week1_presentation/training_history.json` - Epoch-by-epoch metrics

---

## 3. Week 2 Progress (🔄 IN PROGRESS)

### 3.1 Compression Framework Implementation

**✅ Quantization Module** (`ecgdae/quantization.py`)
- Uniform quantization for latent codes (4-bit, 6-bit, 8-bit support)
- Dequantization for reconstruction
- Compression ratio calculation with side information accounting
- Function: `compute_compression_ratio(latent_dim, quantization_bits, window_size)`

**✅ Evaluation Script** (`scripts/evaluate_compression.py`)
- CR sweep across multiple compression ratios (4, 8, 16, 32)
- Per-CR metric computation (PRD, WWPRD, SNR)
- JSON export for visualization pipeline
- Configurable quantization bits and test sample count

### 3.2 Visualization Suite

**✅ Visualization Module** (`ecgdae/visualization.py`)
Five publication-quality plotting functions:

1. **`plot_rate_distortion_curves()`** - PRD-CR and WWPRD-CR side-by-side
2. **`plot_snr_bar_chart()`** - SNR improvement bars across CRs
3. **`plot_reconstruction_overlay()`** - Detailed ECG overlay with QRS zoom
4. **`plot_multiple_cr_comparison()`** - Multi-CR stacked comparison
5. **`create_week2_summary_figure()`** - Comprehensive 4-panel summary

**✅ Plotting Script** (`scripts/plot_rate_distortion.py`)
- Main script to generate all Week 2 plots
- Supports both real data and mock data for testing
- Automatic figure generation (300 DPI, publication-ready)
- Rich console output with progress tracking

### 3.3 Week 2 Deliverables (Framework Ready)

**Expected Outputs:**
- `rate_distortion_curves.png` - PRD/WWPRD vs CR curves with clinical thresholds
- `snr_bar_chart.png` - SNR improvement across compression ratios
- `reconstruction_overlay_cr8.png` - Sample reconstruction at CR=8
- `reconstruction_overlay_cr16.png` - Sample reconstruction at CR=16
- `multi_cr_comparison.png` - Stacked comparison across CRs
- `week2_summary.png` - Comprehensive 4-panel summary figure

**Status:**
- ✅ Visualization framework complete
- ✅ Mock data generation for independent testing
- ⏳ Awaiting final CR sweep results from `evaluate_compression.py`

**Demo Visualizations Generated:**
The visualization framework has been tested with mock data, demonstrating:
- Rate-distortion trade-off curves
- SNR improvement analysis
- Reconstruction quality comparison
- Multi-CR visualizations

*Note: Final Week 2 results will be updated once CR sweep completes.*

---

## 4. Technical Methodology

### 4.1 Architecture Overview

**Encoder-Decoder Structure:**
```
Input ECG Window (512 samples, 1.4s @ 360Hz)
  ↓ [Encoder: Strided Convolutions]
Latent Code (32 dimensions) ← Quantization (8-bit)
  ↓ [Decoder: Transposed Convolutions]
Reconstructed ECG (512 samples)
```

**Compression Ratio Calculation:**
```
CR = (Original Bits) / (Latent Bits + Side Info Bits)
   = (512 × 16-bit) / (32 × quantization_bits + metadata)
```

### 4.2 WWPRD Weight Computation

**Derivative-Based Weighting:**
1. Compute signal derivative: `x'(t) = x[t] - x[t-1]`
2. Normalize by maximum: `w_norm(t) = |x'(t)| / max(|x'|)`
3. Apply alpha scaling: `w(t) = 1 + α × w_norm(t)`
4. Weight applied in PRD calculation to emphasize high-derivative regions (QRS)

**Why It Works:**
- QRS complexes have high derivatives (rapid voltage changes)
- P-waves and T-waves also emphasized (moderate derivatives)
- Baseline segments get lower weight (low derivatives)
- Automatic, no manual annotation needed

### 4.3 Training Strategy

**Loss Function:**
- Primary: WWPRD loss (derivative-weighted)
- Weight alpha: 2.0 (balances baseline vs QRS emphasis)
- Gradient-based optimization (Adam, lr=0.001)

**Data Augmentation:**
- Random SNR per window (5-15 dB)
- Random noise type selection (muscle/baseline/electrode)
- Window normalization (zero-mean, unit-variance)

---

## 5. Key Findings and Insights

### 5.1 Denoising Performance

✅ **Significant SNR Improvement**
- Input SNR: ~6.3 dB (moderately noisy)
- Output SNR: ~12.3 dB (improved by ~6 dB)
- **Interpretation**: Model successfully removes noise while preserving signal structure

### 5.2 Reconstruction Quality

⚠️ **Quality Above Target**
- Current PRD: ~28% (target: <15% for "Good", <4.33% for "Excellent")
- Current WWPRD: ~25% (target: <24.7% for "Good", <7.4% for "Excellent")
- **Interpretation**: Model needs further optimization or architecture improvements

**Possible Reasons:**
1. Limited training data (10 records vs full 48)
2. Bottleneck too small (latent_dim=32 may be limiting)
3. Loss function may need tuning (alpha parameter)
4. Training epochs may need extension (currently 50)

### 5.3 Training Stability

✅ **Convergence Observed**
- Smooth loss decrease over 50 epochs
- No overfitting (train/val tracking closely)
- Consistent metric improvements
- Model checkpointing working correctly

---

## 6. Next Steps (Week 3-4 Plan)

### 6.1 Immediate Actions (Week 3)

1. **Complete Week 2 Deliverables**
   - Run full CR sweep (4, 8, 16, 32)
   - Generate final rate-distortion curves
   - Analyze optimal operating points

2. **Loss Function Ablation Study**
   - Compare: MSE vs PRD vs WWPRD vs STFT-WWPRD
   - Train identical models with different losses
   - Measure convergence speed and final quality

3. **Architecture Improvements**
   - Experiment with larger latent dimensions
   - Try ResidualAutoEncoder for better gradient flow
   - Tune hyperparameters (alpha, learning rate)

4. **VP Layer Prototype** (Novel Contribution)
   - Implement Variable Projection layer
   - Replace first encoder block
   - Compare VP vs standard convolution

### 6.2 Future Work (Week 4)

1. **Full Dataset Training**
   - Expand to all 48 MIT-BIH records
   - Extended training (100+ epochs)
   - Final hyperparameter optimization

2. **Literature Comparison**
   - Benchmark against published methods
   - Compare CR vs quality trade-offs
   - Document computational complexity

3. **Report and Presentation**
   - 4-6 page TDK report
   - Comprehensive slides with results
   - Code documentation

---

## 7. Project Timeline

| Week | Objectives | Status |
|------|------------|--------|
| **Week 1** | Data pipeline, WWPRD training, baseline results | ✅ Complete |
| **Week 2** | Compression framework, rate-distortion analysis | 🔄 In Progress |
| **Week 3** | Loss ablation, VP layer, architecture improvements | 📅 Planned |
| **Week 4** | Full evaluation, report, presentation | 📅 Planned |

---

## 8. Challenges and Solutions

### Challenge 1: Data Loading Complexity
**Problem**: MIT-BIH requires PhysioNet credentials and NSTDB integration  
**Solution**: Automated downloader with fallback to synthetic noise

### Challenge 2: WWPRD Differentiability
**Problem**: Ensuring gradient flow through derivative computation  
**Solution**: Careful implementation with numerical stability (eps=1e-8)

### Challenge 3: Training Stability
**Problem**: Initial training showed instability  
**Solution**: Proper normalization, BatchNorm, and learning rate tuning

### Challenge 4: Compression Accounting
**Problem**: Accurate CR calculation with side information  
**Solution**: Comprehensive accounting including latent bits, quantization, metadata

---

## 9. Code Repository Structure

```
ecg-vp-denoising/
├── ecgdae/                          # Core package
│   ├── data.py                      # MIT-BIH loader, NSTDB mixer
│   ├── losses.py                    # PRD, WWPRD losses
│   ├── models.py                    # Autoencoder architectures
│   ├── metrics.py                   # Evaluation metrics
│   ├── quantization.py              # CR calculation
│   └── visualization.py              # Plotting functions
├── scripts/
│   ├── train_mitbih.py              # Main training script
│   ├── evaluate_compression.py     # CR sweep evaluation
│   └── plot_rate_distortion.py     # Week 2 visualizations
├── outputs/
│   ├── week1_presentation/         # Week 1 results
│   └── week2_plots_demo/           # Week 2 demo plots
└── requirements.txt                 # Dependencies
```

---

## 10. Conclusion

We have successfully completed Week 1 objectives and made significant progress on Week 2 deliverables:

**✅ Completed:**
- Complete data pipeline with real MIT-BIH and NSTDB noise
- Trained baseline autoencoder with WWPRD loss
- Achieved ~6 dB SNR improvement on noisy ECG signals
- Implemented comprehensive evaluation and visualization framework

**🔄 In Progress:**
- Compression ratio analysis (framework ready, awaiting final results)
- Rate-distortion curve generation

**📅 Planned:**
- Loss function ablation study
- Variable Projection (VP) layer implementation
- Full dataset training and final evaluation

**Next Meeting:** We plan to present final Week 2 results (CR sweep curves) and discuss VP layer architecture for Week 3.

---

**Report Generated:** progress_report_week1_week2  
**Code Repository:** https://github.com/[your-repo]/ecg-vp-denoising  
**Contact:** [Your Email]
