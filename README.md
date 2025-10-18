# ECG Denoising and Compression with WWPRD-based Optimization

A deep learning framework for ECG signal denoising and compression using convolutional autoencoders, optimized with differentiable **Waveform-Weighted PRD (WWPRD)** loss function aligned with clinical diagnostic quality metrics.

## 🎯 Project Objectives

This project develops an ECG signal compression algorithm that:

- **Preserves Diagnostic Quality**: Maintains PRD < 4.33% and WWPRD < 7.4% for "Excellent" clinical usability
- **Adaptive Compression**: Uses Variable Projection (VP) methods for flexible compression ratios
- **Clinical Alignment**: Optimizes directly with WWPRD loss, emphasizing diagnostically-critical features (QRS complexes)
- **High Efficiency**: Achieves significant compression without losing critical diagnostic information

## ✨ Key Innovations

1. **Direct WWPRD Optimization**: Train with differentiable WWPRD loss instead of MSE, aligning training objective with clinical evaluation metrics
2. **Unified Denoising + Compression**: Single end-to-end model performs both tasks simultaneously
3. **Real Noise Training**: Integrates NSTDB (MIT-BIH Noise Stress Test Database) for realistic ECG noise patterns
4. **Automatic QRS Weighting**: Derivative-based weights automatically emphasize QRS complexes without manual annotation
5. **Planned VP Layer Integration**: Variable Projection layer for adaptive signal representation (Week 3)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Main Dependencies:**
- PyTorch >= 2.3.0 (deep learning framework)
- wfdb >= 4.1.0 (MIT-BIH data access)
- numpy, scipy (numerical computing)
- matplotlib, seaborn (visualization)

### 2. Verify Setup
```bash
python scripts/test_setup.py
```
This script tests all dependencies and downloads a sample MIT-BIH record.

### 3. Train Model (Quick Test - 5 minutes)
```bash
python scripts/train_mitbih.py \
    --num_records 3 \
    --epochs 10 \
    --batch_size 32 \
    --loss_type wwprd \
    --output_dir ./outputs/quick_test
```

### 4. Train Model (Standard - 30-60 minutes)
```bash
python scripts/train_mitbih.py \
    --num_records 10 \
    --epochs 50 \
    --batch_size 32 \
    --loss_type wwprd \
    --latent_dim 32 \
    --save_model \
    --output_dir ./outputs/week1
```

### 5. Evaluate Trained Model
```bash
python scripts/evaluate_mitbih.py \
    --model_path ./outputs/week1/best_model.pth \
    --config_path ./outputs/week1/config.json \
    --output_dir ./outputs/week1/evaluation \
    --num_samples 500
```

## 📁 Repository Structure

```
ecg-vp-denoising/
├── ecgdae/                              # Core Python package
│   ├── __init__.py                      # Package initialization
│   ├── data.py                          # Data loading and preprocessing
│   │   ├── MITBIHLoader                 # Automatic MIT-BIH downloader
│   │   ├── NSTDBNoiseMixer              # Real ECG noise injection
│   │   ├── WindowingConfig              # Signal windowing
│   │   └── MITBIHDataset                # PyTorch dataset
│   ├── losses.py                        # Differentiable loss functions
│   │   ├── PRDLoss                      # Standard PRD loss
│   │   ├── WWPRDLoss                    # Waveform-weighted PRD
│   │   └── STFTWeightedWWPRDLoss        # Frequency-domain WWPRD
│   ├── models.py                        # Neural network architectures
│   │   ├── ConvAutoEncoder              # Standard convolutional AE
│   │   └── ResidualAutoEncoder          # Residual connections AE
│   └── metrics.py                       # Evaluation metrics
│       ├── compute_prd, compute_wwprd   # Quality metrics
│       ├── compute_snr                  # Signal-to-noise ratio
│       ├── compute_compression_ratio    # Compression efficiency
│       └── evaluate_reconstruction      # Comprehensive evaluation
├── scripts/                             # Training and evaluation scripts
│   ├── train_mitbih.py                  # Main training pipeline
│   ├── evaluate_mitbih.py               # Model evaluation with visualizations
│   ├── train_synthetic.py               # Synthetic data experiments (optional)
│   └── test_setup.py                    # Environment verification
├── outputs/                             # Training outputs (generated)
│   └── test/                            # Example test run outputs
├── requirements.txt                     # Python package dependencies
├── README.md                            # This file (project overview)
├── WEEK1_GUIDE.md                       # Detailed Week 1 implementation guide
├── slides.md                            # Presentation slides (for TDK report)
└── Generalized_Rational_Variable_Projection_With_Application_in_ECG_Compression(1).pdf
```

## 📊 Week 1 Implementation (✅ COMPLETED)

### ✅ Implemented Components

**1. Data Pipeline**
- ✅ **MITBIHLoader**: Automatic download from PhysioNet, supports all 48 records
- ✅ **NSTDBNoiseMixer**: Three types of real ECG noise (muscle artifact, baseline wander, electrode motion)
- ✅ **Windowing System**: Configurable window length (default 512 samples = 1.4s at 360Hz)
- ✅ **SNR Control**: Precise signal-to-noise ratio adjustment (default 5-15dB range)
- ✅ **PyTorch Integration**: Efficient batched data loading with automatic normalization

**2. Loss Functions (Fully Differentiable)**
- ✅ **PRDLoss**: Standard percent root-mean-square difference
- ✅ **WWPRDLoss**: Derivative-based waveform weights (emphasizes QRS complexes)
- ✅ **STFTWeightedWWPRDLoss**: Frequency-domain weighting (5-40Hz QRS band)
- ✅ All losses support gradient-based optimization

**3. Model Architectures**
- ✅ **ConvAutoEncoder**: 1D convolutional encoder-decoder with BatchNorm and GELU
- ✅ **ResidualAutoEncoder**: Enhanced version with residual blocks for better gradient flow
- ✅ **Configurable Compression**: Adjustable latent dimension (16-128) controls compression ratio

**4. Training Infrastructure**
- ✅ **Complete Training Loop**: Automatic train/val split, early stopping, model checkpointing
- ✅ **Real-time Monitoring**: PRD, WWPRD, SNR improvement tracked per epoch
- ✅ **Loss Curve Generation**: Automatic visualization of training progress
- ✅ **Configuration Saving**: JSON export of all hyperparameters

**5. Evaluation Suite**
- ✅ **Metric Computation**: PRD, WWPRD, SNR, SNR improvement, compression ratio
- ✅ **Quality Classification**: Automatic categorization (Excellent/Very Good/Good/Not Good)
- ✅ **Comprehensive Visualizations**:
  - Training curves (4 subplots: loss, PRD, WWPRD, SNR)
  - Metric distributions (histograms with quality thresholds)
  - PRD vs WWPRD scatter plots (colored by SNR improvement)
  - Quality classification pie charts
  - Best/worst reconstruction gallery (side-by-side comparisons)

### 📈 Generated Outputs

After training, the following files are automatically generated in `outputs/`:

```
outputs/week1/
├── config.json                    # Training configuration
├── best_model.pth                 # Best model checkpoint (based on val loss)
├── training_history.json          # Epoch-by-epoch metrics
├── final_metrics.json             # Final evaluation results
├── training_curves.png            # 4-panel training visualization
└── reconstruction_examples.png    # Sample input/output comparisons

outputs/week1/evaluation/          # Generated by evaluate_mitbih.py
├── evaluation_metrics.json        # Per-sample metrics
├── metric_distributions.png       # PRD/WWPRD/SNR histograms
├── prd_wwprd_scatter.png         # Correlation analysis
├── quality_classification.png     # Pie charts
└── reconstruction_gallery.png     # Best/worst examples
```

## 📈 Clinical Quality Standards

ECG compression quality is evaluated using standardized metrics aligned with clinical diagnostic requirements.

### PRD (Percent Root-mean-square Difference)

**Formula**: `PRD = 100 × sqrt(Σ(x - x̂)² / Σ(x²))`

| Quality Level | PRD Range | Diagnostic Usability | Clinical Interpretation |
|--------------|-----------|---------------------|------------------------|
| **Excellent** | < 4.33% | Fully diagnostic | Suitable for all clinical applications |
| **Very Good** | 4.33% - 9.00% | Diagnostic | Suitable for most clinical applications |
| **Good** | 9.00% - 15.00% | Limited | May miss subtle abnormalities |
| **Not Good** | ≥ 15.00% | Not recommended | Significant diagnostic information loss |

### WWPRD (Waveform-Weighted PRD)

**Formula**: `WWPRD = 100 × sqrt(Σ(w·(x - x̂)²) / Σ(w·x²))`
**Weights**: `w(t) = 1 + α·|x'(t)|/max(|x'|)` (emphasizes high-derivative regions like QRS)

| Quality Level | WWPRD Range | Diagnostic Usability | Clinical Interpretation |
|--------------|-------------|---------------------|------------------------|
| **Excellent** | < 7.4% | Fully diagnostic | QRS complexes well-preserved |
| **Very Good** | 7.4% - 14.8% | Diagnostic | Minor QRS distortion acceptable |
| **Good** | 14.8% - 24.7% | Limited | QRS morphology partially degraded |
| **Not Good** | ≥ 24.7% | Not recommended | QRS complex significantly distorted |

**Why WWPRD Matters**: Unlike PRD which treats all signal regions equally, WWPRD emphasizes:
- **QRS complexes** (ventricular depolarization) - most critical for arrhythmia diagnosis
- **P waves** (atrial depolarization)
- **T waves** (ventricular repolarization)

This makes WWPRD a better indicator of diagnostic quality preservation.

## 🗓️ Four-Week Development Plan

### Week 1: ✅ Data Pipeline + WWPRD Training (COMPLETED)

**Objectives:**
- ✅ Implement MIT-BIH data loading with automatic PhysioNet download
- ✅ Integrate NSTDB real noise mixing with precise SNR control
- ✅ Develop derivative-based WWPRD weight computation
- ✅ Build complete training pipeline with real-time metrics
- ✅ Create comprehensive evaluation suite with visualizations

**Deliverables:**
- ✅ Training curves showing loss, PRD, WWPRD, SNR improvement
- ✅ Reconstruction examples comparing noisy input → denoised output
- ✅ Metric distribution histograms with quality thresholds
- ✅ Quality classification analysis (percentage in each category)
- ✅ PRD vs WWPRD correlation plots

**Status**: All Week 1 objectives completed successfully ✅

---

### Week 2: 📅 Compression Ratio Analysis (IN PROGRESS)

**Objectives:**
- [ ] Implement latent space quantization (4-bit, 6-bit, 8-bit)
- [ ] Calculate theoretical compression ratio (CR)
- [ ] Measure actual CR with entropy coding
- [ ] Generate PRD-CR curves (quality vs compression trade-off)
- [ ] Generate WWPRD-CR curves
- [ ] Analyze optimal operating points (best quality/compression balance)

**Deliverables:**
- [ ] PRD-CR and WWPRD-CR curves with multiple latent dimensions
- [ ] Quantization bit-depth comparison (4/6/8-bit)
- [ ] Rate-distortion analysis table
- [ ] Optimal configuration recommendations

---

### Week 3: 📅 Loss Function Ablation + VP Layer (PLANNED)

**Objectives:**
- [ ] Comparative study: MSE vs PRD vs WWPRD vs STFT-WWPRD
- [ ] Train identical models with different loss functions
- [ ] Implement Variable Projection (VP) adaptive layer
- [ ] Compare VP layer vs standard convolution
- [ ] Analyze computational complexity trade-offs

**Deliverables:**
- [ ] Loss function comparison plots (PRD, WWPRD, convergence speed)
- [ ] VP layer architecture diagram and implementation
- [ ] VP vs convolution ablation study
- [ ] Computational cost analysis

---

### Week 4: 📅 Final Evaluation + Report (PLANNED)

**Objectives:**
- [ ] Train on complete MIT-BIH dataset (all 48 records)
- [ ] Determine optimal hyperparameter configuration
- [ ] Benchmark against literature baselines
- [ ] Prepare comprehensive TDK report
- [ ] Create presentation slides and figures

**Deliverables:**
- [ ] Final performance benchmarks on full dataset
- [ ] Comparison table with state-of-the-art methods
- [ ] TDK research report
- [ ] Presentation slides with key results
- [ ] Code documentation and usage guide

## 🔧 Technical Components

### 1. Data Loading and Preprocessing (`ecgdae/data.py`)

**MITBIHLoader**
- Automatic download from PhysioNet (https://physionet.org/content/mitdb/)
- Supports all 48 MIT-BIH Arrhythmia Database records
- Two-channel ECG (MLII and V5/V2), defaults to MLII (channel 0)
- Sampling rate: 360 Hz

**NSTDBNoiseMixer**
- Three realistic noise types from MIT-BIH Noise Stress Test Database:
  - `muscle_artifact`: High-frequency noise from muscle contractions
  - `baseline_wander`: Low-frequency drift from respiration
  - `electrode_motion`: Motion artifacts from electrode displacement
- Precise SNR control via power normalization
- Optional synthetic noise fallback if NSTDB unavailable

**WindowingConfig**
- Sliding window approach for handling long ECG records
- Default: 512 samples (1.42 seconds at 360 Hz)
- Configurable overlap (default 50%)
- Automatic zero-mean, unit-variance normalization per window

**MITBIHDataset**
- PyTorch Dataset integration for efficient batching
- On-the-fly noise mixing with randomized SNR
- Memory-efficient window caching

### 2. Differentiable Loss Functions (`ecgdae/losses.py`)

**PRDLoss**
```python
PRD = 100 × sqrt(Σ(x - x̂)² / Σ(x²))
```
- Baseline clinical quality metric
- Fully differentiable for gradient descent

**WWPRDLoss** ⭐ (Primary Innovation)
```python
WWPRD = 100 × sqrt(Σ(w·(x - x̂)²) / Σ(w·x²))
w(t) = 1 + α·|dx/dt|/max(|dx/dt|)
```
- Derivative-based weights automatically emphasize QRS complexes
- Parameter α (default 2.0) controls emphasis strength
- No manual R-peak detection required

**STFTWeightedWWPRDLoss**
- Frequency-domain alternative using Short-Time Fourier Transform
- Emphasizes 5-40 Hz band (main QRS spectral content)
- Gaussian frequency weighting

### 3. Neural Network Architectures (`ecgdae/models.py`)

**ConvAutoEncoder**
```
Input: [B, 1, 512]
├─ Encoder (stride-2 convolutions)
│  ├─ Conv1D(1→32, k=9, s=2) + BatchNorm + GELU  → [B, 32, 256]
│  ├─ Conv1D(32→64, k=9, s=2) + BatchNorm + GELU → [B, 64, 128]
│  ├─ Conv1D(64→128, k=9, s=2) + BatchNorm + GELU → [B, 128, 64]
│  └─ Conv1D(128→32, k=9, s=2) + BatchNorm + GELU → [B, 32, 32]  ← Bottleneck
└─ Decoder (transposed convolutions)
   ├─ ConvT1D(32→128, k=4, s=2) + BatchNorm + GELU → [B, 128, 64]
   ├─ ConvT1D(128→64, k=4, s=2) + BatchNorm + GELU → [B, 64, 128]
   ├─ ConvT1D(64→32, k=4, s=2) + BatchNorm + GELU  → [B, 32, 256]
   └─ ConvT1D(32→1, k=4, s=2)                      → [B, 1, 512]
```
- Compression ratio controlled by latent_dim (default 32)
- Parameters: ~147K (for default config)

**ResidualAutoEncoder**
- Enhanced version with ResidualBlock modules at each scale
- Better gradient flow for deeper networks
- 2 residual blocks per resolution level (default)
- Parameters: ~387K (for default config with 2 res blocks)

### 4. Evaluation Metrics (`ecgdae/metrics.py`)

**Core Metrics**
- `compute_prd()`: Percent root-mean-square difference
- `compute_wwprd()`: Waveform-weighted PRD with derivative weights
- `compute_snr()`: Signal-to-noise ratio in dB
- `compute_compression_ratio()`: Theoretical compression efficiency

**Batch Processing**
- `batch_evaluate()`: Vectorized evaluation for entire validation sets
- `evaluate_reconstruction()`: Comprehensive single-sample analysis
- `format_metrics()`: Pretty-printing for console output

**Automatic Quality Classification**
- Maps PRD/WWPRD values to clinical quality levels
- Generates percentage breakdowns (e.g., "85% Excellent, 15% Very Good")

## 💻 Usage Examples

### Training with Different Configurations

**Quick Test (3 records, 10 epochs, ~5 min)**
```bash
python scripts/train_mitbih.py \
    --num_records 3 \
    --epochs 10 \
    --output_dir ./outputs/quick_test
```

**Standard Configuration (10 records, 50 epochs, ~30-60 min)**
```bash
python scripts/train_mitbih.py \
    --num_records 10 \
    --epochs 50 \
    --loss_type wwprd \
    --weight_alpha 2.0 \
    --latent_dim 32 \
    --hidden_dims 32 64 128 \
    --batch_size 32 \
    --save_model \
    --output_dir ./outputs/week1
```

**Full Dataset (48 records, 100 epochs, ~2-4 hours)**
```bash
python scripts/train_mitbih.py \
    --num_records 48 \
    --epochs 100 \
    --model_type residual \
    --loss_type wwprd \
    --latent_dim 32 \
    --save_model \
    --output_dir ./outputs/full_training
```

### Loss Function Comparison

```bash
# Train with MSE (baseline)
python scripts/train_mitbih.py --loss_type mse --output_dir ./outputs/mse

# Train with PRD
python scripts/train_mitbih.py --loss_type prd --output_dir ./outputs/prd

# Train with WWPRD (recommended)
python scripts/train_mitbih.py --loss_type wwprd --output_dir ./outputs/wwprd

# Train with frequency-weighted WWPRD
python scripts/train_mitbih.py --loss_type stft_wwprd --output_dir ./outputs/stft_wwprd
```

### Compression Ratio Exploration

```bash
# High compression (latent_dim=16)
python scripts/train_mitbih.py --latent_dim 16 --output_dir ./outputs/cr_high

# Medium compression (latent_dim=32, default)
python scripts/train_mitbih.py --latent_dim 32 --output_dir ./outputs/cr_medium

# Low compression (latent_dim=64)
python scripts/train_mitbih.py --latent_dim 64 --output_dir ./outputs/cr_low
```

### Evaluation

```bash
# Evaluate trained model on 500 samples
python scripts/evaluate_mitbih.py \
    --model_path ./outputs/week1/best_model.pth \
    --config_path ./outputs/week1/config.json \
    --output_dir ./outputs/week1/evaluation \
    --num_samples 500
```

## 📖 Documentation

- **[WEEK1_GUIDE.md](WEEK1_GUIDE.md)**: Comprehensive Week 1 implementation guide with detailed explanations
- **[slides.md](slides.md)**: Presentation slides for TDK report

## 📄 References

**Primary Theoretical Foundation:**
- **"Generalized Rational Variable Projection with Application in ECG Compression"**
  *(Included in repository as PDF)*

**Datasets:**
- **MIT-BIH Arrhythmia Database**
  Moody GB, Mark RG. *PhysioNet*. 2001.
  https://physionet.org/content/mitdb/

- **MIT-BIH Noise Stress Test Database (NSTDB)**
  Moody GB, Muldrow WE, Mark RG. *PhysioNet*. 1984.
  https://physionet.org/content/nstdb/

**Quality Metrics:**
- PRD and WWPRD standards from ECG compression literature
- Clinical quality thresholds based on diagnostic usability studies

## 🤝 Contributing

This is an active TDK research project. For questions or collaboration:
- Open an issue for bug reports or feature requests
- Check [WEEK1_GUIDE.md](WEEK1_GUIDE.md) for implementation details

## 📝 License

This project is developed for academic and research purposes.

## 📊 Current Status

- ✅ **Week 1**: Data pipeline, WWPRD training, evaluation suite - **COMPLETED**
- 🔄 **Week 2**: Compression ratio analysis - **IN PROGRESS**
- 📅 **Week 3**: Loss ablation + VP layer - **PLANNED**
- 📅 **Week 4**: Final report - **PLANNED**

---

**Last Updated**: October 2025
**Status**: Week 1 Complete ✅ | Ready for Week 2 🚀
**Next Milestone**: Quantization and compression ratio analysis
