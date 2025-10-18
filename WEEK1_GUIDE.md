# Week 1 Implementation Guide

**Complete Tutorial for ECG Denoising with WWPRD-based Optimization**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Detailed Component Explanations](#detailed-component-explanations)
5. [Training Configurations](#training-configurations)
6. [Evaluation and Visualization](#evaluation-and-visualization)
7. [Understanding the Outputs](#understanding-the-outputs)
8. [Troubleshooting](#troubleshooting)
9. [Next Steps](#next-steps)

---

## Overview

### ✅ Week 1 Achievements

Week 1 has been **successfully completed** with all objectives met:

- ✅ **Task 1**: MIT-BIH data loader with automatic PhysioNet download
- ✅ **Task 2**: NSTDB noise mixing with precise SNR control
- ✅ **Task 3**: Derivative-based WWPRD weight computation
- ✅ **Task 4**: Complete training pipeline with real-time metrics
- ✅ **Task 5**: Comprehensive evaluation suite with visualizations

### Project Structure

```
ecg-vp-denoising/
├── ecgdae/                      # Core Python package
│   ├── __init__.py              # Package exports
│   ├── data.py                  # Data loading & preprocessing (338 lines)
│   ├── losses.py                # Differentiable loss functions (163 lines)
│   ├── models.py                # Neural network architectures (345 lines)
│   └── metrics.py               # Evaluation metrics (322 lines)
├── scripts/                     # Executable scripts
│   ├── train_mitbih.py          # Main training script (558 lines)
│   ├── evaluate_mitbih.py       # Evaluation & visualization (496 lines)
│   ├── train_synthetic.py       # Synthetic data demo (optional)
│   └── test_setup.py            # Environment verification
├── outputs/                     # Generated results
│   └── test/                    # Example training outputs
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
├── WEEK1_GUIDE.md              # This file
└── slides.md                    # Presentation slides
```

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/asmaravianti/ecg-vp-denoising.git
cd ecg-vp-denoising
```

### Step 2: Create Python Environment (Recommended)

```bash
# Using conda
conda create -n ecg python=3.10
conda activate ecg

# Or using venv
python -m venv ecg_env
source ecg_env/bin/activate  # On Windows: ecg_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Packages:**
- `torch >= 2.3.0` - Deep learning framework
- `torchaudio >= 2.3.0` - Audio/signal processing utilities
- `wfdb >= 4.1.0` - PhysioNet database access
- `numpy >= 1.24.0` - Numerical computing
- `scipy >= 1.10.0` - Scientific computing
- `matplotlib >= 3.7.0` - Plotting
- `seaborn >= 0.12.0` - Statistical visualization
- `scikit-learn >= 1.3.0` - Machine learning utilities
- `einops >= 0.7.0` - Tensor operations
- `PyYAML >= 6.0.0` - Configuration files
- `rich >= 13.7.0` - Terminal formatting

### Step 4: Verify Installation

```bash
python scripts/test_setup.py
```

**Expected Output:**
```
✓ PyTorch installed: 2.3.0
✓ CUDA available: True (or False for CPU)
✓ wfdb installed: 4.1.0
✓ Testing MIT-BIH download...
✓ Successfully loaded record 100: 650000 samples
✓ Testing NSTDB noise...
✓ Noise added successfully
✓ All tests passed!
```

---

## Quick Start

### Option 1: Quick Test (5 minutes)

For rapid prototyping and code verification:

```bash
python scripts/train_mitbih.py \
    --num_records 3 \
    --epochs 10 \
    --batch_size 32 \
    --loss_type wwprd \
    --output_dir ./outputs/quick_test
```

**What this does:**
- Loads 3 MIT-BIH records (~1,950,000 samples)
- Creates ~3,800 training windows
- Trains for 10 epochs (~5 minutes on GPU)
- Generates training curves and reconstruction examples

### Option 2: Standard Training (30-60 minutes)

For Week 1 deliverables:

```bash
python scripts/train_mitbih.py \
    --num_records 10 \
    --epochs 50 \
    --batch_size 32 \
    --loss_type wwprd \
    --weight_alpha 2.0 \
    --latent_dim 32 \
    --hidden_dims 32 64 128 \
    --lr 1e-3 \
    --save_model \
    --output_dir ./outputs/week1
```

**What this does:**
- Loads 10 MIT-BIH records (~6,500,000 samples)
- Creates ~12,700 training windows
- Trains for 50 epochs (~45 minutes on GPU)
- Saves best model checkpoint
- Generates comprehensive training visualizations

### Option 3: Full Dataset (2-4 hours)

For final evaluation:

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

---

## Detailed Component Explanations

### 1. Data Pipeline (`ecgdae/data.py`)

#### MITBIHLoader

**Purpose**: Automatically downloads and loads MIT-BIH Arrhythmia Database from PhysioNet.

**Key Features:**
```python
loader = MITBIHLoader(data_dir="./data/mitbih")

# Load single record
signal, metadata = loader.load_record('100', channel=0)
# signal: numpy array of ECG samples (650,000 points at 360 Hz)
# metadata: {'fs': 360, 'sig_name': 'MLII', 'units': 'mV', 'record': '100'}

# Load multiple records
records = loader.load_multiple_records(max_records=10)
```

**MIT-BIH Database Details:**
- **48 records** from 47 patients (30 minutes each)
- **360 Hz sampling rate**
- **Two channels**: MLII (modified limb lead II) and V5/V2
- **Annotations**: Beat types, rhythm changes (not used in Week 1)

#### NSTDBNoiseMixer

**Purpose**: Adds realistic ECG noise for robust model training.

**Noise Types:**

1. **Muscle Artifact (`ma`)**:
   - High-frequency noise from muscle contractions
   - Frequency range: 5-50 Hz
   - Most common in real ECG recordings

2. **Baseline Wander (`bw`)**:
   - Low-frequency drift from respiration
   - Frequency range: 0.05-0.5 Hz
   - Affects signal baseline

3. **Electrode Motion (`em`)**:
   - Transient artifacts from electrode displacement
   - Irregular spikes and jumps

**Usage Example:**
```python
noise_mixer = NSTDBNoiseMixer()

# Add noise at 10 dB SNR
noisy_signal = noise_mixer.add_noise(
    clean_signal,
    snr_db=10.0,
    noise_type='muscle_artifact'
)
```

**SNR Calculation:**
```
SNR_dB = 10 × log₁₀(P_signal / P_noise)

where:
P_signal = mean(signal²)
P_noise = mean(noise²)
```

#### WindowingConfig

**Purpose**: Splits long ECG records into fixed-length windows for CNN processing.

**Configuration:**
```python
config = WindowingConfig(
    window_length=512,    # samples (~1.42 seconds at 360 Hz)
    step_size=256,        # 50% overlap
    fs=360                # sampling frequency
)
```

**Why Windowing?**
- CNNs require fixed input sizes
- 512 samples captures 1-2 heartbeats
- Overlap increases training data diversity
- Maintains temporal context

**Window Processing:**
1. Extract sliding windows with overlap
2. Normalize each window: `(x - mean(x)) / std(x)`
3. Shuffle windows for training
4. Create train/val split

#### MITBIHDataset

**Purpose**: PyTorch Dataset wrapper for efficient batching and data loading.

**Features:**
- On-the-fly noise mixing (different noise per epoch)
- Randomized SNR within specified range
- Memory-efficient window caching
- Automatic train/validation splitting

**Example:**
```python
from ecgdae.data import create_dataloaders

train_loader, val_loader = create_dataloaders(
    num_records=10,
    train_split=0.85,
    batch_size=32,
    snr_range=(5.0, 15.0),
    num_workers=4
)

# Iterate over batches
for noisy_batch, clean_batch in train_loader:
    # noisy_batch: [32, 1, 512] - input
    # clean_batch: [32, 1, 512] - target
    ...
```

---

### 2. Loss Functions (`ecgdae/losses.py`)

#### PRDLoss

**Standard Percent Root-mean-square Difference**

**Formula:**
```
PRD = 100 × sqrt(Σ(x - x̂)² / Σ(x²))
```

**Implementation:**
```python
from ecgdae.losses import PRDLoss

criterion = PRDLoss()
loss = criterion(clean_signal, reconstructed_signal)
# Returns scalar PRD value (percentage)
```

**Characteristics:**
- Treats all signal regions equally
- Sensitive to baseline shifts
- Clinical threshold: PRD < 4.33% for "Excellent"

#### WWPRDLoss ⭐

**Waveform-Weighted PRD (Primary Innovation)**

**Formula:**
```
WWPRD = 100 × sqrt(Σ(w·(x - x̂)²) / Σ(w·x²))

where weights: w(t) = 1 + α·|dx/dt| / max(|dx/dt|)
```

**Why WWPRD?**
- **Emphasizes QRS complexes**: High derivative regions (R-peaks) get higher weights
- **Automatic weighting**: No manual R-peak detection required
- **Clinical relevance**: QRS morphology is critical for arrhythmia diagnosis
- **Better metric**: Aligns training objective with diagnostic quality

**Weight Visualization:**

```
ECG Signal:
     ___
    /   \      ← T wave (low derivative, weight ≈ 1.0)
___/     \___
   ↑      ↑
   R      S   ← QRS complex (high derivative, weight ≈ 3.0)
P  Q      T
```

**Implementation:**
```python
from ecgdae.losses import WWPRDLoss

criterion = WWPRDLoss()

# Weights computed automatically from clean signal
loss = criterion(clean_signal, reconstructed_signal)
```

**Alpha Parameter:**
- `α = 1.0`: Mild emphasis (weights: 1.0 - 2.0)
- `α = 2.0`: **Standard** (weights: 1.0 - 3.0) ← Recommended
- `α = 3.0`: Strong emphasis (weights: 1.0 - 4.0)

#### STFTWeightedWWPRDLoss

**Frequency-domain Weighted PRD**

**Approach:**
1. Compute STFT (Short-Time Fourier Transform) of signals
2. Apply frequency weighting (emphasize 5-40 Hz QRS band)
3. Calculate PRD in STFT domain

**Advantages:**
- Frequency-specific emphasis
- Reduces baseline wander sensitivity
- Handles different ECG frequency content

**Usage:**
```python
from ecgdae.losses import STFTWeightedWWPRDLoss

criterion = STFTWeightedWWPRDLoss(
    n_fft=256,           # FFT size
    hop_length=64,       # STFT hop
    win_length=256       # Window size
)

loss = criterion(clean_signal, reconstructed_signal)
```

---

### 3. Model Architectures (`ecgdae/models.py`)

#### ConvAutoEncoder

**1D Convolutional Autoencoder for ECG Compression**

**Architecture:**

```
INPUT: [Batch, 1, 512]

ENCODER:
  Conv1D(1 → 32, k=9, s=2, p=4)  + BatchNorm + GELU  → [B, 32, 256]
  Conv1D(32 → 64, k=9, s=2, p=4) + BatchNorm + GELU  → [B, 64, 128]
  Conv1D(64 → 128, k=9, s=2, p=4) + BatchNorm + GELU → [B, 128, 64]
  Conv1D(128 → 32, k=9, s=2, p=4) + BatchNorm + GELU → [B, 32, 32]
                                                         ↑ BOTTLENECK

DECODER:
  ConvT1D(32 → 128, k=4, s=2, p=1) + BatchNorm + GELU → [B, 128, 64]
  ConvT1D(128 → 64, k=4, s=2, p=1) + BatchNorm + GELU → [B, 64, 128]
  ConvT1D(64 → 32, k=4, s=2, p=1) + BatchNorm + GELU  → [B, 32, 256]
  ConvT1D(32 → 1, k=4, s=2, p=1)                      → [B, 1, 512]

OUTPUT: [Batch, 1, 512]
```

**Key Design Choices:**

1. **Kernel Size 9**: Captures multiple ECG samples for temporal context
2. **Stride 2**: Halves resolution at each layer (progressive compression)
3. **BatchNorm**: Normalizes activations for stable training
4. **GELU**: Smooth activation function (better than ReLU for signals)
5. **Bottleneck**: latent_dim controls compression ratio

**Compression Ratio:**
```
Original: 512 samples × 11 bits/sample = 5,632 bits
Compressed: 32 latent × 32 features × 8 bits = 8,192 bits
Theoretical CR = 5,632 / 8,192 = 0.69:1 (without quantization)

With 8-bit quantization:
Compressed: 32 × 32 × 8 = 8,192 bits
CR = 5,632 / 8,192 = 0.69:1

With 4-bit quantization:
Compressed: 32 × 32 × 4 = 4,096 bits
CR = 5,632 / 4,096 = 1.38:1
```

**Instantiation:**
```python
from ecgdae.models import ConvAutoEncoder

model = ConvAutoEncoder(
    in_channels=1,
    hidden_dims=(32, 64, 128),
    latent_dim=32,
    kernel_size=9,
    activation='gelu'
)

# Count parameters
from ecgdae.models import count_parameters
print(f"Parameters: {count_parameters(model):,}")  # ~147,000
```

#### ResidualAutoEncoder

**Enhanced Architecture with Residual Connections**

**Residual Block:**
```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        residual = x
        out = Conv1D(x)
        out = BatchNorm(out)
        out = GELU(out)
        out = Conv1D(out)
        out = BatchNorm(out)
        out = out + residual  # ← Skip connection
        return GELU(out)
```

**Why Residual Connections?**
- **Better gradient flow**: Gradients bypass deep layers
- **Easier optimization**: Network can learn identity mapping
- **Better reconstruction**: Preserves fine details

**Architecture:**
```
INPUT: [B, 1, 512]

ENCODER:
  Conv1D(1 → 32, s=2) + BN + GELU → [B, 32, 256]
  ResBlock(32) × 2

  Conv1D(32 → 64, s=2) + BN + GELU → [B, 64, 128]
  ResBlock(64) × 2

  Conv1D(64 → 128, s=2) + BN + GELU → [B, 128, 64]
  ResBlock(128) × 2

  Conv1D(128 → 32, s=2) + BN + GELU → [B, 32, 32] ← BOTTLENECK

DECODER: (symmetric structure)

OUTPUT: [B, 1, 512]
```

**Usage:**
```python
from ecgdae.models import ResidualAutoEncoder

model = ResidualAutoEncoder(
    in_channels=1,
    hidden_dims=(32, 64, 128),
    latent_dim=32,
    num_res_blocks=2,  # Blocks per scale
    kernel_size=9
)

print(f"Parameters: {count_parameters(model):,}")  # ~387,000
```

---

### 4. Evaluation Metrics (`ecgdae/metrics.py`)

#### Core Metric Functions

**compute_prd()**
```python
from ecgdae.metrics import compute_prd

prd = compute_prd(clean_signal, reconstructed_signal)
# Returns: PRD value in percentage (e.g., 3.25)
```

**compute_wwprd()**
```python
from ecgdae.metrics import compute_wwprd, compute_derivative_weights

# Automatic weighting
wwprd = compute_wwprd(clean_signal, reconstructed_signal)

# Custom weights
weights = compute_derivative_weights(clean_signal, alpha=2.0)
wwprd = compute_wwprd(clean_signal, reconstructed_signal, weights=weights)
```

**compute_snr()**
```python
from ecgdae.metrics import compute_snr

# Input SNR (noisy vs clean)
snr_in = compute_snr(clean_signal, noisy_signal)

# Output SNR (reconstructed vs clean)
snr_out = compute_snr(clean_signal, reconstructed_signal)

# SNR improvement
snr_improvement = snr_out - snr_in  # Should be positive!
```

#### Comprehensive Evaluation

```python
from ecgdae.metrics import evaluate_reconstruction

metrics = evaluate_reconstruction(
    clean=clean_signal,
    reconstructed=reconstructed_signal,
    noisy=noisy_signal,
    latent_dim=32,
    latent_bits=8
)

print(metrics)
# {
#     'PRD': 3.45,
#     'WWPRD': 6.21,
#     'SNR_in': 10.5,
#     'SNR_out': 25.3,
#     'SNR_improvement': 14.8,
#     'CR': 1.38,
#     'PRD_quality': 'Excellent',
#     'WWPRD_quality': 'Excellent'
# }
```

#### Quality Classification

**Automatic categorization based on clinical standards:**

| Metric | Excellent | Very Good | Good | Not Good |
|--------|-----------|-----------|------|----------|
| PRD | < 4.33% | 4.33-9.00% | 9.00-15.00% | ≥ 15.00% |
| WWPRD | < 7.4% | 7.4-14.8% | 14.8-24.7% | ≥ 24.7% |

---

## Training Configurations

### Configuration 1: Development (Fast Iteration)

**Goal**: Test code changes quickly

```bash
python scripts/train_mitbih.py \
    --num_records 2 \
    --epochs 5 \
    --batch_size 64 \
    --hidden_dims 16 32 64 \
    --latent_dim 16 \
    --output_dir ./outputs/dev
```

**Characteristics:**
- ⏱️ Time: ~3 minutes
- 📊 Data: ~1,300,000 samples → ~2,500 windows
- 🎯 Purpose: Code debugging, architecture testing

---

### Configuration 2: Quick Test (Verification)

**Goal**: Verify implementation works correctly

```bash
python scripts/train_mitbih.py \
    --num_records 3 \
    --epochs 10 \
    --batch_size 32 \
    --loss_type wwprd \
    --output_dir ./outputs/quick_test
```

**Characteristics:**
- ⏱️ Time: ~5-10 minutes
- 📊 Data: ~1,950,000 samples → ~3,800 windows
- 🎯 Purpose: Functionality verification

---

### Configuration 3: Standard Training (Week 1 Deliverable)

**Goal**: Generate Week 1 results and visualizations

```bash
python scripts/train_mitbih.py \
    --num_records 10 \
    --epochs 50 \
    --batch_size 32 \
    --loss_type wwprd \
    --weight_alpha 2.0 \
    --model_type conv \
    --hidden_dims 32 64 128 \
    --latent_dim 32 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --val_split 0.15 \
    --save_model \
    --output_dir ./outputs/week1
```

**Characteristics:**
- ⏱️ Time: ~30-60 minutes (GPU), ~2-3 hours (CPU)
- 📊 Data: ~6,500,000 samples → ~12,700 windows
- 💾 Model size: ~590 KB (147K parameters)
- 🎯 Purpose: **Week 1 deliverable**

**Expected Results:**
- Training converges to loss ~0.05-0.10
- PRD: 2-4% (Excellent range)
- WWPRD: 4-7% (Excellent range)
- SNR improvement: 10-15 dB

---

### Configuration 4: Full Training (Final Evaluation)

**Goal**: Maximum performance on complete dataset

```bash
python scripts/train_mitbih.py \
    --num_records 48 \
    --epochs 100 \
    --batch_size 32 \
    --loss_type wwprd \
    --weight_alpha 2.0 \
    --model_type residual \
    --hidden_dims 32 64 128 \
    --latent_dim 32 \
    --lr 1e-3 \
    --save_model \
    --output_dir ./outputs/full_training
```

**Characteristics:**
- ⏱️ Time: ~2-4 hours (GPU), ~12-24 hours (CPU)
- 📊 Data: ~31,200,000 samples → ~60,900 windows
- 💾 Model size: ~1.5 MB (387K parameters)
- 🎯 Purpose: Final benchmarking, Week 4 report

---

## Evaluation and Visualization

### Running Evaluation

After training, evaluate the model on test data:

```bash
python scripts/evaluate_mitbih.py \
    --model_path ./outputs/week1/best_model.pth \
    --config_path ./outputs/week1/config.json \
    --output_dir ./outputs/week1/evaluation \
    --num_samples 500
```

### Generated Visualizations

#### 1. Training Curves (`training_curves.png`)

**4-panel visualization:**

```
┌─────────────────────────────────┬─────────────────────────────────┐
│  Training & Validation Loss     │  PRD Evolution                  │
│  ● train_loss                   │  ─── train_prd                  │
│  ○ val_loss                     │  ─ ─ val_prd                    │
│  Shows convergence behavior     │  Threshold lines at 4.33%, 9%   │
└─────────────────────────────────┴─────────────────────────────────┘
┌─────────────────────────────────┬─────────────────────────────────┐
│  WWPRD Evolution                │  SNR Improvement                │
│  ─── train_wwprd                │  Bars showing dB gain           │
│  ─ ─ val_wwprd                  │  Input SNR vs Output SNR        │
│  Threshold lines at 7.4%, 14.8% │  Higher is better               │
└─────────────────────────────────┴─────────────────────────────────┘
```

#### 2. Metric Distributions (`metric_distributions.png`)

**4 histograms:**
- PRD distribution with quality threshold lines
- WWPRD distribution with quality threshold lines
- Input/Output SNR comparison
- SNR improvement distribution

#### 3. PRD vs WWPRD Scatter (`prd_wwprd_scatter.png`)

**Scatter plot showing:**
- Each point: one test sample
- X-axis: PRD value
- Y-axis: WWPRD value
- Color: SNR improvement (colorbar)
- Reference lines: Quality thresholds

**Interpretation:**
- Points in bottom-left: Excellent quality
- Tight clustering: Consistent performance
- Color gradient: SNR improvement correlation

#### 4. Quality Classification (`quality_classification.png`)

**Two pie charts:**
- Left: PRD quality distribution
- Right: WWPRD quality distribution

**Example:**
```
PRD Quality:              WWPRD Quality:
┌───────────────┐         ┌───────────────┐
│ Excellent 85% │         │ Excellent 78% │
│ Very Good 12% │         │ Very Good 18% │
│ Good 3%       │         │ Good 4%       │
│ Not Good 0%   │         │ Not Good 0%   │
└───────────────┘         └───────────────┘
```

#### 5. Reconstruction Gallery (`reconstruction_gallery.png`)

**8 example reconstructions:**
- Top 4: Best reconstructions (lowest PRD)
- Bottom 4: Worst reconstructions (highest PRD)

**Each example shows:**
```
Clean Signal:     ────────────  (blue)
Noisy Input:      ─ ─ ─ ─ ─    (orange)
Reconstructed:    ············  (green)

Metrics: PRD=2.34%, WWPRD=5.12%, SNR_imp=+12.5dB
```

---

## Understanding the Outputs

### Training Output Files

```
outputs/week1/
├── config.json                    # All hyperparameters
├── best_model.pth                 # Model checkpoint (PyTorch state_dict)
├── training_history.json          # Epoch-by-epoch metrics
├── final_metrics.json             # Final evaluation summary
├── training_curves.png            # 4-panel training visualization
└── reconstruction_examples.png    # Sample reconstructions (4 examples)
```

### Evaluation Output Files

```
outputs/week1/evaluation/
├── evaluation_metrics.json        # Per-sample metrics (500 samples)
├── metric_distributions.png       # Histograms
├── prd_wwprd_scatter.png         # Correlation plot
├── quality_classification.png     # Pie charts
└── reconstruction_gallery.png     # Best/worst examples (8 samples)
```

### Interpreting Results

#### Good Training Indicators

✅ **Loss curves:**
- Smooth descent (not oscillating)
- Val loss tracks train loss
- Converges to < 0.15

✅ **PRD/WWPRD:**
- Majority in "Excellent" range
- PRD < 4.33% for >70% samples
- WWPRD < 7.4% for >70% samples

✅ **SNR improvement:**
- Positive for all samples
- Average > 10 dB
- Consistent across samples

#### Warning Signs

⚠️ **Overfitting:**
- Val loss increases while train loss decreases
- Large gap between train/val metrics
- **Solution**: Reduce model size, add dropout, or increase data

⚠️ **Underfitting:**
- Both train/val loss plateau high
- Poor PRD/WWPRD scores
- **Solution**: Increase model capacity, train longer, tune learning rate

⚠️ **Unstable training:**
- Loss oscillates wildly
- NaN values appear
- **Solution**: Reduce learning rate, check data normalization

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Cannot download MIT-BIH records"

**Symptoms:**
```
Error: Failed to download record 100
ConnectionError: ...
```

**Solutions:**
1. **Check internet connection**: PhysioNet requires internet access
2. **Try manual download**:
   ```python
   import wfdb
   wfdb.dl_database('mitdb', dl_dir='./data/mitbih')
   ```
3. **Use VPN** if institutional firewall blocks PhysioNet

#### Issue 2: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 256.00 MiB
```

**Solutions:**
1. **Reduce batch size**:
   ```bash
   --batch_size 16  # Instead of 32
   ```
2. **Reduce model size**:
   ```bash
   --hidden_dims 16 32 64 --latent_dim 16
   ```
3. **Use CPU** (slower but more memory):
   ```bash
   --device cpu
   ```

#### Issue 3: "Training is very slow"

**Solutions:**
1. **Verify GPU usage**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))
   ```
2. **Reduce num_workers** (if causing bottleneck):
   ```bash
   # Dataloader workers set internally, but can modify in code
   ```
3. **Use fewer records for testing**:
   ```bash
   --num_records 3 --epochs 10
   ```

#### Issue 4: "Loss becomes NaN"

**Symptoms:**
```
Epoch 5/50 - Loss: nan
```

**Solutions:**
1. **Reduce learning rate**:
   ```bash
   --lr 1e-4  # Instead of 1e-3
   ```
2. **Check data normalization**: Ensure windows are properly normalized
3. **Use gradient clipping** (modify in `train_mitbih.py`):
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

#### Issue 5: "Poor reconstruction quality (high PRD)"

**Solutions:**
1. **Train longer**:
   ```bash
   --epochs 100  # Instead of 50
   ```
2. **Use WWPRD loss**:
   ```bash
   --loss_type wwprd  # Instead of mse or prd
   ```
3. **Increase model capacity**:
   ```bash
   --model_type residual --hidden_dims 32 64 128 256
   ```
4. **Reduce compression** (larger latent dimension):
   ```bash
   --latent_dim 64  # Instead of 32
   ```

---

## Next Steps

### Week 2 Objectives

**Compression Ratio Analysis**

1. **Implement Quantization**:
   - 4-bit, 6-bit, 8-bit latent quantization
   - Quantization-aware training (optional)

2. **Calculate Compression Ratios**:
   - Theoretical CR: `original_bits / compressed_bits`
   - Actual CR with entropy coding

3. **Generate Rate-Distortion Curves**:
   - PRD vs CR (multiple latent dimensions)
   - WWPRD vs CR
   - Identify optimal operating points

4. **Deliverables**:
   - PRD-CR curves
   - WWPRD-CR curves
   - Quantization comparison table
   - Optimal configuration recommendations

### Week 3 Objectives

**Loss Function Ablation + VP Layer**

1. **Loss Function Comparison**:
   - Train with MSE, PRD, WWPRD, STFT-WWPRD
   - Compare convergence speed and final quality

2. **Variable Projection Layer**:
   - Implement adaptive VP layer
   - Replace standard convolutions
   - Compare performance and complexity

### Week 4 Objectives

**Final Report**

1. Train on full dataset (48 records)
2. Benchmark against literature methods
3. Prepare TDK report and slides
4. Create demo visualizations

---

## Additional Resources

### Key Equations

**PRD Formula:**
```
PRD = 100 × sqrt(Σ(x[n] - x̂[n])² / Σ(x[n]²))
```

**WWPRD Formula:**
```
WWPRD = 100 × sqrt(Σ(w[n]·(x[n] - x̂[n])²) / Σ(w[n]·x[n]²))

where: w[n] = 1 + α·|x'[n]| / max(|x'|)
```

**SNR Formula:**
```
SNR_dB = 10 × log₁₀(Σ(x[n]²) / Σ((x[n] - x̂[n])²))
```

**Compression Ratio:**
```
CR = (N_samples × bits_per_sample) / (latent_dim × latent_length × quantization_bits)
```

### References

**Papers:**
- Generalized Rational Variable Projection with Application in ECG Compression (provided in repo)

**Databases:**
- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/
- MIT-BIH Noise Stress Test Database: https://physionet.org/content/nstdb/

**Documentation:**
- wfdb Python library: https://wfdb.readthedocs.io/
- PyTorch: https://pytorch.org/docs/

---

## Support and Questions

For issues or questions:
1. Check this guide's Troubleshooting section
2. Review code comments in source files
3. Open a GitHub issue with:
   - Error message
   - Command used
   - System information (OS, GPU, Python version)

---

**Last Updated**: October 2025
**Week 1 Status**: ✅ COMPLETE
**Next Milestone**: Week 2 - Compression Ratio Analysis 🚀
