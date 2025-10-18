# ECG Denoising + Compression with WWPRD-based Optimization

This repository contains a deep autoencoder baseline for simultaneous ECG denoising and compression, trained directly with a differentiable waveform-weighted PRD (WWPRD) objective.

## 🎯 Project Goals

Develop an ECG signal compression algorithm that:
- Maintains diagnostic quality (PRD < 4.33%, WWPRD < 7.4% for "Excellent")
- Uses Variable Projection (VP) methods for adaptive compression
- Optimizes with differentiable WWPRD loss aligned to clinical evaluation metrics
- Achieves high compression ratios without losing critical diagnostic information

## ✨ Novelty (for TDK)
- Train with **differentiable WWPRD** aligned to evaluation metrics, rather than MSE
- Single model performs denoising and compression with controllable bottleneck
- NSTDB real noise integration for robust training
- Optional VP projection layer ablation (Week 3)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Setup Test
```bash
python scripts/test_setup.py
```

### 3. Train on MIT-BIH (Quick Test)
```bash
python scripts/train_mitbih.py \
    --num_records 5 \
    --epochs 20 \
    --batch_size 32 \
    --loss_type wwprd \
    --output_dir ./outputs/week1
```

### 4. Evaluate Model
```bash
python scripts/evaluate_mitbih.py \
    --model_path ./outputs/week1/best_model.pth \
    --config_path ./outputs/week1/config.json \
    --output_dir ./outputs/week1/evaluation
```

## 📁 Repository Structure

```
ecg-vp-denoising/
├── ecgdae/
│   ├── data.py          # MIT-BIH loader, NSTDB noise mixer, windowing
│   ├── losses.py        # PRD, WWPRD, STFT-weighted WWPRD losses
│   ├── models.py        # Convolutional and Residual autoencoders
│   └── metrics.py       # Evaluation metrics (PRD, WWPRD, SNR, CR)
├── scripts/
│   ├── train_mitbih.py      # Main training script
│   ├── evaluate_mitbih.py   # Comprehensive evaluation script
│   ├── train_synthetic.py   # Synthetic data demo
│   └── test_setup.py        # Setup verification test
├── requirements.txt     # Python dependencies
├── WEEK1_GUIDE.md      # Detailed Week 1 guide
└── README.md           # This file
```

## 📊 Week 1 Deliverables (✅ Completed)

### Implemented Features
- ✅ **MIT-BIH Data Loader**: Automatic download and processing of MIT-BIH Arrhythmia Database
- ✅ **NSTDB Noise Mixing**: Real ECG noise (muscle artifact, baseline wander, electrode motion)
- ✅ **WWPRD Weights**: Derivative-based weights emphasizing QRS complexes
- ✅ **Training Pipeline**: Complete training with loss curves and metrics
- ✅ **Evaluation Suite**: Comprehensive PRD/WWPRD analysis with visualizations

### Generated Outputs
- Training loss curves (train/val loss, PRD, WWPRD, SNR improvement)
- Reconstruction examples comparing clean, noisy, and denoised signals
- Metric distributions (PRD, WWPRD, SNR histograms)
- Quality classification pie charts (Excellent/Very Good/Good/Not Good)
- PRD vs WWPRD scatter plots with SNR improvement coloring
- Best/worst reconstruction gallery

## 📈 Quality Standards

### PRD (Percent Root-mean-square Difference)
| Quality | PRD Range | Diagnostic Use |
|---------|-----------|----------------|
| **Excellent** | < 4.33% | Fully usable for diagnosis |
| **Very Good** | 4.33% - 9.00% | Usable for diagnosis |
| **Good** | 9.00% - 15.00% | Limited diagnostic value |
| **Not Good** | ≥ 15.00% | Not recommended for diagnosis |

### WWPRD (Waveform-Weighted PRD)
| Quality | WWPRD Range | Diagnostic Use |
|---------|-------------|----------------|
| **Excellent** | < 7.4% | Fully usable for diagnosis |
| **Very Good** | 7.4% - 14.8% | Usable for diagnosis |
| **Good** | 14.8% - 24.7% | Limited diagnostic value |
| **Not Good** | ≥ 24.7% | Not recommended for diagnosis |

**Note**: WWPRD emphasizes QRS complex regions (ventricular depolarization) more than PRD, thus better reflects preservation of diagnostic information.

## 🗓️ Four-Week Plan

### Week 1: ✅ Data Pipeline + WWPRD Training (COMPLETED)
- ✅ MIT-BIH data loader with windowing
- ✅ NSTDB noise mixing with SNR control
- ✅ WWPRD weight computation and evaluation
- **Deliverables**: Training curves, PRD/WWPRD charts

### Week 2: 📅 Compression Ratio Analysis (PLANNED)
- [ ] Quantization of latent space (4-8 bits)
- [ ] Compression ratio (CR) calculation
- [ ] PRD-CR and WWPRD-CR curves
- **Deliverables**: PRD-CR, WWPRD-CR curves

### Week 3: 📅 Loss Ablation + VP Layer (PLANNED)
- [ ] Loss function comparison (MSE vs PRD vs WWPRD)
- [ ] Variable Projection (VP) layer implementation
- [ ] VP layer vs standard convolution comparison
- **Deliverables**: Loss comparison plots, VP layer ablation

### Week 4: 📅 Final Report (PLANNED)
- [ ] Full dataset training with optimal configuration
- [ ] Final performance benchmarks
- [ ] Report and presentation slides
- **Deliverables**: Final charts, report, slides

## 🔧 Key Components

### 1. Data Loading (`ecgdae/data.py`)
- **MITBIHLoader**: Downloads MIT-BIH from PhysioNet automatically
- **NSTDBNoiseMixer**: Realistic ECG noise (3 types: muscle artifact, baseline wander, electrode motion)
- **WindowingConfig**: Configurable window length and step size
- **MITBIHDataset**: PyTorch dataset with noise mixing

### 2. Loss Functions (`ecgdae/losses.py`)
- **PRDLoss**: Standard percent root-mean-square difference
- **WWPRDLoss**: Time-domain waveform-weighted PRD
- **STFTWeightedWWPRDLoss**: Frequency-domain weighted PRD
- All losses are fully differentiable for gradient-based optimization

### 3. Models (`ecgdae/models.py`)
- **ConvAutoEncoder**: Standard 1D convolutional autoencoder
- **ResidualAutoEncoder**: Improved gradient flow with residual blocks
- Configurable compression ratio via latent dimension

### 4. Evaluation Metrics (`ecgdae/metrics.py`)
- PRD, WWPRD computation
- SNR calculation and improvement measurement
- Compression ratio estimation
- Automatic quality classification

## 📖 Documentation

For detailed usage instructions, see [WEEK1_GUIDE.md](WEEK1_GUIDE.md)

## 📄 References

This project implements concepts from:
- "Generalized Rational Variable Projection with Application in ECG Compression"
- MIT-BIH Arrhythmia Database (PhysioNet)
- MIT-BIH Noise Stress Test Database (NSTDB)

## 📝 License

This is an academic project for research purposes.

---

**Status**: Week 1 Complete ✅ | Ready for Week 2 🚀
