# Week 3 Plan: Ablations + VP Layer

## Objectives

1. **Loss Ablation**: MSE vs PRD vs WWPRD at CR≈8 and CR≈16
2. **Noise Ablation**: Train with vs without augmentation (same CR, same loss)
3. **Bottleneck Sweep**: Verify monotonic rate–distortion behavior
4. **VP Layer Prototype**: Implement VP layer in first encoder block; run at one CR with WWPRD

## Deliverables

- Ablation plots/tables (loss, noise, bottleneck)
- VP vs conv comparison slide (WWPRD/SNR and an overlay)

---

## 1. Loss Ablation Study

### Objective
Compare training with different loss functions: MSE, PRD, WWPRD

### Experiments
- **Model**: ResidualAutoEncoder (or ConvAutoEncoder)
- **CR ≈ 8**: latent_dim = 16 (or appropriate value)
- **CR ≈ 16**: latent_dim = 8 (or appropriate value)
- **Loss functions**: MSE, PRD, WWPRD
- **Other settings**: Same (20 records, 150 epochs, lr=0.0005)

### Expected Output
- Training curves for each loss function
- Final PRD, WWPRD, SNR for each configuration
- Comparison table

### Commands
```powershell
# CR ≈ 8, MSE loss
python scripts/train_ablation.py --loss_type mse --latent_dim 16 --output_dir outputs/week3/loss_ablation/cr8_mse

# CR ≈ 8, PRD loss
python scripts/train_ablation.py --loss_type prd --latent_dim 16 --output_dir outputs/week3/loss_ablation/cr8_prd

# CR ≈ 8, WWPRD loss
python scripts/train_ablation.py --loss_type wwprd --latent_dim 16 --output_dir outputs/week3/loss_ablation/cr8_wwprd

# CR ≈ 16, MSE loss
python scripts/train_ablation.py --loss_type mse --latent_dim 8 --output_dir outputs/week3/loss_ablation/cr16_mse

# CR ≈ 16, PRD loss
python scripts/train_ablation.py --loss_type prd --latent_dim 8 --output_dir outputs/week3/loss_ablation/cr16_prd

# CR ≈ 16, WWPRD loss
python scripts/train_ablation.py --loss_type wwprd --latent_dim 8 --output_dir outputs/week3/loss_ablation/cr16_wwprd
```

---

## 2. Noise Ablation Study

### Objective
Compare training with noise augmentation vs without noise

### Experiments
- **Model**: ResidualAutoEncoder
- **CR**: Fixed (latent_dim = 32)
- **Loss**: WWPRD (fixed)
- **With noise**: SNR = 10 dB, NSTDB muscle_artifact
- **Without noise**: No noise augmentation

### Expected Output
- Training curves comparison
- Final metrics comparison
- Analysis of denoising capability

### Commands
```powershell
# With noise (current setup)
python scripts/train_mitbih.py --model_type residual --num_records 20 --epochs 150 --loss_type wwprd --noise_type nstdb --snr_db 10.0 --output_dir outputs/week3/noise_ablation/with_noise

# Without noise
python scripts/train_mitbih.py --model_type residual --num_records 20 --epochs 150 --loss_type wwprd --noise_type none --output_dir outputs/week3/noise_ablation/without_noise
```

---

## 3. Bottleneck Sweep (Rate-Distortion Curve)

### Objective
Verify monotonic rate–distortion behavior by training models with different latent_dim values

### Experiments
- **Model**: ResidualAutoEncoder
- **Loss**: WWPRD
- **latent_dim values**: [8, 12, 16, 20, 24, 32, 40, 48]
- **Other settings**: Same (20 records, 150 epochs)

### Expected Output
- PRD vs CR curve (should be monotonic: higher CR → higher PRD)
- WWPRD vs CR curve
- Rate-distortion table

### Commands
```powershell
# Create script to train multiple models
python scripts/train_bottleneck_sweep.py --latent_dims 8 12 16 20 24 32 40 48 --output_dir outputs/week3/bottleneck_sweep
```

---

## 4. VP Layer Prototype

### Objective
Implement Variable Projection (VP) layer and compare with standard convolution

### Implementation Plan
1. Create VP layer class in `ecgdae/models.py`
2. Modify ResidualAutoEncoder to support VP in first encoder block
3. Train VP model and standard conv model at same CR
4. Compare performance

### Expected Output
- VP layer implementation
- VP vs conv comparison (WWPRD, SNR)
- Overlay visualization

### Commands
```powershell
# Standard conv model
python scripts/train_mitbih.py --model_type residual --latent_dim 32 --output_dir outputs/week3/vp_comparison/conv

# VP model (once implemented)
python scripts/train_mitbih.py --model_type residual_vp --latent_dim 32 --output_dir outputs/week3/vp_comparison/vp
```

---

## Implementation Priority

### Phase 1: Prepare Scripts (While Training)
1. ✅ Create ablation training script
2. ✅ Create bottleneck sweep script
3. ✅ Create comparison visualization script

### Phase 2: Run Experiments (After Training)
1. Loss ablation (6 experiments)
2. Noise ablation (2 experiments)
3. Bottleneck sweep (8 experiments)

### Phase 3: VP Layer (Later)
1. Implement VP layer
2. Train and compare

---

## Time Estimates

- **Loss ablation**: 6 × 10-16 hours = 60-96 hours (can run sequentially)
- **Noise ablation**: 2 × 10-16 hours = 20-32 hours
- **Bottleneck sweep**: 8 × 10-16 hours = 80-128 hours (can run sequentially)
- **VP layer**: Implementation + 10-16 hours training

**Total**: ~1-2 weeks if running sequentially (overnight training)

---

## Next Steps

1. Create `scripts/train_ablation.py` for loss ablation
2. Create `scripts/train_bottleneck_sweep.py` for rate-distortion sweep
3. Create `scripts/plot_ablation_results.py` for visualization
4. Start with loss ablation (can run while waiting for main training)

