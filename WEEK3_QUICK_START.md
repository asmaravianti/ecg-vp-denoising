# Week 3 Quick Start Guide

## What We Can Do While Waiting for Main Training

While the Residual model trains to 200 epochs (10-16 hours), we can prepare and start Week 3 experiments.

---

## ✅ Ready to Use Scripts

### 1. Loss Ablation Study

Compare MSE, PRD, and WWPRD loss functions at different compression ratios.

**Quick test (can start now):**
```powershell
# Test with fewer epochs first (30 epochs, ~2-3 hours)
python scripts/train_ablation.py `
    --loss_type mse `
    --latent_dim 16 `
    --epochs 30 `
    --num_records 20 `
    --model_type residual `
    --output_dir outputs/week3/loss_ablation/cr8_mse_test `
    --save_model
```

**Full experiments (after testing):**
```powershell
# CR ≈ 8 (latent_dim=16)
python scripts/train_ablation.py --loss_type mse --latent_dim 16 --epochs 150 --output_dir outputs/week3/loss_ablation/cr8_mse
python scripts/train_ablation.py --loss_type prd --latent_dim 16 --epochs 150 --output_dir outputs/week3/loss_ablation/cr8_prd
python scripts/train_ablation.py --loss_type wwprd --latent_dim 16 --epochs 150 --output_dir outputs/week3/loss_ablation/cr8_wwprd

# CR ≈ 16 (latent_dim=8)
python scripts/train_ablation.py --loss_type mse --latent_dim 8 --epochs 150 --output_dir outputs/week3/loss_ablation/cr16_mse
python scripts/train_ablation.py --loss_type prd --latent_dim 8 --epochs 150 --output_dir outputs/week3/loss_ablation/cr16_prd
python scripts/train_ablation.py --loss_type wwprd --latent_dim 8 --epochs 150 --output_dir outputs/week3/loss_ablation/cr16_wwprd
```

### 2. Noise Ablation Study

Compare training with vs without noise augmentation.

```powershell
# With noise (current setup)
python scripts/train_mitbih.py `
    --model_type residual `
    --num_records 20 `
    --epochs 150 `
    --loss_type wwprd `
    --noise_type nstdb `
    --snr_db 10.0 `
    --output_dir outputs/week3/noise_ablation/with_noise `
    --save_model

# Without noise (need to modify train_mitbih.py to support --noise_type none)
# Or use train_ablation.py with --no_noise flag
```

### 3. Bottleneck Sweep (Rate-Distortion Curve)

Train multiple models with different latent_dim values.

**Dry run first (see what will be trained):**
```powershell
python scripts/train_bottleneck_sweep.py `
    --latent_dims 8 12 16 20 24 32 `
    --epochs 150 `
    --dry_run
```

**Actual training:**
```powershell
python scripts/train_bottleneck_sweep.py `
    --latent_dims 8 12 16 20 24 32 `
    --epochs 150 `
    --output_dir outputs/week3/bottleneck_sweep
```

**Note:** This will train 6 models sequentially. Each takes 10-16 hours, so total ~60-96 hours.

### 4. Plot Results

After experiments complete, generate comparison plots:

```powershell
# Loss ablation plot
python scripts/plot_ablation_results.py `
    --ablation_type loss `
    --results_dir outputs/week3/loss_ablation `
    --output outputs/week3/plots/loss_ablation.png

# Rate-distortion curve
python scripts/plot_ablation_results.py `
    --ablation_type bottleneck `
    --results_dir outputs/week3/bottleneck_sweep `
    --output outputs/week3/plots/rate_distortion.png
```

---

## Recommended Order

### Phase 1: Quick Tests (While Main Training Runs)
1. ✅ Test loss ablation with 30 epochs (2-3 hours)
2. ✅ Verify scripts work correctly
3. ✅ Check output formats

### Phase 2: Full Experiments (After Main Training)
1. Loss ablation (6 experiments, can run sequentially)
2. Noise ablation (2 experiments)
3. Bottleneck sweep (6-8 experiments, sequential)

### Phase 3: VP Layer (Later)
1. Implement VP layer in `ecgdae/models.py`
2. Train and compare with standard conv

---

## Time Estimates

| Experiment | Models | Time per Model | Total Time |
|------------|--------|----------------|------------|
| Loss ablation (CR≈8) | 3 | 10-16h | 30-48h |
| Loss ablation (CR≈16) | 3 | 10-16h | 30-48h |
| Noise ablation | 2 | 10-16h | 20-32h |
| Bottleneck sweep | 6-8 | 10-16h | 60-128h |

**Total**: ~140-256 hours if running sequentially (can parallelize some)

---

## Next Steps

1. **Now**: Test loss ablation script with 30 epochs
2. **After main training**: Review results, then start full experiments
3. **Week 3**: Complete all ablations and implement VP layer

---

## Notes

- All scripts save results in structured format
- Can run experiments sequentially (overnight training)
- Results can be plotted together after all experiments complete
- VP layer implementation can be done in parallel with experiments

