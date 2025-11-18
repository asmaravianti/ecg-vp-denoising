# Team Division Plan: Achieving QS > 1

## Recommended Approach: Strategy 3 (Combined Strategy)

**Why Strategy 3?**
- Simultaneously increases CR and reduces PRD (highest success rate)
- Two directions in parallel, saves time
- Can compare effectiveness of different strategies

## Two-Person Division

### 👤 Teammate A: Train Small Model (Increase CR)

**Task:** Train `latent_dim=4` model to achieve higher CR

**Command:**
```powershell
# Train new model (latent_dim=4, 100 epochs)
python scripts/train_mitbih.py `
    --loss_type wwprd `
    --latent_dim 4 `
    --epochs 100 `
    --num_records 20 `
    --batch_size 32 `
    --lr 0.0005 `
    --weight_decay 0.0001 `
    --output_dir outputs/wwprd_latent4_highcr `
    --save_model `
    --device auto
```

**Expected Results:**
- CR ≈ 10-15:1 (2-3x higher than current 5.5:1)
- PRD ≈ 30-40% (may be slightly higher, but CR increases more)
- **Target: QS = 12/35 = 0.34 → If PRD drops to 20%, QS = 12/20 = 0.6**

**Time Estimate:** 2-4 hours (CPU) or 1-2 hours (GPU)

---

### 👤 Teammate B: Extend Existing Model Training (Reduce PRD)

**Task:** Continue training `latent_dim=8` model from 50 to 100-150 epochs

**Command:**
```powershell
# Continue training existing model (from 50 to 100 epochs)
python scripts/train_mitbih.py `
    --loss_type wwprd `
    --latent_dim 8 `
    --epochs 100 `
    --num_records 20 `
    --batch_size 32 `
    --lr 0.0005 `
    --weight_decay 0.0001 `
    --output_dir outputs/wwprd_latent8_improved `
    --resume outputs/wwprd_latent8/best_model.pth `
    --save_model `
    --device auto
```

**Expected Results:**
- CR ≈ 5.5:1 (unchanged)
- PRD ≈ 20-25% (reduced from 35%)
- **Target: If PRD drops to 20%, QS = 5.5/20 = 0.275 → If drops to 10%, QS = 5.5/10 = 0.55**

**Time Estimate:** 2-3 hours (CPU) or 1-1.5 hours (GPU)

---

## Why This Division?

### ✅ Advantages:

1. **Parallel Work:** Both teammates train simultaneously, saves time
2. **Different Strategies:** Can compare which method is more effective
3. **Complementary:** One increases CR, one reduces PRD
4. **Flexibility:** If one method works well, can continue optimizing

### 📊 Expected Results Comparison:

| Strategy | Executor | CR | PRD | QS | Achieve QS>1? |
|----------|----------|----|----|----|---------------|
| Small Model (latent_dim=4) | Teammate A | 12:1 | 30% | 0.40 | ❌ Need PRD<12% |
| Extended Training (latent_dim=8) | Teammate B | 5.5:1 | 20% | 0.28 | ❌ Need PRD<5.5% |
| **Combined** | Both | 12:1 | 15% | **0.80** | ⚠️ Close |
| **Ideal Combined** | Both | 15:1 | 10% | **1.50** | ✅ Achieved |

---

## More Aggressive Approach (If Time Permits)

### Teammate A Enhanced:
```powershell
# Train smaller model + longer time
python scripts/train_mitbih.py `
    --loss_type wwprd `
    --latent_dim 4 `
    --epochs 150 `  # Longer training
    --num_records 20 `
    --lr 0.0003 `   # Smaller learning rate
    --output_dir outputs/wwprd_latent4_optimized
```

### Teammate B Enhanced:
```powershell
# Use more data + longer time
python scripts/train_mitbih.py `
    --loss_type wwprd `
    --latent_dim 8 `
    --epochs 150 `
    --num_records 30 `  # More records
    --lr 0.0003 `
    --resume outputs/wwprd_latent8/best_model.pth `
    --output_dir outputs/wwprd_latent8_optimized
```

---

## Evaluation and Comparison

After training, both teammates run evaluation:

```powershell
# Teammate A evaluation
python scripts/evaluate_compression.py `
    --model_path outputs/wwprd_latent4_highcr/best_model.pth `
    --config_path outputs/wwprd_latent4_highcr/config.json `
    --compression_ratios 8 16 32 `
    --quantization_bits 4 `
    --output_file outputs/week2/wwprd_latent4_results.json

# Teammate B evaluation
python scripts/evaluate_compression.py `
    --model_path outputs/wwprd_latent8_improved/best_model.pth `
    --config_path outputs/wwprd_latent8_improved/config.json `
    --compression_ratios 4 8 16 `
    --quantization_bits 4 `
    --output_file outputs/week2/wwprd_latent8_improved_results.json

# Generate comparison table
python calculate_qs_scores.py --results_dir outputs/week2 --output_dir outputs/week2
```

---

## Final Recommendation

**Recommended Division:**
- **Teammate A**: Train `latent_dim=4` (100 epochs) - Focus on increasing CR
- **Teammate B**: Extend `latent_dim=8` (100 epochs) - Focus on reducing PRD

**Reasons:**
1. Parallel work, saves time
2. Can compare two strategies
3. If one works well, can continue optimizing
4. Can combine best results from both methods

**Time Schedule:**
- Training: 2-4 hours (can run simultaneously)
- Evaluation: 10-20 minutes
- Comparison analysis: 30 minutes

**If QS > 1 still not achieved:**
- Continue training to 150-200 epochs
- Or try `latent_dim=2` (more aggressive compression)
