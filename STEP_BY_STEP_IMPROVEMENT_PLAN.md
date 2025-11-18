# Step-by-Step Improvement Plan

## 🔍 Understanding the Discrepancy

### Why `final_metrics.json` Shows Lower Values Than Comparison Table?

**Your `final_metrics.json`:**
- PRD = 24.37%
- PRDN = 27.49%
- WWPRD = 18.88%

**Comparison Table (Records 117 & 119):**
- PRDN = 31.27%
- WWPRD = 22.42%

### The Reason:

1. **Different Data Splits:**
   - `final_metrics.json`: Validation set (15% random split from training records)
   - Comparison table: Specific records 117 & 119 (all windows from these records)

2. **Records 117 & 119 ARE in Your Training Set:**
   - Your config: `num_records = 20`
   - Training records: First 20 records = ['100', '101', ..., '117', '118', '119', ...]
   - Records 117 & 119 are records #17 and #19 in your training set

3. **Why They Perform Worse:**
   - Different noise realizations (each evaluation generates new noise)
   - All windows from these records (not just validation subset)
   - These records might be more challenging cases
   - Model may have overfitted to easier patterns in other records

### Which Records Were Used for Training?

**Training Records (first 20):**
```
['100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
 '111', '112', '113', '114', '115', '116', '117', '118', '119', '121']
```

**Validation:** 15% random split from these 20 records

**Test Records (if using standard evaluation):**
- Records 21-25: ['122', '123', '124', '200', '201']

---

## 🎯 Step-by-Step Improvement Plan

### **PHASE 1: Fix Compression Ratio (CR)** ⭐ **HIGHEST PRIORITY**

**Goal:** Achieve actual compression (CR > 1:1, target 8:1 to 16:1)

**Why First:** CR improvement has the biggest impact on QSN (10-20× improvement)

#### Step 1.1: Train Model with Smaller Latent Dimension

**Current:** `latent_dim = 32` → CR = 0.69:1 (expanded!)

**Target:** `latent_dim = 8` → CR ≈ 8:1

**Command:**
```powershell
$env:PYTHONPATH = "."

python scripts/train_mitbih.py `
    --data_dir ./data/mitbih `
    --num_records 20 `
    --window_seconds 2.0 `
    --sample_rate 360 `
    --noise_type nstdb `
    --snr_db 10.0 `
    --nstdb_noise muscle_artifact `
    --model_type residual `
    --hidden_dims 32 64 128 `
    --latent_dim 8 `
    --loss_type wwprd `
    --weight_alpha 2.0 `
    --batch_size 32 `
    --epochs 100 `
    --lr 0.0005 `
    --weight_decay 0.0001 `
    --val_split 0.15 `
    --output_dir outputs/cr8_wwprd_100e `
    --save_model `
    --device auto
```

**Expected Results:**
- CR: 0.69:1 → ~8:1 (11.6× improvement)
- QSN: 0.022 → ~0.26 (if PRDN stays 31%) or → ~1.14 (if PRDN improves to 7%)

**Time:** ~5-7 hours on CPU, ~1-2 hours on GPU

#### Step 1.2: Train Model for CR ≈ 16:1

**Command:**
```powershell
python scripts/train_mitbih.py `
    --data_dir ./data/mitbih `
    --num_records 20 `
    --model_type residual `
    --hidden_dims 32 64 128 `
    --latent_dim 4 `
    --loss_type wwprd `
    --epochs 100 `
    --output_dir outputs/cr16_wwprd_100e `
    --save_model `
    --device auto
```

**Expected Results:**
- CR: ~16:1
- QSN: ~0.51 (if PRDN=31%) or ~3.69 (if PRDN=4.33%)

#### Step 1.3: Verify CR After Training

**Check actual CR:**
```powershell
python scripts/calculate_qs_scores.py `
    --wwprd_model outputs/cr8_wwprd_100e/best_model.pth `
    --wwprd_config outputs/cr8_wwprd_100e/config.json `
    --combined_model outputs/cr8_wwprd_100e/best_model.pth `
    --combined_config outputs/cr8_wwprd_100e/config.json `
    --record_ids 117 119 `
    --quantization_bits 8 `
    --output outputs/cr8_verification.json
```

**Check:** CR should be ~8:1 (not 0.69:1)

---

### **PHASE 2: Improve PRDN/WWPRD** ⭐ **SECOND PRIORITY**

**Goal:** Reduce PRDN from 31% to <15% (ideally <7%)

#### Step 2.1: Train Longer (100-150 epochs)

**Why:** Your training curves show loss still decreasing at epoch 50

**Command:**
```powershell
python scripts/train_mitbih.py `
    --data_dir ./data/mitbih `
    --num_records 20 `
    --model_type residual `
    --hidden_dims 32 64 128 `
    --latent_dim 8 `
    --loss_type wwprd `
    --epochs 150 `
    --lr 0.0005 `
    --weight_decay 0.0001 `
    --output_dir outputs/cr8_wwprd_150e `
    --save_model `
    --device auto
```

**Expected Improvement:**
- PRDN: 31% → 25-28% (moderate improvement)
- WWPRD: 22% → 18-20%

#### Step 2.2: Use More Training Data (All 48 Records)

**Why:** More diverse data = better generalization

**Command:**
```powershell
python scripts/train_mitbih.py `
    --data_dir ./data/mitbih `
    --num_records 48 `
    --model_type residual `
    --hidden_dims 32 64 128 `
    --latent_dim 8 `
    --loss_type wwprd `
    --epochs 100 `
    --output_dir outputs/cr8_wwprd_48records `
    --save_model `
    --device auto
```

**Expected Improvement:**
- PRDN: 31% → 20-25% (better generalization)
- Lower variance across different records

#### Step 2.3: Tune Hyperparameters

**Try different learning rates:**

```powershell
# Lower learning rate (more stable, slower)
python scripts/train_mitbih.py `
    --latent_dim 8 `
    --lr 0.0002 `
    --epochs 150 `
    --output_dir outputs/cr8_wwprd_lr0.0002

# Higher learning rate (faster, may be less stable)
python scripts/train_mitbih.py `
    --latent_dim 8 `
    --lr 0.001 `
    --epochs 100 `
    --output_dir outputs/cr8_wwprd_lr0.001
```

**Try different weight decay:**
```powershell
python scripts/train_mitbih.py `
    --latent_dim 8 `
    --weight_decay 0.0005 `
    --epochs 100 `
    --output_dir outputs/cr8_wwprd_wd0.0005
```

#### Step 2.4: Try Combined Loss with Different Alpha

**Test different alpha values:**

```powershell
# More emphasis on PRDN (alpha = 0.7)
python scripts/train_mitbih.py `
    --latent_dim 8 `
    --loss_type combined `
    --combined_alpha 0.7 `
    --epochs 100 `
    --output_dir outputs/cr8_combined_alpha0.7

# More emphasis on WWPRD (alpha = 0.3)
python scripts/train_mitbih.py `
    --latent_dim 8 `
    --loss_type combined `
    --combined_alpha 0.3 `
    --epochs 100 `
    --output_dir outputs/cr8_combined_alpha0.3
```

---

### **PHASE 3: Comprehensive Evaluation** ⭐ **FOR PAPER**

#### Step 3.1: Evaluate on All Standard Records

**Evaluate on records commonly used in literature (117, 119, 201, 202, etc.):**

```powershell
python scripts/calculate_qs_scores.py `
    --wwprd_model outputs/cr8_wwprd_100e/best_model.pth `
    --wwprd_config outputs/cr8_wwprd_100e/config.json `
    --combined_model outputs/cr8_combined_alpha0.5/best_model.pth `
    --combined_config outputs/cr8_combined_alpha0.5/config.json `
    --record_ids 100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 117 118 119 121 122 123 124 200 201 202 203 205 207 208 209 210 212 213 214 215 217 219 220 221 222 223 228 230 231 232 `
    --quantization_bits 8 `
    --output outputs/full_comparison_table.json
```

#### Step 3.2: Generate Rate-Distortion Curves

**Evaluate at multiple CRs (4:1, 8:1, 16:1, 32:1):**

```powershell
# For CR=8:1 model
python scripts/evaluate_compression.py `
    --model_path outputs/cr8_wwprd_100e/best_model.pth `
    --config_path outputs/cr8_wwprd_100e/config.json `
    --compression_ratios 8 `
    --output_file outputs/cr8_evaluation.json

# For CR=16:1 model
python scripts/evaluate_compression.py `
    --model_path outputs/cr16_wwprd_100e/best_model.pth `
    --config_path outputs/cr16_wwprd_100e/config.json `
    --compression_ratios 16 `
    --output_file outputs/cr16_evaluation.json
```

#### Step 3.3: Compare with Baseline Methods

**Extract values from professor's paper Table IV and create comparison:**

1. Create `baseline_comparison.csv`:
```csv
Method,PRDN_avg,WWPRD_avg,CR_avg,QSN
Basic,7.35,13.69,15.40,2.10
Aligned,7.85,14.45,19.17,2.44
B-spline,7.70,14.56,12.28,1.59
AWT,7.22,15.50,8.57,1.19
AWPT,6.98,24.35,6.26,0.90
Hermite,9.22,19.16,9.31,1.01
Our_WWPRD,?,?,?,?
Our_Combined,?,?,?,?
```

2. Fill in your results from `full_comparison_table.json`

3. Create comparison plot/table for paper

---

### **PHASE 4: Paper Preparation** ⭐ **FINAL STEPS**

#### Step 4.1: Update Results Section

**Create table similar to Table IV with:**
- Your methods (WWPRD-only, Combined)
- Baseline methods from paper
- All metrics: PRDN, WWPRD, CR, QSN

#### Step 4.2: Update Abstract

**Based on your best results, update abstract:**
- Report actual CR achieved (e.g., "CR ≈ 8:1 to 16:1")
- Report QSN scores (e.g., "QSN = 1.14 to 3.69")
- Compare with baselines if competitive

#### Step 4.3: Create Visualizations

**Generate:**
1. Rate-distortion curves (PRDN vs CR, WWPRD vs CR)
2. QSN comparison bar chart
3. Reconstruction examples at different CRs
4. Training curves showing convergence

---

## 📊 Expected Timeline

### **Week 1: Fix CR**
- Day 1-2: Train CR=8:1 model (100 epochs)
- Day 3-4: Train CR=16:1 model (100 epochs)
- Day 5: Verify CR and calculate QS

**Expected:** QSN improves from 0.022 to 0.26-0.51

### **Week 2: Improve Quality**
- Day 1-3: Train longer (150 epochs)
- Day 4-5: Use all 48 records
- Day 6-7: Hyperparameter tuning

**Expected:** PRDN improves from 31% to 20-25%, QSN improves to 0.40-0.80

### **Week 3: Comprehensive Evaluation**
- Day 1-2: Evaluate on all records
- Day 3-4: Generate rate-distortion curves
- Day 5-7: Compare with baselines, prepare paper

**Expected:** QSN = 0.80-1.14 (acceptable) or 1.5+ (competitive)

---

## ✅ Success Criteria

### **Minimum Acceptable:**
- ✅ CR > 8:1 (actual compression)
- ✅ PRDN < 20%
- ✅ QSN > 0.5

### **Good (Publication-worthy):**
- ✅ CR = 8:1 to 16:1
- ✅ PRDN < 10%
- ✅ QSN > 1.0

### **Excellent (Competitive):**
- ✅ CR = 16:1
- ✅ PRDN < 7%
- ✅ QSN > 1.5 (competitive with paper)

---

## 🚀 Quick Start (Do This First!)

**Priority 1: Train CR=8:1 model (this alone will 10× improve QSN)**

```powershell
$env:PYTHONPATH = "."

python scripts/train_mitbih.py --data_dir ./data/mitbih --num_records 20 --window_seconds 2.0 --sample_rate 360 --noise_type nstdb --snr_db 10.0 --nstdb_noise muscle_artifact --model_type residual --hidden_dims 32 64 128 --latent_dim 8 --loss_type wwprd --weight_alpha 2.0 --batch_size 32 --epochs 100 --lr 0.0005 --weight_decay 0.0001 --val_split 0.15 --output_dir outputs/cr8_wwprd_100e --save_model --device auto
```

**After training completes, verify:**
```powershell
python scripts/calculate_qs_scores.py --wwprd_model outputs/cr8_wwprd_100e/best_model.pth --wwprd_config outputs/cr8_wwprd_100e/config.json --combined_model outputs/cr8_wwprd_100e/best_model.pth --combined_config outputs/cr8_wwprd_100e/config.json --record_ids 117 119 --quantization_bits 8 --output outputs/cr8_verification.json
```

**Expected:** CR ≈ 8:1, QSN ≈ 0.26-0.51 (10-20× improvement!)

---

## 📝 Notes

- **Training records:** First 20 records (100-121)
- **Records 117 & 119:** ARE in training set (records #17 and #19)
- **Discrepancy reason:** Different noise realizations and window selection
- **Focus:** Fix CR first (biggest impact), then improve quality

Good luck! 🎉


