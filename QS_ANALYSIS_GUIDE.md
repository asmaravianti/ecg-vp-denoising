# Quality Score (QS) Analysis Guide

## 📚 Understanding Table IV from Professor's Paper

### What is Table IV?

**Table IV** is a comprehensive comparison table that evaluates different **Variable Projection (VP) methods** on the MIT-BIH Arrhythmia dataset. It shows how well each method performs across multiple ECG records.

### Table Structure Explained

```
┌─────────┬──────────┬──────────────┬──────────────┬──────────┐
│ Record  │ Method   │ PRDN (%)    │ WWPRD (%)   │ CR 1:X │
├─────────┼──────────┼──────────────┼──────────────┼────────┤
│ 100    │ Basic    │ 6.23 (V)    │ 12.45 (V)   │ 15.2  │
│ 100    │ Aligned  │ 7.11 (V)    │ 13.22 (V)   │ 18.5  │
│ ...    │ ...      │ ...         │ ...         │ ...   │
├─────────┼──────────┼──────────────┼──────────────┼────────┤
│ Average│ Basic    │ 7.35 (V)    │ 13.69 (V)   │ 15.40 │
│ Average│ Aligned  │ 7.85 (G)    │ 14.45 (V)   │ 19.17 │
└─────────┴──────────┴──────────────┴──────────────┴────────┘
```

### What Each Column Means:

1. **Record**: MIT-BIH ECG record ID (100, 101, ..., 232)
2. **Method**: Different VP algorithm variants:
   - **Basic**: Standard VP method
   - **Aligned**: VP with alignment optimization
   - **B-spline**: VP using B-spline basis
   - **Hermite**: VP using Hermite polynomials
   - **AWT**: Adaptive Wavelet Transform
   - **AWPT**: Adaptive Wavelet Packet Transform

3. **PRDN (%)**: Percent Root-mean-square Difference Normalized
   - **Formula**: `PRDN = 100 × √[Σ(x - x̂)² / Σ(x - x̄)²]`
   - **Meaning**: Reconstruction error normalized by signal variance
   - **Lower is better**
   - **Quality Bands**:
     - < 4.33% = Excellent (E)
     - 4.33-9% = Very Good (V)
     - 9-15% = Good (G)
     - ≥ 15% = Fair (F)

4. **WWPRD (%)**: Waveform-Weighted PRD
   - **Formula**: `WWPRD = 100 × √[Σw(t)(x - x̂)² / Σw(t)x²]`
   - **Meaning**: PRD weighted to emphasize QRS complexes (diagnostically important)
   - **Lower is better**
   - **Quality Bands**:
     - < 7.4% = Excellent (E)
     - 7.4-14.8% = Very Good (V)
     - 14.8-24.7% = Good (G)
     - ≥ 24.7% = Fair (F)

5. **CR 1:X**: Compression Ratio
   - **Formula**: `CR = Original Size / Compressed Size`
   - **Meaning**: How much the signal is compressed
   - **Example**: CR 1:15.4 means original is 15.4× larger than compressed
   - **Higher is better** (more compression)

6. **Letters in Parentheses**: Quality category
   - **(E)** = Excellent
   - **(V)** = Very Good
   - **(G)** = Good
   - **(F)** = Fair

### Summary Rows:

- **Average**: Mean value across all records
- **Std**: Standard deviation (shows variability)

---

## 🎯 Understanding Quality Score (QS)

### What is QS?

**Quality Score (QS)** is a **single metric** that balances compression and quality. It's defined in the professor's paper as:

```
QS = CR / PRD
QSN = CR / PRDN
```

### Why QS Matters:

Even if your **PRD is high** (e.g., 25%), you can still have a **good QS** if your **CR is high** (e.g., 20:1):

```
QS = 20 / 25 = 0.8  (decent)
QS = 20 / 10 = 2.0  (better)
QS = 20 / 5  = 4.0  (excellent)
```

**Higher QS = Better overall performance** (good compression + acceptable quality)

### Example from Paper:

| Method | PRDN (%) | CR 1:X | QSN = CR/PRDN |
|--------|----------|--------|---------------|
| Basic  | 7.35     | 15.40  | 15.40/7.35 = **2.10** |
| Aligned| 7.85     | 19.17  | 19.17/7.85 = **2.44** |
| AWPT   | 6.98     | 6.26   | 6.26/6.98 = **0.90** |

**Aligned has the best QSN** (2.44) because it achieves high compression (19.17:1) with acceptable quality (7.85% PRDN).

---

## 📋 Step-by-Step Action Plan

### **Phase 1: Calculate Current CR and QS** (1-2 hours)

#### Step 1.1: Understand Your Current Setup

Your models are trained with:
- **Latent dimension**: 32
- **Window length**: 512 samples (2 seconds @ 360 Hz)
- **Quantization**: 8 bits (default)

**Current CR calculation:**
```
Original bits = 512 samples × 11 bits/sample = 5,632 bits
Latent shape = (32 channels, 32 length) = 1,024 values
Compressed bits = 1,024 × 8 bits = 8,192 bits
CR = 5,632 / 8,192 ≈ 0.69:1  (NOT compressed! Actually expanded)
```

**Problem**: Your current setup doesn't achieve compression! You need to:
1. Reduce latent dimension (e.g., 16, 24)
2. Use fewer quantization bits (e.g., 4, 6)
3. Or both

#### Step 1.2: Run QS Calculation Script

```bash
python scripts/calculate_qs_scores.py \
    --wwprd_model outputs/loss_comparison_wwprd/best_model.pth \
    --wwprd_config outputs/loss_comparison_wwprd/config.json \
    --combined_model outputs/loss_comparison_combined_alpha0.5/best_model.pth \
    --combined_config outputs/loss_comparison_combined_alpha0.5/config.json \
    --record_ids 100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 117 118 119 121 122 123 124 200 201 202 203 205 207 208 209 210 212 213 214 215 217 219 220 221 222 223 228 230 231 232 \
    --quantization_bits 8 \
    --output outputs/comparison_table.json
```

This will:
- Evaluate both models on all MIT-BIH records
- Calculate PRDN, WWPRD, CR, QS, QSN for each record
- Generate a comparison table
- Save results to JSON

**Expected Output:**
- Table showing WWPRD-only vs Combined loss for each record
- Average metrics across all records
- QS and QSN scores

---

### **Phase 2: Achieve Target Compression Ratios** (2-3 hours)

#### Step 2.1: Train Models with Different Latent Dimensions

To achieve different CRs, you need models with different `latent_dim`:

| Target CR | Latent Dim | Quantization Bits | Expected CR |
|-----------|------------|-------------------|-------------|
| 4:1       | 64         | 8                 | ~4.4:1      |
| 8:1       | 32         | 8                 | ~8.8:1      |
| 16:1      | 16         | 8                 | ~17.6:1     |
| 32:1      | 8          | 8                 | ~35.2:1     |

**Or use quantization to adjust:**

| Target CR | Latent Dim | Quantization Bits | Expected CR |
|-----------|------------|-------------------|-------------|
| 4:1       | 32         | 4                 | ~4.4:1      |
| 8:1       | 32         | 6                 | ~5.9:1      |
| 16:1      | 32         | 8                 | ~8.8:1      |
| 32:1      | 32         | 4                 | ~17.6:1     |

#### Step 2.2: Train Multiple Models

**Option A: Different Latent Dimensions** (Recommended)

```bash
# Train model for CR ≈ 8:1
python scripts/train_mitbih.py \
    --model_type residual \
    --latent_dim 32 \
    --loss_type wwprd \
    --output_dir outputs/cr8_wwprd \
    --epochs 50

# Train model for CR ≈ 16:1
python scripts/train_mitbih.py \
    --model_type residual \
    --latent_dim 16 \
    --loss_type wwprd \
    --output_dir outputs/cr16_wwprd \
    --epochs 50

# Train model for CR ≈ 32:1
python scripts/train_mitbih.py \
    --model_type residual \
    --latent_dim 8 \
    --loss_type wwprd \
    --output_dir outputs/cr32_wwprd \
    --epochs 50
```

**Option B: Use Quantization** (Faster, but may affect quality)

Keep one model, evaluate with different quantization bits:
- 4 bits → higher CR, lower quality
- 6 bits → medium CR, medium quality
- 8 bits → lower CR, higher quality

#### Step 2.3: Evaluate at Each CR

For each model/quantization setting, run:

```bash
python scripts/evaluate_compression.py \
    --model_path outputs/cr8_wwprd/best_model.pth \
    --config_path outputs/cr8_wwprd/config.json \
    --compression_ratios 8 \
    --output_file outputs/cr8_results.json
```

---

### **Phase 3: Generate Comparison Table** (1 hour)

#### Step 3.1: Collect Results for All Methods

You should have results for:
1. **WWPRD-only** at different CRs
2. **Combined loss** at different CRs
3. **Different latent dimensions** (if you trained multiple models)

#### Step 3.2: Create Table Similar to Table IV

Use the `calculate_qs_scores.py` script output to create a table with:

**Columns:**
- Record ID
- Method (WWPRD-only, Combined, etc.)
- PRDN (%)
- WWPRD (%)
- CR 1:X
- QSN
- QS

**Rows:**
- One row per record-method combination
- Summary rows with averages

#### Step 3.3: Format for Paper

Create a LaTeX table or formatted table similar to the professor's paper:

```latex
\begin{table}[h]
\centering
\caption{Compression Results: WWPRD vs Combined Loss}
\begin{tabular}{ccccccc}
\hline
Record & Method & PRDN (\%) & WWPRD (\%) & CR 1:X & QSN & QS \\
\hline
100 & WWPRD-only & 26.66 & 19.00 & 8.8 & 0.33 & 0.33 \\
100 & Combined & 25.65 & 19.12 & 8.8 & 0.34 & 0.34 \\
... & ... & ... & ... & ... & ... & ... \\
\hline
Average & WWPRD-only & 26.66 & 19.00 & 8.8 & 0.33 & 0.33 \\
Average & Combined & 25.65 & 19.12 & 8.8 & 0.34 & 0.34 \\
\hline
\end{tabular}
\end{table}
```

---

### **Phase 4: Compare with Baseline Methods** (2-3 hours)

#### Step 4.1: Extract Baseline Values from Paper

From Table IV in the professor's paper, extract:
- PRDN averages for each method
- WWPRD averages for each method
- CR averages for each method
- Calculate QSN for each method

#### Step 4.2: Add Your Results to Comparison

Create a comparison table showing:
- Your methods (WWPRD-only, Combined)
- Baseline methods from paper (Basic, Aligned, B-spline, etc.)
- Highlight which method has best QSN

#### Step 4.3: Analyze Results

**Questions to answer:**
1. Does your method achieve competitive QSN?
2. Which method (WWPRD vs Combined) performs better?
3. How does your CR compare to baselines?
4. Is your PRDN/WWPRD acceptable given your CR?

---

### **Phase 5: Update Abstract and Paper** (1-2 hours)

#### Step 5.1: Update Abstract

Based on your QS results, update the abstract to:
- Report actual CR achieved
- Report QSN scores
- Compare with baseline methods (if competitive)
- Frame results appropriately (if QS is good but PRD is high, emphasize QS)

#### Step 5.2: Add Results Section

Create a section in your paper with:
- Table similar to Table IV
- QS/QSN analysis
- Comparison with baselines
- Discussion of trade-offs

---

## 🎯 Quick Reference: Key Formulas

### Compression Ratio (CR)
```
CR = Original Bits / Compressed Bits
Original Bits = Signal Length × 11 bits/sample
Compressed Bits = Latent Channels × Latent Length × Quantization Bits
```

### Quality Score (QS)
```
QS = CR / PRD
QSN = CR / PRDN
```

### Example Calculation

**Given:**
- Signal: 512 samples
- Latent: 32 channels × 32 length = 1,024 values
- Quantization: 8 bits
- PRDN: 25.65%

**Calculate CR:**
```
Original = 512 × 11 = 5,632 bits
Compressed = 1,024 × 8 = 8,192 bits
CR = 5,632 / 8,192 = 0.6875:1  (NOT compressed!)
```

**To achieve CR = 8:1:**
```
Need: Compressed = 5,632 / 8 = 704 bits
Options:
- Latent = 16 × 32 = 512 values, 8 bits = 4,096 bits (still too high)
- Latent = 32 × 32 = 1,024 values, 4 bits = 4,096 bits (still too high)
- Latent = 16 × 16 = 256 values, 8 bits = 2,048 bits (still too high)
- Latent = 8 × 32 = 256 values, 8 bits = 2,048 bits (still too high)
- Latent = 16 × 16 = 256 values, 4 bits = 1,024 bits (closer)
- Latent = 8 × 16 = 128 values, 8 bits = 1,024 bits (closer)
- Latent = 8 × 8 = 64 values, 8 bits = 512 bits (CR ≈ 11:1, too high)
```

**Better approach**: Train models with smaller latent dimensions!

---

## ✅ Checklist

- [ ] Run `calculate_qs_scores.py` to get current QS values
- [ ] Identify target CRs (4:1, 8:1, 16:1, 32:1)
- [ ] Train models with appropriate latent dimensions OR use quantization
- [ ] Evaluate all models on MIT-BIH records
- [ ] Calculate QS/QSN for all methods
- [ ] Generate comparison table (similar to Table IV)
- [ ] Extract baseline values from professor's paper
- [ ] Compare your results with baselines
- [ ] Update abstract with QS results
- [ ] Add results section to paper with table
- [ ] Discuss trade-offs and conclusions

---

## 🚀 Quick Start Command

To immediately calculate QS for your existing models:

```bash
python scripts/calculate_qs_scores.py \
    --wwprd_model outputs/loss_comparison_wwprd/best_model.pth \
    --wwprd_config outputs/loss_comparison_wwprd/config.json \
    --combined_model outputs/loss_comparison_combined_alpha0.5/best_model.pth \
    --combined_config outputs/loss_comparison_combined_alpha0.5/config.json \
    --record_ids 117 119 \
    --quantization_bits 8 \
    --output outputs/qs_analysis.json
```

This will give you QS scores for records 117 and 119 (commonly used in comparisons).

---

## 📊 Expected Results Format

After running the script, you'll get:

```json
{
  "wwprd_results": {
    "117": {
      "PRDN": 26.66,
      "WWPRD": 19.00,
      "CR": 8.8,
      "QS": 0.33,
      "QSN": 0.33
    },
    ...
  },
  "combined_results": {
    "117": {
      "PRDN": 25.65,
      "WWPRD": 19.12,
      "CR": 8.8,
      "QS": 0.34,
      "QSN": 0.34
    },
    ...
  },
  "averages": {
    "wwprd": {
      "PRDN_avg": 26.66,
      "WWPRD_avg": 19.00,
      "CR_avg": 8.8,
      "QSN_avg": 0.33
    },
    "combined": {
      "PRDN_avg": 25.65,
      "WWPRD_avg": 19.12,
      "CR_avg": 8.8,
      "QSN_avg": 0.34
    }
  }
}
```

---

## 💡 Key Insights

1. **QS balances compression and quality**: Even with high PRD, good CR can give good QS
2. **Your current CR is too low**: Need to reduce latent dimension or quantization bits
3. **Compare QSN, not just PRD**: QSN shows overall performance
4. **Table IV format is standard**: Use similar structure for your paper
5. **Averages matter**: Report both per-record and average metrics

---

Good luck! 🎉


