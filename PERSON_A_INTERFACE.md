# Person A: Data Format for Person B

**From:** Person B (Visualization)  
**To:** Person A (Quantization & CR Sweep)

---

## 📋 What Person B Needs From You

Person B has completed all visualization code and is waiting for your CR sweep results!

### **Required Output File:**

**File path:** `outputs/week2/cr_sweep_results.json`

**Format:**
```json
{
  "4": {
    "PRD": 35.2,
    "PRD_std": 8.5,
    "WWPRD": 30.1,
    "WWPRD_std": 7.8,
    "SNR_in": 6.0,
    "SNR_out": 11.2,
    "SNR_improvement": 5.2,
    "latent_dim": 16,
    "quantization_bits": 8
  },
  "8": {
    "PRD": 28.5,
    "PRD_std": 7.2,
    "WWPRD": 24.3,
    "WWPRD_std": 6.5,
    "SNR_in": 6.0,
    "SNR_out": 12.1,
    "SNR_improvement": 6.1,
    "latent_dim": 24,
    "quantization_bits": 8
  },
  "16": {
    "PRD": 22.1,
    "PRD_std": 5.8,
    "WWPRD": 18.7,
    "WWPRD_std": 5.2,
    "SNR_in": 6.0,
    "SNR_out": 13.8,
    "SNR_improvement": 7.8,
    "latent_dim": 32,
    "quantization_bits": 8
  },
  "32": {
    "PRD": 18.3,
    "PRD_std": 4.5,
    "WWPRD": 15.2,
    "WWPRD_std": 4.1,
    "SNR_in": 6.0,
    "SNR_out": 15.5,
    "SNR_improvement": 9.5,
    "latent_dim": 48,
    "quantization_bits": 8
  }
}
```

---

## 📊 Required Metrics Per CR

For each compression ratio (4, 8, 16, 32), provide:

| Metric | Type | Description |
|--------|------|-------------|
| `PRD` | float | Average PRD across all test samples (%) |
| `PRD_std` | float | Standard deviation of PRD |
| `WWPRD` | float | Average WWPRD across all test samples (%) |
| `WWPRD_std` | float | Standard deviation of WWPRD |
| `SNR_in` | float | Average input SNR (noisy signal, dB) |
| `SNR_out` | float | Average output SNR (reconstructed, dB) |
| `SNR_improvement` | float | `SNR_out - SNR_in` (dB) |
| `latent_dim` | int | Bottleneck dimension used for this CR |
| `quantization_bits` | int | Number of bits per latent value (4, 6, or 8) |

---

## 🔧 How to Generate This File

### **Option 1: From your evaluation script**

```python
import json
from pathlib import Path

# After running CR sweep experiments
results = {}

for cr in [4, 8, 16, 32]:
    # Your evaluation code here that computes:
    # prd_list, wwprd_list, snr_in_list, snr_out_list
    
    results[str(cr)] = {  # Use STRING keys for JSON
        "PRD": float(np.mean(prd_list)),
        "PRD_std": float(np.std(prd_list)),
        "WWPRD": float(np.mean(wwprd_list)),
        "WWPRD_std": float(np.std(wwprd_list)),
        "SNR_in": float(np.mean(snr_in_list)),
        "SNR_out": float(np.mean(snr_out_list)),
        "SNR_improvement": float(np.mean(snr_out_list) - np.mean(snr_in_list)),
        "latent_dim": latent_dim_used,
        "quantization_bits": quantization_bits_used
    }

# Save to JSON
output_dir = Path("outputs/week2")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "cr_sweep_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Saved CR sweep results to {output_dir / 'cr_sweep_results.json'}")
```

### **Option 2: Manual creation (for testing)**

1. Run your evaluation at each CR
2. Record the metrics in a text file
3. Format into JSON using the template above
4. Save to `outputs/week2/cr_sweep_results.json`

---

## ✅ Verification Checklist

Before sending to Person B, verify:

- [ ] File exists at `outputs/week2/cr_sweep_results.json`
- [ ] JSON is valid (use https://jsonlint.com/)
- [ ] All 4 CRs are present (4, 8, 16, 32)
- [ ] All metrics are present for each CR
- [ ] PRD and WWPRD are in reasonable ranges (1-50%)
- [ ] SNR values are in dB (typically 0-30 dB)
- [ ] SNR_improvement is positive (should be > 0)
- [ ] Keys are strings: `"4"` not `4`
- [ ] Values are numbers: `35.2` not `"35.2"`

---

## 🔗 Integration Test

After creating the file, Person B will run:

```bash
python -m scripts.plot_rate_distortion \
    --results_file outputs/week2/cr_sweep_results.json \
    --output_dir outputs/week2_final
```

This should generate:
- `rate_distortion_curves.png`
- `snr_bar_chart.png`
- `reconstruction_overlay_cr8.png`
- `reconstruction_overlay_cr16.png`
- `multi_cr_comparison.png`
- `week2_summary.png`
- `week2_visualization_summary.json`

**If errors occur, check:**
1. JSON format is correct
2. All required metrics are present
3. Metric names match exactly (case-sensitive!)

---

## 📞 Questions for Person B

If you need clarification:

1. **"What CR values should I use?"**
   - Use: 4, 8, 16, 32 (powers of 2)
   - These are standard in ECG compression literature

2. **"How do I calculate CR?"**
   - `CR = original_bits / compressed_bits`
   - Example: `CR = (512 * 11) / (32 * 32 * 4) = 1.38:1`

3. **"What if I get different CRs than expected?"**
   - Report the ACTUAL CR you achieved
   - Person B's plots will use your reported values

4. **"Should I include additional metrics?"**
   - Yes! Add any extra metrics to the JSON
   - Person B will ignore unknown fields (no harm)

5. **"What if some experiments failed?"**
   - Include only successful CRs
   - Minimum 2 CRs needed for plotting (e.g., 8 and 16)

---

## 📚 Example Complete Workflow

### **Your workflow (Person A):**

1. Implement `ecgdae/quantization.py`:
   - `uniform_quantize(latent, bits)`
   - `compute_compression_ratio(original_size, latent_shape, quantization_bits)`

2. Create `scripts/evaluate_compression.py`:
   - Load trained model from Week 1
   - For each CR setting:
     - Quantize latent representation
     - Reconstruct signals
     - Compute PRD, WWPRD, SNR
   - Save results to JSON

3. Run CR sweep:
   ```bash
   python -m scripts.evaluate_compression \
       --model_path outputs/week1_presentation/best_model.pth \
       --compression_ratios 4 8 16 32 \
       --output_file outputs/week2/cr_sweep_results.json
   ```

4. Verify JSON file was created

5. Notify Person B: "CR sweep complete! File ready at `outputs/week2/cr_sweep_results.json`"

---

### **Person B's workflow (after you finish):**

1. Load your JSON file
2. Generate all plots
3. Verify plots match your numbers
4. Create presentation slides
5. Both present to professor!

---

## 🎯 Timeline

**Suggested schedule:**

- **Day 1-2 (Person A):** Implement quantization and CR calculation
- **Day 3 (Person A):** Create evaluation script
- **Day 4 (Person A):** Run CR sweep and save JSON
- **Day 4 (Person B):** Generate plots from your JSON
- **Day 5 (Both):** Integration test and verify results
- **Day 6 (Both):** Create presentation slides
- **Day 7 (Both):** Practice presentation, meet with professor

---

## 💡 Tips

1. **Test early:** Create a dummy JSON file to give Person B for testing
2. **Communicate:** Let Person B know when you're stuck or delayed
3. **Version control:** Commit your code frequently
4. **Document:** Add comments explaining your CR calculation
5. **Sanity check:** PRD should decrease as CR decreases (more bits = better quality)

---

## 🐛 Common Mistakes to Avoid

### **❌ Wrong:**
```json
{
  4: {  // ← ERROR: Numeric key
    "PRD": "35.2",  // ← ERROR: String value
    "WWPRD": 30.1
    // ← ERROR: Missing other metrics
  }
}
```

### **✅ Correct:**
```json
{
  "4": {
    "PRD": 35.2,
    "PRD_std": 8.5,
    "WWPRD": 30.1,
    "WWPRD_std": 7.8,
    "SNR_in": 6.0,
    "SNR_out": 11.2,
    "SNR_improvement": 5.2,
    "latent_dim": 16,
    "quantization_bits": 8
  }
}
```

---

Good luck with your implementation! Person B is ready to visualize your results as soon as you're done! 🚀

**Contact:** Check `PERSON_B_GUIDE.md` for Person B's implementation details.

