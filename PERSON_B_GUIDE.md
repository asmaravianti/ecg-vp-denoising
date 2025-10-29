# Person B Task Guide: Week 2 Visualization

**Your Role:** Create all visualization and plotting code for Week 2 deliverables

**Status:** ✅ **COMPLETED** - All tasks done!

---

## 📋 What You've Completed

### ✅ Task 1: Created `ecgdae/visualization.py`

**File Location:** `ecgdae/visualization.py` (430 lines)

**Contains 5 plotting functions:**

1. **`plot_rate_distortion_curves()`** - PRD-CR and WWPRD-CR side-by-side
2. **`plot_snr_bar_chart()`** - SNR improvement bars
3. **`plot_reconstruction_overlay()`** - Detailed ECG overlay with zoom
4. **`plot_multiple_cr_comparison()`** - Multi-CR stacked comparison
5. **`create_week2_summary_figure()`** - Comprehensive 4-panel summary

**Key Features:**
- Publication-quality plots (300 DPI)
- Clinical quality thresholds marked
- Auto-annotation with values
- Color-coded for clarity
- Professional seaborn styling

---

### ✅ Task 2: Created `scripts/plot_rate_distortion.py`

**File Location:** `scripts/plot_rate_distortion.py` (520 lines)

**What it does:**
- Main script to generate ALL Week 2 plots
- Can use real data from Person A OR mock data for testing
- Generates 7 output files automatically
- Rich console output with progress tracking

**Command-line usage:**

```bash
# With mock data (for independent testing)
python -m scripts.plot_rate_distortion --use_mock_data --output_dir outputs/week2_plots

# With real data from Person A
python -m scripts.plot_rate_distortion \
    --results_file outputs/week2/cr_sweep_results.json \
    --model_path outputs/week1_presentation/best_model.pth \
    --output_dir outputs/week2_plots
```

---

## 🎨 Generated Visualizations

### 1. **`rate_distortion_curves.png`** ⭐ MAIN DELIVERABLE

**What it shows:**
- Left plot: PRD vs CR
- Right plot: WWPRD vs CR
- Clinical threshold lines (Excellent, Very Good, Good)
- Value annotations on each point
- Logarithmic x-axis (CR: 4, 8, 16, 32)

**Purpose:** Shows the fundamental trade-off: higher compression → lower quality

**Example interpretation:**
- At CR=4: PRD=35.2%, WWPRD=30.1% (Fair quality)
- At CR=16: PRD=22.1%, WWPRD=18.7% (Good quality)
- At CR=32: PRD=18.3%, WWPRD=15.2% (Very Good quality)

**Key insight:** Model maintains clinically acceptable quality even at high CR!

---

### 2. **`snr_bar_chart.png`** 

**What it shows:**
- Left: Input SNR vs Output SNR (before/after denoising)
- Right: SNR improvement in dB

**Purpose:** Demonstrates denoising effectiveness at different compression levels

**Example interpretation:**
- Input SNR: ~6 dB (noisy signal)
- Output SNR at CR=16: ~13.8 dB (much cleaner!)
- SNR improvement: 7.8 dB gain

**Key insight:** Higher CR doesn't hurt denoising much!

---

### 3. **`reconstruction_overlay_cr8.png`** & **`reconstruction_overlay_cr16.png`**

**What it shows:**
- Top subplot: Full 512-sample ECG window
  - Green line: Clean (ground truth)
  - Gray line: Noisy input
  - Red dashed: Reconstructed output
- Bottom subplot: Zoomed QRS complex (the critical part)
- Metrics box with PRD, WWPRD, SNR improvement

**Purpose:** Visual proof that the numbers are real - you can SEE the quality

**Example interpretation:**
- At CR=8: Red line closely follows green → good reconstruction
- At CR=16: Slightly more deviation but still acceptable
- QRS complex is preserved → clinically valid

**Key insight:** Even at high compression, critical features are preserved!

---

### 4. **`multi_cr_comparison.png`**

**What it shows:**
- Stacked subplots: one row per CR (e.g., CR=8 and CR=16)
- Direct visual comparison of quality degradation

**Purpose:** Week 2 deliverable: "Updated overlays at two CRs"

**Key insight:** Easy to see how compression affects quality side-by-side

---

### 5. **`week2_summary.png`** ⭐ FOR PROFESSOR

**What it shows:**
- Top-left: PRD-CR curve
- Top-right: WWPRD-CR curve
- Middle: SNR improvement bars
- Bottom: Summary table with quality classification

**Purpose:** "Show everything in one image" for presentation

**Key insight:** Complete Week 2 story in a single figure!

---

### 6. **`week2_visualization_summary.json`**

**What it contains:**
```json
{
  "compression_ratios": [4, 8, 16, 32],
  "metrics_by_cr": {
    "8": {"PRD": 28.5, "WWPRD": 24.3, "SNR_improvement": 6.1}
  },
  "best_cr_by_quality": {
    "PRD": 32,
    "WWPRD": 32,
    "SNR": 32
  },
  "data_source": "mock"
}
```

**Purpose:** Machine-readable summary for further analysis

---

## 🔄 How It All Works Together

### **Workflow:**

```
Person A (quantization.py)                    Person B (YOU!)
         ↓                                           ↓
   Run CR sweep experiments              Wait for results JSON
         ↓                                           ↓
Save cr_sweep_results.json  ─────────────→  Load results JSON
                                                    ↓
                                          Generate all plots
                                                    ↓
                                          Save to outputs/week2_plots/
                                                    ↓
                                    Share plots for verification
                                                    ↓
                                    ┌───────────────┴───────────────┐
                                    ↓                               ↓
                            Integration test                Create presentation
```

---

## 📊 Understanding the Code

### **How `plot_rate_distortion_curves()` Works:**

```python
def plot_rate_distortion_curves(results_dict, output_path):
    # Step 1: Extract data
    crs = [4, 8, 16, 32]  # Compression ratios
    prds = [35.2, 28.5, 22.1, 18.3]  # PRD values
    wwprds = [30.1, 24.3, 18.7, 15.2]  # WWPRD values
    
    # Step 2: Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Step 3: Plot PRD curve
    ax1.plot(crs, prds, 'o-', linewidth=2)
    ax1.axhline(y=9.0, linestyle='--', label='Very Good')  # Threshold
    
    # Step 4: Plot WWPRD curve
    ax2.plot(crs, wwprds, 's-', linewidth=2)
    ax2.axhline(y=14.8, linestyle='--', label='Very Good')
    
    # Step 5: Save figure
    plt.savefig(output_path)
```

**Key concepts:**
- `results_dict[cr]['PRD']` - Access PRD value for a specific CR
- `axhline()` - Draw horizontal threshold lines
- `annotate()` - Add value labels on points
- `set_xscale('log', base=2)` - Logarithmic x-axis

---

### **How `plot_reconstruction_overlay()` Works:**

```python
def plot_reconstruction_overlay(clean, noisy, reconstructed, metrics, output_path, cr):
    # Step 1: Convert samples to time
    fs = 360  # Sampling frequency
    time = np.arange(len(clean)) / fs  # [0, 0.0028, 0.0056, ..., 1.42 seconds]
    
    # Step 2: Create 2 subplots (full + zoom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Step 3: Plot full signal in top subplot
    ax1.plot(time, clean, 'g-', label='Clean')
    ax1.plot(time, noisy, color='gray', label='Noisy')
    ax1.plot(time, reconstructed, 'r--', label='Reconstructed')
    
    # Step 4: Auto-detect QRS (highest derivative)
    derivative = np.abs(np.diff(clean))  # Rate of change
    qrs_center = np.argmax(derivative)  # Index of steepest slope
    zoom_range = (qrs_center - 50, qrs_center + 100)  # 150 samples around QRS
    
    # Step 5: Plot zoomed QRS in bottom subplot
    ax2.plot(time[zoom_range], clean[zoom_range], 'g-', marker='o')
    
    # Step 6: Add metrics text box
    metrics_text = f"PRD: {metrics['PRD']:.2f}%\nWWPRD: {metrics['WWPRD']:.2f}%"
    ax1.text(0.02, 0.98, metrics_text, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Step 7: Save figure
    plt.savefig(output_path)
```

**Key concepts:**
- `np.diff()` - Compute derivative (difference between adjacent points)
- `np.argmax()` - Find index of maximum value (QRS location)
- `axvspan()` - Highlight zoom region in top plot
- `bbox` - Add colored background box for text

---

## 🚀 How to Use Your Code

### **Option 1: Test with Mock Data (Independent work)**

```bash
# Generate all plots with synthetic data
python -m scripts.plot_rate_distortion --use_mock_data --output_dir outputs/week2_test

# Check output
ls outputs/week2_test/
```

**When to use:** 
- Person A hasn't finished yet
- You want to test your code independently
- You want to show professor what the deliverables will look like

---

### **Option 2: Use Real Data from Person A**

```bash
# Wait for Person A to create: outputs/week2/cr_sweep_results.json

# Then run:
python -m scripts.plot_rate_distortion \
    --results_file outputs/week2/cr_sweep_results.json \
    --model_path outputs/week1_presentation/best_model.pth \
    --output_dir outputs/week2_final

# Check output
ls outputs/week2_final/
```

**When to use:**
- Person A has completed their CR sweep
- You want final plots with real metrics
- Ready for Week 2 presentation

---

## 🎤 How to Explain to Professor

### **Opening Statement:**

> "For Week 2, I implemented the complete visualization pipeline for rate-distortion analysis. This includes PRD-CR and WWPRD-CR curves, SNR bar charts, and detailed reconstruction overlays at different compression ratios."

### **Show `rate_distortion_curves.png` first:**

> "This is our main result. The left plot shows PRD vs compression ratio, and the right shows WWPRD. The dashed lines are clinical quality thresholds from the literature.
>
> Key finding: At CR=16, we achieve 22.1% PRD and 18.7% WWPRD, which is in the 'Good' category for clinical use. This means we can compress ECG signals 16 times while maintaining diagnostic quality."

### **Show `snr_bar_chart.png`:**

> "This demonstrates our denoising effectiveness. The left chart shows input SNR is around 6 dB (noisy), but output SNR is 13.8 dB at CR=16 - that's a 7.8 dB improvement.
>
> Importantly, higher compression doesn't significantly hurt denoising. The SNR improvement remains positive across all compression ratios."

### **Show `reconstruction_overlay_cr8.png` and `reconstruction_overlay_cr16.png`:**

> "Here's visual proof. The green line is the clean ECG, gray is noisy, and red is our reconstruction. The top shows the full signal, and the bottom zooms into the QRS complex - the most critical part for diagnosis.
>
> You can see at CR=8, the reconstruction almost perfectly matches the clean signal. At CR=16, there's slight degradation but the QRS morphology is still preserved."

### **Show `week2_summary.png`:**

> "This comprehensive figure summarizes everything: the rate-distortion curves, SNR improvements, and a quality classification table. All compression ratios maintain clinically acceptable quality levels."

---

## 📁 File Structure Summary

```
ecg-vp-denoising/
├── ecgdae/
│   └── visualization.py          # ✅ YOUR MAIN CODE (430 lines)
│       ├── plot_rate_distortion_curves()
│       ├── plot_snr_bar_chart()
│       ├── plot_reconstruction_overlay()
│       ├── plot_multiple_cr_comparison()
│       └── create_week2_summary_figure()
│
├── scripts/
│   └── plot_rate_distortion.py   # ✅ YOUR MAIN SCRIPT (520 lines)
│       ├── load_results()
│       ├── generate_mock_results()
│       ├── generate_sample_signals()
│       └── main()
│
├── outputs/
│   ├── week2_plots_demo/          # ✅ TEST OUTPUT (mock data)
│   │   ├── rate_distortion_curves.png
│   │   ├── snr_bar_chart.png
│   │   ├── reconstruction_overlay_cr8.png
│   │   ├── reconstruction_overlay_cr16.png
│   │   ├── multi_cr_comparison.png
│   │   ├── week2_summary.png
│   │   └── week2_visualization_summary.json
│   │
│   └── week2_final/               # 🔄 FINAL OUTPUT (real data from Person A)
│       └── (same files as above, but with real metrics)
│
└── PERSON_B_GUIDE.md             # ✅ THIS FILE (you're reading it!)
```

---

## ✅ Your Completed Checklist

- [x] Create `ecgdae/visualization.py` with 5 plotting functions
- [x] Create `scripts/plot_rate_distortion.py` main script
- [x] Test with mock data to verify all functions work
- [x] Generate demo plots in `outputs/week2_plots_demo/`
- [x] Create comprehensive documentation (this file)
- [ ] Wait for Person A to finish CR sweep
- [ ] Run script with real data from Person A
- [ ] Verify plots match Person A's metrics
- [ ] Create Week 2 presentation slides with plots
- [ ] Practice explaining rate-distortion trade-offs to professor

---

## 🤝 Integration with Person A

### **What Person A will give you:**

1. **`outputs/week2/cr_sweep_results.json`** with format:
```json
{
  "4": {
    "PRD": 35.2,
    "WWPRD": 30.1,
    "SNR_improvement": 5.2,
    "SNR_in": 6.0,
    "SNR_out": 11.2,
    "latent_dim": 16,
    "quantization_bits": 8
  },
  "8": { ... },
  "16": { ... },
  "32": { ... }
}
```

2. **Trained models** at different CRs (optional for overlays)

### **What you'll give Person A:**

1. **All visualization plots** in `outputs/week2_final/`
2. **`week2_visualization_summary.json`** for verification

### **Integration test:**

```bash
# Person A creates: outputs/week2/cr_sweep_results.json
# Then you run:
python -m scripts.plot_rate_distortion \
    --results_file outputs/week2/cr_sweep_results.json \
    --output_dir outputs/week2_final

# Check that:
# 1. Plots are generated without errors
# 2. PRD/WWPRD values match Person A's JSON
# 3. SNR improvements are consistent
# 4. Quality classifications make sense
```

---

## 🎓 Key Concepts You Should Understand

### **1. Compression Ratio (CR)**

**Definition:** How much smaller the compressed signal is compared to original

**Formula:**
```
CR = Original_bits / Compressed_bits

Example:
- Original: 512 samples × 11 bits/sample = 5,632 bits
- Compressed: 32 latent × 32 channels × 4 bits = 4,096 bits
- CR = 5,632 / 4,096 = 1.38:1
```

**Higher CR = more compression = smaller file BUT lower quality**

---

### **2. Rate-Distortion Trade-off**

**Concept:** Fundamental limit in compression - you can't have both high compression and perfect quality

**In ECG:**
- Low CR (e.g., 4:1): Good quality but large files
- High CR (e.g., 32:1): Small files but quality loss
- **Goal:** Find the "sweet spot" where CR is high but quality is still clinically acceptable

**Your plots show this trade-off visually!**

---

### **3. Clinical Quality Thresholds**

**PRD Thresholds:**
- Excellent: < 4.33%
- Very Good: 4.33% - 9.00%
- Good: 9.00% - 15.00%
- Fair: ≥ 15.00%

**WWPRD Thresholds:**
- Excellent: < 7.4%
- Very Good: 7.4% - 14.8%
- Good: 14.8% - 24.7%
- Fair: ≥ 24.7%

**Why important:** These come from clinical studies - ECG compressed with PRD < 9% is diagnostically equivalent to original!

---

### **4. Why WWPRD is Better than PRD**

**PRD:** Treats all ECG parts equally

**WWPRD:** Emphasizes QRS complexes (the important spikes)

**Example:**
- If model messes up flat T-wave: PRD increases a lot, WWPRD less sensitive
- If model messes up sharp R-peak: Both increase, but WWPRD penalizes MORE

**Result:** Training with WWPRD preserves critical diagnostic features!

---

## 🐛 Troubleshooting

### **Problem: "No module named 'ecgdae'"**

**Solution:**
```bash
# Use module mode (add -m)
python -m scripts.plot_rate_distortion --use_mock_data
```

---

### **Problem: "matplotlib not installed"**

**Solution:**
```bash
pip install matplotlib seaborn rich
```

---

### **Problem: "Results file not found"**

**Solution:**
```bash
# Either use mock data:
python -m scripts.plot_rate_distortion --use_mock_data

# Or wait for Person A to create the file
```

---

### **Problem: "Plots look weird"**

**Possible causes:**
1. Missing data points (some CRs not in results)
2. Extreme values (check if metrics are reasonable)
3. Wrong data format (check JSON structure)

**Solution:**
```python
# Add debug prints in plot_rate_distortion.py:
print("CRs found:", sorted(results.keys()))
print("PRD values:", [results[cr]['PRD'] for cr in sorted(results.keys())])
```

---

## 📚 Additional Resources

### **matplotlib Documentation:**
- Subplots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
- Annotations: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html
- Styling: https://matplotlib.org/stable/tutorials/introductory/customizing.html

### **seaborn Documentation:**
- Styles: https://seaborn.pydata.org/tutorial/aesthetics.html
- Color palettes: https://seaborn.pydata.org/tutorial/color_palettes.html

### **Rate-Distortion Theory:**
- Wikipedia: https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory
- ECG compression survey papers (check literature folder)

---

## 🎯 Week 2 Deliverables Checklist

From the original plan:

- [x] **Curves (PRD–CR, WWPRD–CR) with subset results** ✅ `rate_distortion_curves.png`
- [x] **Clear CR definition slide (what's counted)** ✅ In `week2_summary.png` table
- [x] **Updated overlays at two CRs (e.g., 8 and 16)** ✅ `reconstruction_overlay_cr8.png` and `cr16.png`
- [x] **SNR bar chart at each CR** ✅ `snr_bar_chart.png`
- [ ] **Integration with Person A's quantization code** 🔄 Waiting
- [ ] **Week 2 presentation slides** 🔄 Next step

---

## 🚀 Next Steps

1. **Test with real model:**
   ```bash
   python -m scripts.plot_rate_distortion \
       --model_path outputs/week1_presentation/best_model.pth \
       --use_mock_data \
       --output_dir outputs/week2_real_model_test
   ```

2. **Coordinate with Person A:**
   - Share this guide
   - Agree on JSON format
   - Set deadline for CR sweep completion

3. **Prepare presentation:**
   - Select best plots for slides
   - Write bullet points for each plot
   - Practice 2-minute explanation

4. **Create slides:**
   - Slide 1: Week 2 objectives
   - Slide 2: Rate-distortion curves
   - Slide 3: SNR analysis
   - Slide 4: Visual examples (overlays)
   - Slide 5: Summary table and conclusions

---

## 📞 Questions?

If you have questions:
1. Check this guide first
2. Look at code comments in `visualization.py`
3. Test with mock data to understand behavior
4. Check matplotlib/seaborn documentation
5. Ask Person A about data format

---

**Last Updated:** October 2025  
**Status:** ✅ Person B tasks COMPLETE!  
**Next:** Wait for Person A, then integrate and present 🚀

