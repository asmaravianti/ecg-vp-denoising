# Tasks for Teammate - Quick Reference

## 🎯 Main Objective
**NEW: Train Combined Loss model in parallel** to save time! Then prepare materials for the loss function comparison experiment results.

---

## 🚀 Task 0: Parallel Training (5-6 hours) ⭐ **START HERE**

**What to do**:
Train the **Combined Loss model** while the main training runs the WWPRD-only model. This cuts total time in half!

**Quick Start**:
- **Windows**: Run `start_combined_training.ps1` or see command below
- **Linux/Mac**: Run `bash start_combined_training.sh` or see command below

**Command to run**:
```bash
python scripts/train_mitbih.py --data_dir ./data/mitbih --num_records 20 --window_seconds 2.0 --sample_rate 360 --noise_type nstdb --snr_db 10.0 --nstdb_noise muscle_artifact --model_type residual --hidden_dims 32 64 128 --latent_dim 32 --loss_type combined --combined_alpha 0.5 --weight_alpha 2.0 --batch_size 32 --epochs 50 --lr 0.0005 --weight_decay 0.0001 --val_split 0.15 --output_dir outputs/loss_comparison_combined_alpha0.5 --save_model --device auto
```

**Output**: Results will be saved to `outputs/loss_comparison_combined_alpha0.5/`

**Time**: ~5-6 hours (can run in background)

**See**: `PARALLEL_TRAINING_GUIDE.md` for detailed instructions

**Why**: This saves 5-6 hours! Both models train simultaneously.

---

## ✅ Task 1: Extract Baseline Values from Paper (1-2 hours)

**What to do**:
1. Open the professor's paper: `Generalized_Rational_Variable_Projection_With_Application_in_ECG_Compression(1).pdf`
2. Find **Table IV** (should have results for Records 117 and 119)
3. Extract the following for each method:
   - Method name
   - PRDN (%) for Record 117
   - PRDN (%) for Record 119
   - WWPRD (%) for Record 117 (if available)
   - WWPRD (%) for Record 119 (if available)
   - Compression Ratio

**Output**: Create a file `baseline_comparison.csv` or `baseline_comparison.md` with a table like:

| Method | Record 117 PRDN | Record 119 PRDN | Record 117 WWPRD | Record 119 WWPRD | CR |
|--------|----------------|----------------|------------------|------------------|-----|
| Method 1 | ... | ... | ... | ... | ... |
| Method 2 | ... | ... | ... | ... | ... |

**Why**: We need to compare our results against these baselines.

---

## ✅ Task 2: Review Evaluation Code (1-2 hours)

**What to do**:
1. Open `scripts/evaluate_records_117_119.py`
2. Verify the PRDN formula matches the paper:
   ```
   PRDN = 100 * sqrt(sum((clean - recon)^2) / sum((clean - mean(clean))^2))
   ```
3. Check if WWPRD calculation is correct
4. Verify that records 117 and 119 are evaluated correctly
5. Check output format matches Table IV structure

**Output**: Create `code_review_notes.md` with:
- ✅ Confirmed correct / ❌ Issues found
- Any suggestions for improvement

**Why**: Ensure our evaluation matches the paper's methodology.

---

## ✅ Task 3: Create Report Template (1-2 hours)

**What to do**:
1. Create `experiment_section_template.md`
2. Structure it with:
   - **Introduction**: Why compare WWPRD-only vs Combined loss?
   - **Methodology**:
     - Model architecture
     - Training setup
     - Loss functions (WWPRD, Combined: alpha*PRDN + (1-alpha)*WWPRD)
   - **Results**:
     - Training metrics (placeholders: [PRDN_VALUE], [WWPRD_VALUE])
     - Test set evaluation (Records 117 & 119)
     - Comparison with baselines
   - **Discussion**: Which loss works better and why?

**Output**: `experiment_section_template.md` ready to fill in numbers

**Why**: Save time when results are ready - just plug in the numbers.

---

## ✅ Task 4: Prepare Visualization Scripts (2-3 hours)

**What to do**:
1. Create `scripts/plot_loss_comparison.py`:
   - Plot training curves for both models
   - Plot validation loss comparison
   - Plot PRDN and WWPRD metrics side-by-side
2. Create `scripts/plot_records_117_119.py`:
   - Plot reconstruction examples
   - Overlay clean, noisy, and reconstructed signals
   - Show differences between models

**Output**: Visualization scripts ready to run

**Why**: Visual results are easier to understand than numbers.

---

## 📋 Quick Checklist

- [ ] **Task 0: Start Combined Loss training** (Do this first!)
- [ ] Task 1: Extract baseline values from paper (while training runs)
- [ ] Task 2: Review evaluation code (while training runs)
- [ ] Task 3: Create report template (while training runs)
- [ ] Task 4: Prepare visualization scripts (while training runs)

---

## 📁 Files You'll Need

1. `scripts/evaluate_records_117_119.py` - Evaluation script
2. `scripts/compare_loss_functions.py` - Comparison script
3. `ecgdae/metrics.py` - Metrics formulas
4. Professor's paper PDF
5. `outputs/loss_comparison_summary.json` - (Will be generated after training)

---

## ⏰ Timeline

- **Now**: Start Task 0 (Combined Loss training) - runs in background
- **While training (5-6 hours)**: Do Tasks 1, 2, 3, 4 in parallel
  - Task 1 (Paper review): 1-2 hours
  - Task 3 (Report template): 1-2 hours
  - Task 2 (Code review): 1-2 hours
  - Task 4 (Visualizations): 2-3 hours
- **After training completes**: Help integrate results into report

---

## 💡 Tips

- Start with Task 1 and Task 3 (they're independent and quick)
- Task 2 requires understanding the code - ask if unclear
- Task 4 can be done incrementally - start with simple plots
- All work should be pushed to GitHub branch `improved-wwprd-prd`

---

## ❓ Questions?

If you encounter any issues:
1. Check the code comments
2. Review `TEAM_COLLABORATION_TASKS.md` for detailed instructions
3. Ask for clarification on unclear parts

---

**Good luck! 🚀**

