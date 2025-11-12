# Team Collaboration Tasks - Loss Function Comparison Experiment

## Current Status
- **Training in Progress**: WWPRD-only model training (Step 1/4)
- **Estimated Completion**: 10-12 hours total
- **Next Steps**: Combined Loss training, evaluation, and report generation

## Task Division

### For Teammate (While Training Runs)

#### Task 1: Literature Review & Paper Analysis ⭐ **HIGH PRIORITY**
**Objective**: Understand the baseline methods and prepare for comparison

**Tasks**:
- [ ] Read Professor's paper: "Generalized_Rational_Variable_Projection_With_Application_in_ECG_Compression"
- [ ] Extract key metrics from **Table IV** (Records 117 & 119):
  - PRDN values for different methods
  - WWPRD values for different methods
  - Compression ratios used
- [ ] Document baseline methods mentioned in the paper
- [ ] Prepare a comparison table template for our results

**Deliverable**: `baseline_comparison_table.md` or Excel sheet with:
  - Method names
  - PRDN values (Record 117, Record 119)
  - WWPRD values (Record 117, Record 119)
  - Compression ratios

**Time Estimate**: 1-2 hours

---

#### Task 2: Code Review & Documentation ⭐ **MEDIUM PRIORITY**
**Objective**: Ensure code quality and prepare documentation

**Tasks**:
- [ ] Review `scripts/compare_loss_functions.py`:
  - Check if evaluation logic is correct
  - Verify metrics calculation matches paper formulas
  - Ensure proper handling of records 117 & 119
- [ ] Review `scripts/evaluate_records_117_119.py`:
  - Verify PRDN formula: `PRD = 100 * sqrt(sum((ref - recon)^2) / sum((ref - mean(ref))^2))`
  - Verify WWPRD calculation
  - Check if output format matches paper's Table IV
- [ ] Document any issues or improvements needed
- [ ] Create a code review checklist

**Deliverable**: `code_review_notes.md` with:
  - Issues found (if any)
  - Suggestions for improvement
  - Verification of formulas

**Time Estimate**: 1-2 hours

---

#### Task 3: Prepare Visualization Scripts ⭐ **MEDIUM PRIORITY**
**Objective**: Create scripts to visualize comparison results

**Tasks**:
- [ ] Create `scripts/plot_loss_comparison.py`:
  - Plot training curves (WWPRD vs Combined)
  - Plot validation loss comparison
  - Plot PRDN and WWPRD metrics side-by-side
- [ ] Create `scripts/plot_records_117_119.py`:
  - Plot reconstruction examples for both records
  - Overlay clean, noisy, and reconstructed signals
  - Highlight differences between WWPRD-only and Combined models
- [ ] Create comparison bar charts:
  - PRDN comparison (WWPRD-only vs Combined vs Paper baselines)
  - WWPRD comparison (WWPRD-only vs Combined vs Paper baselines)

**Deliverable**: Visualization scripts ready to run after training completes

**Time Estimate**: 2-3 hours

---

#### Task 4: Prepare Report Template ⭐ **HIGH PRIORITY**
**Objective**: Structure the experiment section for TDK report

**Tasks**:
- [ ] Create `experiment_section_template.md`:
  - Introduction to loss function comparison
  - Methodology section
  - Results section (with placeholders for numbers)
  - Discussion section
  - Tables and figures placeholders
- [ ] Prepare LaTeX/Word template if needed
- [ ] Structure the comparison with:
  - Training metrics comparison
  - Test set evaluation (records 117 & 119)
  - Comparison with paper baselines
  - Discussion of which loss works better

**Deliverable**: Report template ready to fill in results

**Time Estimate**: 1-2 hours

---

#### Task 5: Data Validation & Sanity Checks ⭐ **LOW PRIORITY**
**Objective**: Verify data integrity and prepare validation checks

**Tasks**:
- [ ] Verify MIT-BIH records 117 and 119 are correctly loaded
- [ ] Check if noise augmentation is consistent
- [ ] Create a script to validate:
  - Signal lengths match expected values
  - SNR values are correct
  - Window sizes are consistent
- [ ] Prepare sanity check script for final results

**Deliverable**: `scripts/validate_results.py` to check result consistency

**Time Estimate**: 1 hour

---

### For You (Main Tasks)

#### Task 1: Monitor Training Progress
- [ ] Check training status periodically
- [ ] Verify training completes successfully
- [ ] Handle any errors or interruptions

#### Task 2: Run Evaluation After Training
- [ ] Execute `scripts/compare_loss_functions.py` with `--skip_training`
- [ ] Verify evaluation results on records 117 & 119
- [ ] Generate comparison summary

#### Task 3: Final Analysis & Integration
- [ ] Integrate teammate's visualizations
- [ ] Fill in report template with actual results
- [ ] Create final comparison tables
- [ ] Prepare presentation materials if needed

---

## Priority Order (Recommended)

1. **Task 4** (Report Template) - Start immediately, can work in parallel
2. **Task 1** (Literature Review) - Critical for understanding baselines
3. **Task 2** (Code Review) - Important for correctness
4. **Task 3** (Visualization) - Can start after understanding requirements
5. **Task 5** (Validation) - Can do while waiting for results

---

## Communication Checklist

- [ ] Share GitHub branch: `improved-wwprd-prd`
- [ ] Share this document with teammate
- [ ] Set up communication channel (WeChat/Email/etc.)
- [ ] Agree on deadline for each task
- [ ] Schedule review meeting after training completes

---

## Expected Timeline

### Today (Training Day)
- **Morning/Afternoon**: Training runs (10-12 hours)
- **Teammate Tasks**: Literature review, code review, report template
- **Evening**: Training should complete

### Tomorrow (Results Day)
- **Morning**: Run evaluation and generate results
- **Afternoon**: Integrate all work, create visualizations
- **Evening**: Finalize report section

---

## Files to Share with Teammate

1. `scripts/compare_loss_functions.py` - Main comparison script
2. `scripts/evaluate_records_117_119.py` - Evaluation script
3. `ecgdae/metrics.py` - Metrics calculation (PRDN, WWPRD)
4. `ecgdae/losses.py` - Loss functions (WWPRD, Combined)
5. Professor's paper PDF
6. This collaboration document

---

## Questions to Clarify

- [ ] Which visualization style does the professor prefer?
- [ ] What format for the report? (LaTeX, Word, Markdown?)
- [ ] Any specific requirements for the comparison table?
- [ ] Should we include additional metrics beyond PRDN and WWPRD?

---

## Notes

- All code should be pushed to GitHub branch `improved-wwprd-prd`
- Use clear commit messages
- Document any assumptions or decisions
- Keep backup of important results

