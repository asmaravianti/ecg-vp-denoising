# Parallel Training Guide for Teammate

## 🎯 Objective
Train the **Combined Loss model** in parallel while the main training runs the **WWPRD-only model**. This will cut total training time in half!

---

## 📋 Setup Instructions

### Step 1: Clone/Pull the Repository
```bash
# Make sure you're on the correct branch
git checkout improved-wwprd-prd
git pull origin improved-wwprd-prd
```

### Step 2: Verify Environment
```bash
# Check Python version (should be 3.8+)
python --version

# Check if required packages are installed
pip list | grep torch
pip list | grep numpy
pip list | grep rich
```

### Step 3: Verify Data Files
Make sure these directories exist:
- `data/mitbih/` - MIT-BIH dataset
- `data/nstdb/` - NSTDB noise database

---

## 🚀 Training Command

### For Teammate: Train Combined Loss Model

Run this command in your terminal:

```bash
python scripts/train_mitbih.py \
  --data_dir ./data/mitbih \
  --num_records 20 \
  --window_seconds 2.0 \
  --sample_rate 360 \
  --noise_type nstdb \
  --snr_db 10.0 \
  --nstdb_noise muscle_artifact \
  --model_type residual \
  --hidden_dims 32 64 128 \
  --latent_dim 32 \
  --loss_type combined \
  --combined_alpha 0.5 \
  --weight_alpha 2.0 \
  --batch_size 32 \
  --epochs 50 \
  --lr 0.0005 \
  --weight_decay 0.0001 \
  --val_split 0.15 \
  --output_dir outputs/loss_comparison_combined_alpha0.5 \
  --save_model \
  --device auto
```

### Windows PowerShell Version:
```powershell
python scripts/train_mitbih.py --data_dir ./data/mitbih --num_records 20 --window_seconds 2.0 --sample_rate 360 --noise_type nstdb --snr_db 10.0 --nstdb_noise muscle_artifact --model_type residual --hidden_dims 32 64 128 --latent_dim 32 --loss_type combined --combined_alpha 0.5 --weight_alpha 2.0 --batch_size 32 --epochs 50 --lr 0.0005 --weight_decay 0.0001 --val_split 0.15 --output_dir outputs/loss_comparison_combined_alpha0.5 --save_model --device auto
```

---

## ⏱️ Expected Timeline

- **Training Time**: ~5-6 hours (on CPU)
- **Output Directory**: `outputs/loss_comparison_combined_alpha0.5/`
- **Output Files**:
  - `best_model.pth` - Best model checkpoint
  - `config.json` - Training configuration
  - `training_history.json` - Training metrics over epochs
  - `final_metrics.json` - Final evaluation metrics
  - `training_curves.png` - Training visualization
  - `reconstruction_examples.png` - Sample reconstructions

---

## ✅ What to Expect

### During Training:
- Progress bar showing epoch progress
- Every 5 epochs, you'll see:
  - Train Loss
  - Val Loss
  - PRDN (%)
  - WWPRD (%)
  - SNR Improvement (dB)

### Example Output:
```
Epoch 5/50
  Train Loss: 25.1234
  Val Loss:   22.5678
  PRDN:       28.50% (15.20)
  WWPRD:      19.30% (8.50)
  SNR Improv: 6.20 dB
```

---

## 🔍 How to Monitor Progress

### Check Training Status:
```bash
# Check if Python process is running
# Windows:
Get-Process python

# Linux/Mac:
ps aux | grep python
```

### Check Training History (after some epochs):
```python
import json
with open('outputs/loss_comparison_combined_alpha0.5/training_history.json') as f:
    history = json.load(f)
    print(f"Completed epochs: {len(history['train_loss'])}/50")
    print(f"Latest val loss: {history['val_loss'][-1]:.4f}")
```

---

## ⚠️ Important Notes

1. **Don't Interrupt Training**: Let it run to completion (50 epochs)
2. **Save Output Directory**: The results will be in `outputs/loss_comparison_combined_alpha0.5/`
3. **Share Results**: After training completes, push the output directory to GitHub or share it
4. **Same Configuration**: Make sure you use the exact same command above to match the WWPRD training

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory" or "Device not found"
**Solution**: The command uses `--device auto` which will use CPU if GPU is not available. This is fine - CPU training just takes longer.

### Issue: "File not found: data/mitbih"
**Solution**: Make sure you're in the project root directory and the data files are downloaded.

### Issue: "Module not found"
**Solution**: Install dependencies:
```bash
pip install torch numpy matplotlib rich
```

### Issue: Training seems stuck
**Solution**:
- First epoch takes longer (loading data, model initialization)
- Progress is printed every 5 epochs
- Check CPU usage to confirm it's working

---

## 📤 After Training Completes

### Step 1: Verify Output Files
Check that these files exist:
- ✅ `outputs/loss_comparison_combined_alpha0.5/best_model.pth`
- ✅ `outputs/loss_comparison_combined_alpha0.5/training_history.json`
- ✅ `outputs/loss_comparison_combined_alpha0.5/final_metrics.json`

### Step 2: Share Results
Option A: Push to GitHub
```bash
git add outputs/loss_comparison_combined_alpha0.5/
git commit -m "Add Combined Loss model training results"
git push origin improved-wwprd-prd
```

Option B: Share the directory (if too large for GitHub, use cloud storage)

### Step 3: Notify Teammate
Send a message: "Combined Loss training complete! Results in outputs/loss_comparison_combined_alpha0.5/"

---

## 🎯 Next Steps (After Both Trainings Complete)

Once both models are trained:
1. Run evaluation on records 117 & 119
2. Generate comparison report
3. Create visualizations
4. Write experiment section

---

## 💡 Tips

- **Run in Background**: You can run training in background and do other tasks
- **Monitor Progress**: Check `training_history.json` periodically
- **Save Early**: The `best_model.pth` is saved automatically when validation improves
- **Don't Worry About Time**: CPU training is slow but will complete

---

## ❓ Questions?

If you encounter any issues:
1. Check the error message carefully
2. Verify all dependencies are installed
3. Make sure data files are in the correct location
4. Contact teammate for help

**Good luck with training! 🚀**

