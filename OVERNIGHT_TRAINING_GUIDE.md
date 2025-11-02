# Overnight Training Guide - Windows Laptop

## ✅ **Yes, You Can Train Overnight!**

Training on CPU overnight is **absolutely safe and recommended**. Here's how to set it up properly:

---

## 🔧 **Step-by-Step Setup**

### **1. Prevent Laptop from Sleeping/Locking**

#### **Method A: Windows Power Settings (Recommended)**

1. **Open Power Settings:**
   - Press `Win + X` → Select "Power Options"
   - Or: Settings → System → Power & sleep

2. **Configure When Plugged In:**
   - **"When plugged in, PC goes to sleep after:"** → Set to **"Never"**
   - **"When plugged in, turn off screen after:"** → Set to **"30 minutes"** (screen can turn off, but PC stays on)
   - **"When plugged in, turn off screen after:"** → Set to **"Never"** (if you want to monitor)

3. **Additional Settings (Click "Additional power settings"):**
   - Select "High performance" or "Balanced" plan
   - Click "Change plan settings"
   - Set **"Put computer to sleep"** → **"Never"**
   - Set **"Turn off display"** → 30 minutes (optional, saves battery)

4. **Advanced Settings:**
   - Click "Change advanced power settings"
   - Expand "Sleep" → "Allow hybrid sleep" → Set to **"Off"**
   - Expand "USB settings" → "USB selective suspend" → Set to **"Disabled"**

#### **Method B: Keep Screen On (Optional)**

```powershell
# Prevent screen from turning off (run in PowerShell)
powercfg /change monitor-timeout-ac 0
powercfg /change monitor-timeout-dc 0
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

To restore later:
```powershell
powercfg /change monitor-timeout-ac 10
powercfg /change standby-timeout-ac 30
```

---

### **2. Prevent Auto-Lock**

1. **Disable Screen Lock:**
   - Settings → Accounts → Sign-in options
   - Under "Require sign-in": Set to **"Never"**

2. **Or Use Simple Workaround:**
   - Press `Win + L` to lock manually if needed
   - Or disable lock: Settings → Personalization → Lock screen → Screen timeout → Never

---

### **3. Keep Laptop Plugged In**

**IMPORTANT:**
- ✅ **Always keep laptop plugged in** during training
- ✅ Prevents battery drain
- ✅ Prevents automatic sleep on low battery
- ✅ Ensures consistent performance

---

### **4. Prevent Windows Updates**

**Temporarily disable auto-updates:**
1. Settings → Update & Security → Windows Update
2. Click "Advanced options"
3. Pause updates for 7 days (renewable)

**Or schedule updates:**
- Set active hours when you're not training
- Example: Active hours 9 AM - 6 PM, updates happen at night (but training also at night, so pause updates)

---

### **5. Close Unnecessary Applications**

Before starting training:
- ✅ Close browser tabs (save memory)
- ✅ Close other applications
- ✅ Disable antivirus real-time scan (temporarily, if it's resource-intensive)
- ✅ Close backup software (temporarily)

**Keep open:**
- Terminal/PowerShell (to monitor)
- Task Manager (to check CPU/memory usage)

---

## 🚀 **Starting Training**

### **Recommended Command:**

```powershell
# Start training and save output to log file
python scripts/train_mitbih.py `
    --num_records 20 `
    --epochs 150 `
    --batch_size 32 `
    --lr 0.0005 `
    --weight_decay 0.0001 `
    --latent_dim 32 `
    --loss_type wwprd `
    --save_model `
    --output_dir ./outputs/week2_improved 2>&1 | Tee-Object -FilePath training_log.txt
```

**What this does:**
- Saves all output to `training_log.txt`
- Still shows output in terminal
- Can check progress later even if terminal closed

### **Or Use Background Job (Alternative):**

```powershell
# Start job in background
Start-Job -ScriptBlock {
    cd E:\ModelingLab\ecg_TDK
    python scripts/train_mitbih.py `
        --num_records 20 `
        --epochs 150 `
        --output_dir ./outputs/week2_improved `
        > training_output.txt 2>&1
}

# Check status later
Get-Job
Receive-Job -Id 1
```

---

## 📊 **Monitoring Progress**

### **Check Training Status Without Opening Terminal:**

1. **View Log File:**
   ```powershell
   # See last 20 lines of training log
   Get-Content training_log.txt -Tail 20

   # Or use notepad
   notepad training_log.txt
   ```

2. **Check Output Directory:**
   - `outputs/week2_improved/training_curves.png` - Updated after each epoch
   - `outputs/week2_improved/training_history.json` - Contains loss values

3. **Monitor System Resources:**
   - Task Manager → Performance tab
   - Check CPU usage (should be high)
   - Check Memory usage (should be stable)

---

## ⚠️ **Safety Checks**

### **Before Going to Sleep:**

✅ **Checklist:**
- [ ] Laptop is plugged in
- [ ] Power settings prevent sleep
- [ ] Screen lock disabled (or won't lock)
- [ ] Training command started successfully
- [ ] Output directory is being created
- [ ] Training log file is being written
- [ ] CPU usage is high in Task Manager (indicates training is running)

### **Quick Test:**
```powershell
# Check if Python process is running
Get-Process python

# Should show Python using CPU
```

---

## 🔋 **Battery & Performance Tips**

### **Optimize Performance:**

1. **High Performance Mode:**
   - Settings → System → Power & sleep → Additional power settings
   - Select "High performance" plan

2. **Cooling:**
   - ✅ Keep laptop on hard surface (not bed/sofa)
   - ✅ Ensure good ventilation
   - ✅ Consider laptop cooling pad (optional)

3. **Background Apps:**
   - Disable startup programs (Task Manager → Startup)
   - Close unnecessary background services

---

## 📝 **Morning Check List**

**When you wake up:**

1. **Check Training Status:**
   ```powershell
   # Check if still running
   Get-Process python

   # View latest log
   Get-Content training_log.txt -Tail 30
   ```

2. **Check Results:**
   - Look at `outputs/week2_improved/training_curves.png`
   - Check if loss decreased
   - Verify `best_model.pth` exists

3. **If Training Finished:**
   - Check `final_metrics.json` for results
   - Re-evaluate model:
     ```powershell
     python scripts/evaluate_compression.py `
         --model_path outputs/week2_improved/best_model.pth `
         --config_path outputs/week2_improved/config.json `
         --compression_ratios 4 8 16 32 `
         --output_file outputs/week2/improved_results.json
     ```

---

## 🛡️ **Troubleshooting**

### **Problem: Training Stopped Unexpectedly**

**Possible Causes:**
- Out of memory (reduce batch_size to 16)
- Python error (check log file)
- Windows update (pause updates)

**Solution:**
```powershell
# Check log for errors
Get-Content training_log.txt | Select-String "error" -Context 5

# Check if process crashed
Get-EventLog -LogName Application -Source Python -Newest 10
```

### **Problem: Laptop Went to Sleep**

**Check:**
- Power settings again
- Check if plugged in
- Check battery level

**Solution:**
- Resume training (model should auto-save checkpoints)
- Or restart with fewer epochs if close to finishing

### **Problem: Slow Training**

**Check:**
- Task Manager → CPU usage (should be 70-100%)
- Close other applications
- Check if thermal throttling (CPU temp too high)

---

## 💡 **Best Practices**

### **For Multiple Nights of Training:**

1. **Day 1:** Train with improved config (150 epochs)
   - Check next morning
   - If not converged, continue to 200 epochs

2. **Day 2:** Try ResidualAutoEncoder
   - Different architecture experiment

3. **Day 3-6:** Train multiple models (different latent_dim)
   - One model per night
   - Compare results

### **Time Management:**

- **Start training:** 9-10 PM (before sleep)
- **Expected finish:** 3-9 AM next day (6-12 hours)
- **Check results:** Morning after

### **Resource Management:**

- Train **one model at a time** (don't run multiple)
- Each night = one experiment
- Plan ahead which model to train

---

## ✅ **Quick Setup Script**

Create a file `start_overnight_training.ps1`:

```powershell
# Prevent sleep
powercfg /change monitor-timeout-ac 30
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

# Start training
Write-Host "Starting training... This will run overnight."
Write-Host "Check training_log.txt for progress."

python scripts/train_mitbih.py `
    --num_records 20 `
    --epochs 150 `
    --batch_size 32 `
    --lr 0.0005 `
    --weight_decay 0.0001 `
    --latent_dim 32 `
    --loss_type wwprd `
    --save_model `
    --output_dir ./outputs/week2_improved `
    2>&1 | Tee-Object -FilePath training_log.txt

Write-Host "Training completed! Check outputs/week2_improved/"
```

**Run it:**
```powershell
.\start_overnight_training.ps1
```

---

## 📋 **Summary**

### **Can you train overnight?**
✅ **YES!** It's safe and recommended.

### **What to do:**
1. ✅ Set power settings to never sleep (when plugged in)
2. ✅ Disable screen lock
3. ✅ Keep laptop plugged in
4. ✅ Start training with log file
5. ✅ Go to sleep!

### **What to check in morning:**
1. ✅ Training finished (check log file)
2. ✅ Results in `outputs/week2_improved/`
3. ✅ Training curves show improvement

### **Time estimate:**
- **150 epochs:** 6-12 hours (perfect for overnight)
- **200 epochs:** 8-16 hours (start Friday night, check Sunday morning)

---

**Bottom Line:** Overnight training on CPU is totally feasible! Just configure power settings properly, start training before sleep, and check results in the morning. Your laptop will handle it fine as long as it's plugged in and configured correctly.

