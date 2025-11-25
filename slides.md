# Compression-Aware ECG Denoising Using WWPRD Optimization

**Authors:** Gracia Asmara Vianti El, Xu Jinxi
**Supervisor:** Dr. Péter Kovács
**Date:** 2025

---

## 1. Motivation & Problem Statement

**The Challenge:**
- **Telecardiology Constraints**: Wearable devices and hospital telemetry have strict limits on bandwidth and battery.
- **Noise**: ECG signals are often corrupted by muscle artifacts (EMG), baseline wander, and motion.
- **Diagnostic Safety**: Standard compression (MSE-based) treats all errors equally, often smoothing out critical sharp features like QRS complexes.

**Our Goal:**
- Develop a single Deep Learning model that **simultaneously denoises and compresses** ECG signals.
- **Clinical Alignment**: Optimize directly for diagnostic quality (WWPRD) rather than just mathematical error (MSE).
- **Target**: Achieve Quality Score (QS) > 0.5.

---

## 2. Methodology: End-to-End Framework

### A. Model Architecture
- **Deep 1D Convolutional Autoencoder**:
  - Encoder: Compresses signal to a low-dimensional latent space.
  - Bottleneck: Controls compression ratio (CR).
  - Decoder: Reconstructs clean signal from compressed representation.
- **Quantization-Aware Training (QAT)**:
  - Simulates 4-bit quantization errors during training.
  - Uses **Straight-Through Estimator (STE)** to allow gradient flow.
  - Critical for bridging the gap between "theoretical" and "deployment" performance.

### B. WWPRD Loss Function
- **Waveform-Weighted PRD (WWPRD)**:
  - Differentiable loss function that assigns higher weights to high-gradient regions (QRS complexes).
  - $w(t) = 1 + \alpha \cdot \frac{|x'(t)|}{\max|x'|}$
  - Ensures the model prioritizes clinically significant features.

---

## 3. Key Innovation: Quantization-Aware Training

**The "Quantization Gap" Problem:**
- Models trained normally (float32) fail when quantized to 4-bit integers for compression.
- **Before QAT**: Clean PRD 27% $\rightarrow$ Post-Quantization PRD 60%+ (Failed).

**Our Solution:**
- Train with simulated quantization noise.
- **Result**: Gap reduced from $2.2\times$ to $<1.3\times$.
- Enabled the use of extremely small latent dimensions (**Latent Dim = 2**) for high compression.

---

## 4. Experimental Results

### Achieved Target: QS > 0.5 ✅

| Metric | Baseline (Latent 4) | **Our Best (Latent 2 + QAT)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Compression Ratio** | 11.0:1 | **22.0:1** | **+100%** |
| **PRD (Post-Q)** | 42.7% | **36.20%** | **-15%** |
| **Quality Score (QS)** | 0.26 | **0.6078** | **+134%** |

- **Dataset**: MIT-BIH Arrhythmia Database (48 records).
- **Noise**: Real NSTDB noise (Muscle Artifact, 10dB SNR).

---

## 5. Visual Analysis

**Reconstruction Quality:**
- Despite high compression (22:1), QRS complexes are preserved.
- Noise (muscle artifacts) is effectively removed.
- P-waves and T-waves remain visible.

*(Insert Figure: Reconstruction Examples)*

**Rate-Distortion Curve:**
- Latent Dimension 2 offers the best trade-off.
- QAT pushes the Pareto frontier towards better quality at lower bitrates.

*(Insert Figure: QS Summary Table)*

---

## 6. Conclusion & Future Work

**Conclusion:**
1.  Successfully developed a compression-aware denoising autoencoder.
2.  **WWPRD Loss** effectively preserves diagnostic features.
3.  **QAT** is essential for practical deployment, enabling **QS = 0.6078**.
4.  Achieved high compression (22:1) with clinically acceptable quality.

**Future Work:**
- **VP Layer**: Evaluate Variable Projection layer for further structural improvements (In Progress).
- **Clinical Validation**: Blind test with cardiologists.
- **Real-time Deployment**: Optimize for embedded devices.

---

## Thank You!
