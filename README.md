# ❤️ Project Heartbeat Sensei  

<p align="center">
  <img src="https://github.com/sergie-o/cnn-vs-transfer-cifar10/blob/main/image.png" alt="Banner" width="900"/>
</p>  



**When accuracy isn’t enough: building AI that saves lives, one heartbeat at a time**  
*Detecting Arrhythmias with Deep Learning (PVC Detection using 1D CNNs)*  

---

## 📖 The Story  

Would you trust an AI model that claims **95% accuracy** in healthcare?  
I wouldn’t — and here’s why.  

In the **MIT-BIH Arrhythmia dataset**, only ~5% of beats are **Premature Ventricular Contractions (PVCs)**. A model could score 95% accuracy simply by predicting *“No PVC”* every time — while missing all the dangerous beats that matter most.  

This is the reality of working with **imbalanced biomedical data**:  
- ✅ The “easy” metric (accuracy) is misleading.  
- ✅ The **real challenge** is catching the rare, high-risk events.  
- ✅ In healthcare, *missing one dangerous heartbeat is far worse than raising a few false alarms*.  

So I built **Project Heartbeat Sensei**:  
A deep learning pipeline that learns to detect PVCs using **1D Convolutional Neural Networks**, patient-wise validation (to prevent leakage), and recall-focused design.  

👉 The goal? Create a system that prioritizes **sensitivity (recall)** — ensuring risky beats are detected, even if that means more alarms.  

---

## 🧭 Steps Taken in This Project  

1. **Data Acquisition**  
   - Downloaded ECG recordings from the **MIT-BIH Arrhythmia Database (PhysioNet)**.  
   - Explored file formats (`.dat`, `.hea`, `.atr`) and annotations.  

2. **Preprocessing & Beat Segmentation**  
   - Converted continuous ECG signals into **beat-centered windows** (216 samples × 2 leads).  
   - Extracted annotations (PVC vs Non-PVC) from cardiologist labels.  
   - Calculated **RR intervals** (previous/next beat distances) for context.  

3. **Data Cleaning & Splitting**  
   - Removed duplicates and ensured consistent beat labeling.  
   - Used **patient-wise splitting** (entity-based) to avoid leakage between train/val/test sets.  
   - Balanced splits so validation and test had at least some PVCs.  

4. **Feature Engineering**  
   - Created features like `is_pvc`, `is_normal`, `target_aami`, `rr_prev`, `rr_next`.  
   - Stored beat snippets as arrays in **Parquet files** for efficient training.  

5. **Handling Class Imbalance**  
   - Observed PVCs were ~5% of the dataset.  
   - Computed **class weights** to counter imbalance.  
   - Tuned **thresholds** using Precision–Recall analysis to favor **recall**.  

6. **Modeling**  
   - Designed a **1D Convolutional Neural Network (CNN)** with:  
     - Conv1D → BatchNorm → MaxPooling blocks.  
     - Dropout + L2 regularization for stability.  
     - GlobalAveragePooling + Dense softmax output.  
   - Implemented both **single-lead** and **two-lead** input versions.  

7. **Training**  
   - Used **Adam optimizer (1e-4)** and **binary cross-entropy** (PVC task).  
   - Early stopping on **validation PR-AUC** to avoid overfitting.  
   - Trained on GPU (Apple M3 Metal backend).  

8. **Evaluation**  
   - Accuracy alone ≈ 88% → misleading due to imbalance.  
   - At recall-friendly threshold:  
     - **Recall ≈ 0.56** (more than half PVCs detected).  
     - **Precision ≈ 0.22** (many false alarms, but better than random).  
   - Generated:  
     - Precision–Recall Curve (with baseline).  
     - Precision/Recall/F1 vs Threshold plot.  
     - Confusion Matrix at recall-first operating point.  

9. **Interpretation**  
   - Learned that **metrics must align with context**: recall is more important in healthcare than raw accuracy.  
   - Understood how **imbalanced learning** and **thresholding** impact model behavior.  

10. **Next Steps**  
   - Extend to **LSTMs (Update 2.0)** for temporal context.  
   - Expand to **Multiclass CNNs (Update 3.0)** for AAMI 5-class classification.  

---

## ⚙️ Tech Stack  
- **Python**  
- **TensorFlow/Keras** (1D CNN modeling)  
- **Scikit-learn** (class weights, metrics)  
- **Pandas & NumPy** (data preprocessing)  
- **Matplotlib / Plotly** (visualizations)  

---

## 📊 Results (PVC Classification)  
- **Accuracy ≈ 88%** (but misleading in imbalance).  
- At a recall-friendly threshold:  
  - **Recall ≈ 0.56** → caught over half of PVCs.  
  - **Precision ≈ 0.22** → false alarms, but far better than baseline prevalence (≈0.05).  

👉 In healthcare AI, **high recall > high accuracy**. Better to raise false alarms than miss life-threatening beats.  

---

## 📈 Visualizations  
- Precision–Recall Curve (with baseline prevalence line).  
- Precision / Recall / F1 vs Threshold plot.  
- Confusion Matrix at recall-first threshold.  

*(Add images here once you upload them into the repo — e.g. `![PR Curve](assets/pr_curve.png)`)*  

---

## 🚀 Reproduction Guide  

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/Project-Heartbeat-Sensei.git
   cd Project-Heartbeat-Sensei
