### **Updated Pipeline Steps**

1. **Raw Tensors**
   - Load dataset **without** transformations.
   - Convert images to tensors and **normalize per-channel**.
   - Save tensors as `.pt` for reuse.

2. **Compute MTCNN Bounding Boxes and Filter**
   - Apply **MTCNN** on raw tensors to detect face bounding boxes.
   - Save bounding boxes separately.

3. **Apply CLAHE + UNSHARP**
   - Load raw tensors.
   - Apply **CLAHE + UNSHARP** for contrast enhancement.
   - Save enhanced tensors.

4. **Extract Landmarks using Dlib 68**
   - Use **Dlib’s 68 landmarks detector** on **enhanced tensors**.
   - Use MTCNN’s bounding boxes to improve accuracy.
   - Save landmarks.

5. **STEP: Save Original Results**
   - Store bounding boxes, landmarks, image sizes, and file paths in a structured **DataFrame (CSV)**.

6**STEP: Crop Face with Padding**
   - Crop face region using MTCNN box.
   - Apply **padding** (1-3 pixels) to preserve facial details.


7. **Resize Images & Adjust Landmarks**
   - Resize images (e.g., to **512×512**).
   - Adjust **bounding boxes** and **landmarks** proportionally.
   - Save resized images and adjusted coordinates.


8. **STEP: Save Adjusted Results**
   - Save **cropped face images**, resized images, and adjusted landmarks in a new dataset.

9. **STEP: Create "2-Channel" Images**
   - Integrate **landmark positions** as a **4th channel** into cropped face images.
   - Normalize landmark positions to match image dimensions.

10. **STEP: Prepare Data for Model**
   - Organize 4-channel images as input data for model training.

11. **STEP: Train the Model**
   - Train the model using 2-channel images for facial analysis/expression recognition.

---

### **Considerations**
- **Efficiency:**
  - Avoid reloading datasets by saving intermediate results.
  - Ensure **bounding boxes and landmarks remain consistent** across transformations.

- **Landmark Scaling:**
  - Normalize landmark positions before adding them as a **4th channel**.

- **Final Output Structure:**
  - **Tensors:** `raw_tensors.pt`, `enhanced_tensors.pt`
  - **Bounding Boxes:** `mtcnn_boxes.pt`
  - **Landmarks:** `landmarks.pt`
  - **Metadata:** `results.csv`
  - **Final Dataset:** Cropped & resized images for training.

I'll review the **MetricasClasificacionMulticlase.pdf** document now and extract all required metrics, evaluations, and necessary modifications to our training workflow. I will also confirm if additional metrics or considerations are mentioned beyond our current list. Give me a moment.

The document **"MetricasClasificacionMulticlase.pdf"** outlines the necessary evaluation metrics and considerations for a multiclass classification problem. Here’s a **structured breakdown** of what must be implemented:

---

### **Required Modifications & Additions Based on the Document**
#### **1. Changes to `train()`**
   - ✅ **Return loss history** (per epoch).
   - ✅ **Log validation loss** at each epoch.
   - ✅ **Save the best model (based on validation loss or accuracy)**.

#### **2. Changes to `accuracy()`**
   - ✅ Instead of returning just accuracy, return `predictions` and `true labels` for **further metric calculations**.

---

### **Additional Metrics Required**
Beyond **accuracy, precision, recall, and F1-score**, the document mentions:

#### **1. Macro/Micro Averaging for Precision, Recall, and F1**
   - **Micro**: Computes metrics globally (suitable when class imbalance exists).
   - **Macro**: Averages the metric per class (treats all classes equally).

#### **2. Balanced Accuracy**
   - `Balanced Accuracy = (Recall per class) / Number of Classes`
   - Useful for datasets with **class imbalance**.

#### **3. ROC-AUC per Class (if applicable)**
   - Since we are dealing with **multiclass classification**, we need a **one-vs-rest (OvR) approach**.

#### **4. Cohen’s Kappa**
   - Measures **inter-rater agreement** beyond random chance:
   - `Kappa = (Observed Agreement - Expected Agreement) / (1 - Expected Agreement)`
   - Helps assess **model reliability**.

#### **5. Matthews Correlation Coefficient (MCC)**
   - `MCC = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))`
   - A robust **correlation coefficient** for classification.

#### **6. Log-Loss**
   - Measures the uncertainty of predictions.
   - `LogLoss = - Σ (y_true * log(y_pred)) / N`

#### **7. Per-Class Report (Full Breakdown)**
   - Includes all metrics per class (accuracy, precision, recall, F1).

---

### **Updated Roadmap for Implementation**
Since some metrics depend on others, the order should be:

1. ✅ **Modify `train()`** to return loss history and best model.
2. ✅ **Modify `accuracy()`** to return **predictions and true labels**.
3. ✅ **Implement `confusion_matrix()`** (used in many metrics).
4. ✅ **Implement `precision()`, `recall()`** (macro/micro versions).
5. ✅ **Implement `f1_score()`** (macro/micro versions).
6. ✅ **Implement `balanced_accuracy()`**.
7. ✅ **Implement `cohen_kappa()`**.
8. ✅ **Implement `matthews_correlation_coefficient()`**.
9. ✅ **Implement `log_loss()`**.
10. ✅ **Implement `classification_report()`** (full per-class breakdown).
11. ✅ **Implement `roc_auc()` (one-vs-rest method).**

---

### **Next Steps**
Since `train()` and `accuracy()` need changes first, do you want to:
1. **Review the modified `train()` and `accuracy()` first**, or


