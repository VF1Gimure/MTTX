# **Pipeline Steps**

1. **STEP: Apply CLAHE + UNSHARP**
   Preprocess the input images by applying CLAHE and unsharp masking to enhance contrast and sharpness.

2. **STEP: Detect Face Box using MTCNN**
   Use MTCNN to detect the bounding box of the face area in the preprocessed images.

3. **STEP: Extract Landmarks using Dlib 68 with CLAHE + UNSHARP**
   Apply Dlib's 68 facial landmarks detector to the detected face box while keeping CLAHE + UNSHARP applied.

4. **STEP: Save Original Results**
   Save the original bounding boxes, landmarks, and image sizes in a DataFrame (CSV format) for future processing.

5. **STEP: Recalculate Results After Resize**
   Resize the images (e.g., to 512x512), but adjust the bounding box and landmarks coordinates proportionally to maintain their relative positions.

6. **STEP: Crop Face with Padding**
   Crop the face region using the MTCNN box, adding a small padding (e.g., 1-3 pixels) to ensure all facial features are included.

7. **STEP: Save Adjusted Results**
   Save the resized images, cropped faces, and recalculated landmarks into a new dataset.

8. **STEP: Create "4-Channel" Images**
   Add the recalculated landmarks as a 4th channel to the cropped face images. This integrates spatial information about the landmarks into the image.

9. **STEP: Prepare Data for Model**
   Organize the 4-channel images (crops + landmarks) as input data for the model training.

10. **STEP: Train the Model**
    Use the prepared 4-channel images to train the model for the desired task (e.g., facial analysis or expression recognition).

---

### **Considerations**
- **Resizing Impact on Results:**
  - Calculate new bounding box and landmark positions using a proportional scaling factor based on the original and resized image dimensions.
  - Ensure the recalculated landmarks and bounding boxes align correctly with the resized images.

- **Cropping with Padding:**
  - Before resizing, crop the face area using the MTCNN box with padding to preserve important facial regions.

- **Saving Intermediate Results:**
  - Save all intermediate results (bounding boxes, landmarks, resized images) in a well-structured format (e.g., CSV + image folders) for traceability.

- **Landmarks as a 4th Channel:**
  - Normalize the landmark positions to the image dimensions to ensure compatibility when adding them as a 4th channel.

---