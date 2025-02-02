import cv2
import os
import time
import numpy as np


class DatasetSaver:
    def __init__(self, dataset_name="KnightX_exp_dataset", category="neutral", max_duration=5):
        self.category = category
        self.max_duration = max_duration
        self.start_time = time.time()
        self.base_path = "/Users/lsfu/Desktop/MNA/Integrador/MTTX/data"
        self.dataset_path = os.path.join(self.base_path, dataset_name)
        self.full_frame_path = os.path.join(self.dataset_path, "data", self.category)
        self.cropped_face_path = os.path.join(self.dataset_path, "face", self.category)
        os.makedirs(self.full_frame_path, exist_ok=True)
        os.makedirs(self.cropped_face_path, exist_ok=True)
        self.stop_flag = False

    def save_frame(self, frame, detections):
        if detections is None:
            return
        boxes, confidences = detections

        if boxes is None:
            return

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Stop the program when max_duration is reached
        if elapsed_time >= self.max_duration:
            print("Max duration reached. Stopping program.")
            self.stop_flag = True

        timestamp = int(current_time * 1000)  # Unique filename timestamp

        # Save full frame
        frame_filename = os.path.join(self.full_frame_path, f"frame_{timestamp}.jpg")
        cv2.imwrite(frame_filename, frame)

         # Some detectors return (boxes, confidences)

        if len(boxes) > 0:
            # Find index of the highest confidence detection
            max_index = np.argmax(confidences)

            # Extract the best bounding box
            x1, y1, x2, y2 = map(int, boxes[max_index])
            cropped_face = frame[y1:y2, x1:x2]

            # Ensure the face is valid before saving
            if cropped_face.size > 0:
                face_filename = os.path.join(self.cropped_face_path, f"face_{timestamp}.jpg")
                cv2.imwrite(face_filename, cropped_face)