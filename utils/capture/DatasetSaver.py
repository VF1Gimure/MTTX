import cv2
import os
import time
import numpy as np


class DatasetSaver:
    def __init__(self, dataset_name="KnightX_exp_dataset", category="neutral", max_duration=5, ):
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


class DatasetSaver_frames:
    def __init__(self, dataset_name="KnightX_exp_dataset", category="neutral", max_frames=100, frame_name="name"):
        self.category = category
        self.max_frames = max_frames
        self.frame_name = frame_name+category[0].upper()+"_"
        self.frame_count = 0  # Initialize frame counter
        self.base_path = "/Users/lsfu/Desktop/MNA/Integrador/MTTX/data/raw"
        self.dataset_path = os.path.join(self.base_path, dataset_name)
        self.full_frame_path = os.path.join(self.dataset_path, self.category)
        #self.cropped_face_path = os.path.join(self.dataset_path, "face", self.category)
        os.makedirs(self.full_frame_path, exist_ok=True)
#        os.makedirs(self.cropped_face_path, exist_ok=True)
        self.stop_flag = False

    def save_frame(self, frame):
        self.frame_count += 1

        # Stop the program when max_frames is reached
        if self.frame_count >= self.max_frames:
            print(f"Max Frames ({self.max_frames}) Processed.")
            self.stop_flag = True
            return  # Stop saving further frames

        # Save full frame
        frame_filename = os.path.join(self.full_frame_path, f"{self.frame_name}{self.frame_count}.jpg")

        cv2.imwrite(frame_filename, frame)
