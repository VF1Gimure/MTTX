from collections import deque

from utils.capture.cv2VideoCapture_thread import cv2_video_file_capture
from utils.capture.frame_processor_thread import FrameProcessorThread
import cv2
from facenet_pytorch import MTCNN
import numpy as np
from utils.capture.DatasetSaver import DatasetSaver
import os
import cv2
from facenet_pytorch import MTCNN
from utils.capture.DatasetSaver import DatasetSaver_frames
from tqdm import tqdm

def process_videos_in_directory(directory_path, split_index=None):
    """
    Processes all video files in the given directory with optional category splitting logic.

    Args:
        directory_path (str): Path to the directory containing video files.
        split_index (int or None): Index of the category when splitting the filename.
                                   If None, uses the entire filename (without extension) as the category.
    """
    # Initialize face detector
    detector = MTCNN(image_size=640, device="cpu")

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".mov"):  # Process only .mov files
            video_path = os.path.join(directory_path, filename)

            # Extract category based on the split_index
            if split_index is not None:
                # Split the filename and extract the category based on the index
                parts = filename.rsplit(".", 1)[0].split("_")
                category = parts[split_index] if len(parts) > split_index else "unknown"
            else:
                # Use the filename (without extension) as the category
                category = filename.rsplit(".", 1)[0]

            print(f"[INFO] Processing video: {filename}, Category: {category}")

            # Load video
            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            dataset_saver = DatasetSaver_frames(category=category, max_frames=frame_count)

            # Use tqdm for progress tracking
            with tqdm(total=frame_count, desc=f"Processing {filename}", unit="frame") as pbar:
                while not dataset_saver.stop_flag:
                    ret, frame = video.read()
                    if not ret:
                        print(f"[INFO] End of video reached: {filename}")
                        break  # Stop when video ends

                    # Process frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes, confidences = detector.detect(frame_rgb)

                    # Save frame and detections
                    dataset_saver.save_frame(frame=frame, detections=(boxes, confidences))

                    # Update tqdm progress bar
                    pbar.update(1)

            video.release()
            print(f"[INFO] Finished processing {filename}.")

    print("[INFO] All videos processed successfully.")


if __name__ == "__main__":
    directory_path = "/data/kxed_copy/esni_emotions"
    split_index = 1 # None
    process_videos_in_directory(directory_path, split_index)