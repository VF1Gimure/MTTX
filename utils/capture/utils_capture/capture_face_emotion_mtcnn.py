import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms

import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN

# Initialize the MTCNN detector
detector = MTCNN(image_size=640, device='cpu', keep_all=True)

# Initialize time for FPS calculation
frame_count = 0
fps_start_time = time.time()

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, confidences = detector.detect(frame_rgb)
    return boxes, confidences

cap = cv2.VideoCapture(0)

with ThreadPoolExecutor(max_workers=1) as executor:
    while True:
        start_time = time.time()
        _, frame = cap.read()

        frame = cv2.resize(frame, (640, 640))

        future = executor.submit(process_frame, frame)

        boxes, confidences = future.result()

        if boxes is not None:
            for box, confidence in zip(boxes, confidences):
                if confidence is None:
                    continue
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        frame_count += 1
        elapsed_time = time.time() - fps_start_time

        if elapsed_time >= 1.0:  # Update FPS every second
            fps = frame_count / elapsed_time
            frame_count = 0
            fps_start_time = time.time()  # Reset time

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()