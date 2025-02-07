from threading import Thread, Event
import cv2
from facenet_pytorch import MTCNN
import time
import numpy as np
import queue  # For frame buffering

# Initialize MTCNN with batch processing support
detector = MTCNN(image_size=640, device='cpu')

# Global variables
cap = cv2.VideoCapture(0)
stop_event = Event()

# FPS tracking
display_fps = 0
processing_fps = 0
frame_count = 0
fps_start_time = time.time()

# Frame queue for buffering
frame_queue = queue.Queue(maxsize=20)  # Buffer ~0.5s (20 frames at 40 FPS)

# Overlay storage
overlay_frame = np.zeros((480, 640, 3), dtype=np.uint8)


def video_thread():
    """Continuously capture frames and add to the queue."""
    global frame_count
    print("Video thread started")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        if frame_queue.full():
            frame_queue.get()  # Remove the oldest frame to prevent overflow

        frame_queue.put(frame)
        frame_count += 1


def processing_thread():
    """Process frames in batches and run MTCNN."""
    global overlay_frame, processing_fps
    batch_size = 5  # Adjust batch size based on performance

    while not stop_event.is_set():
        batch = []

        # Collect frames for batch processing
        while len(batch) < batch_size and not frame_queue.empty():
            batch.append(frame_queue.get())

        if len(batch) == 0:
            continue  # Skip if no frames are available

        batch_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in batch]
        boxes_list, confidences_list = detector.detect(batch_rgb)

        # Create overlay
        overlay = np.zeros_like(batch[-1])  # Use the last frame in batch

        for boxes, confidences in zip(boxes_list, confidences_list):
            if boxes is None:
                continue
            for box, confidence in zip(boxes, confidences):
                if confidence is None:
                    continue
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(overlay, f'{confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        overlay_frame = overlay  # ✅ Update overlay

        processing_fps = len(batch) / (time.time() - fps_start_time)


def main():
    global display_fps, frame_count, fps_start_time

    # Start video and processing threads
    video_t = Thread(target=video_thread)
    process_t1 = Thread(target=processing_thread)  # Worker 1
    process_t2 = Thread(target=processing_thread)  # Worker 2

    video_t.start()
    process_t1.start()
    process_t2.start()
    try:
        while True:
            elapsed_time = time.time() - fps_start_time

            if elapsed_time >= 1.0:
                display_fps = frame_count / elapsed_time
                frame_count = 0
                fps_start_time = time.time()

            if not frame_queue.empty():
                display_frame = frame_queue.queue[-1].copy()  # Latest frame
            else:
                display_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            display_frame = cv2.addWeighted(display_frame, 0.7, overlay_frame, 1, 0)

            # Display FPS
            cv2.putText(display_frame, f'Display FPS: {int(display_fps)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f'Processing FPS: {int(processing_fps)}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('Face Detection', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Exiting")
        stop_event.set()
        video_t.join()
        process_t1.join()
        process_t2.join()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
