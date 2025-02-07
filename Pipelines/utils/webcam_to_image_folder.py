from collections import deque

from utils.capture.cv2VideoCapture_thread import cv2VideoCapture_thread
from utils.capture.frame_processor_thread import FrameProcessorThread
import cv2
from facenet_pytorch import MTCNN
import numpy as np
from utils.capture.DatasetSaver import DatasetSaver_frames

display_fps = 0
processing_fps = 0
frame_count = 0

fps_list = deque(maxlen=20)
fps_list.append(1)

def initialize_system():
    """Initializes the webcam and detector."""
    detector = MTCNN(image_size=640, device='cpu')
    webcam = cv2VideoCapture_thread(stream_id=0)
    webcam.start()
    processor = FrameProcessorThread(detector_id=1, webcam=webcam, detector=detector)
    processor.start()
    return detector, webcam, processor
# Reset timer

def process_frame(webcam, processor, dataset_saver):
    """Fetches the latest frame and overlay from the processor."""
    #latest_frame = webcam.read()

    overlay_frame, detection, frame, diff = processor.read()

    dataset_saver.save_frame(frame=frame, detections=detection)

    #if latest_frame is None or overlay_frame is None:
    #    return None

    display_frame = frame.copy()
    display_frame = cv2.addWeighted(display_frame, 0.7, overlay_frame, 1, 0)
    fps_list.append(diff)
    # Compute FPS: (total frames - 1) / sum(diff(tms))
    time_mean = np.mean(fps_list)  # Compute differences
    #total_time = np.sum(time_mean)  # Sum of time intervals
    #print(time_mean)
    fps = 1 / time_mean


    # Display FPS
    cv2.putText(display_frame, f'processing FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    return display_frame


def main():
    """Main execution loop."""
    detector, webcam, processor = initialize_system()
    dataset_saver = DatasetSaver_frames(category="surprise", max_frames=300)

    try:
        while not dataset_saver.stop_flag:
            display_frame = process_frame(webcam, processor, dataset_saver)

            if display_frame is not None:
                cv2.imshow('Face Detection', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Exiting")
        processor.stop()
        webcam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()