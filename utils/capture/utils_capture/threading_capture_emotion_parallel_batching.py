from utils.frame_processor_thread_batching import FrameProcessorThread_batching
from utils.capture.cv2VideoCapture_thread import cv2VideoCapture_thread
import cv2


def initialize_system():
    """Initializes the webcam and detector."""
    #detector = MTCNN(image_size=640, device='cpu', keep_all=True)
    webcam = cv2VideoCapture_thread(stream_id=0)
    webcam.start()

    # Create the FrameProcessorThread with dynamic buffer handling
    processor = FrameProcessorThread_batching(detector_id=1,webcam=webcam,max_workers=12,buffer_threshold=5)
    processor.start()

    return webcam, processor

def process_and_display(processor, webcam):
    """Continuously fetch and display processed frames."""
    while True:
        overlays = processor.read()  # Get the latest processed frames

        if overlays:  # Ensure there are frames to display
            for (overlay, frame) in overlays:
                display_frame = cv2.addWeighted(frame, 0.7, overlay, 1, 0)
                cv2.imshow('Face Detection - Buffered Playback', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit on 'q' key

def main():
    """Main execution loop."""
    webcam, processor = initialize_system()
    processor.start()  # Start frame processing thread

    try:
        process_and_display(processor, webcam)
    finally:
        print("Exiting...")
        processor.stop()
        webcam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
