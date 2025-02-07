from collections import deque
import cv2
from facenet_pytorch import MTCNN
from threading import Thread, Event

# Initialize the MTCNN detector
detector = MTCNN(image_size=640, device='cpu')

# Shared deque for frames and results
frame_list = deque(maxlen=30)  # Keeps the last 30 frames
result_list = deque(maxlen=30)  # Keeps the last 30 results

# Event to signal threads to stop
stop_event = Event()

# Initialize the video capture object
cap = cv2.VideoCapture(0)


def video_thread():
    global cap
    print("video thread")
    """Continuously captures frames from the webcam."""
    while not stop_event.is_set():
        _, frame = cap.read()  # Capture the frame
        frame_list.append(frame)


def processing_thread():
    """Processes frames from the deque and runs face detection."""
    while not stop_event.is_set():
        frame = frame_list.pop()  # Get the most recent frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        boxes, confidences = detector.detect(frame_rgb)  # Run face detection
        result_list.append((frame, boxes, confidences))  # Append result


def main():
    video_t = Thread(target=video_thread)
    video_t.start()
    process_t = Thread(target=processing_thread)

    while not frame_list:
        pass
    else:
        process_t.start()

    try:
        while True:
            if result_list:
                frame, boxes, confidences = result_list.pop()  # Get the most recent result

                # Draw bounding boxes and annotations
                if boxes is not None:
                    for box, confidence in zip(boxes, confidences):
                        if confidence is None:
                            continue
                        x1, y1, x2, y2 = [int(b) for b in box]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f'{confidence:.2f}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Display the frame with face detection
                cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Signal threads to stop
        print("Exiting")
        stop_event.set()
        video_t.join()
        process_t.join()
        cap.release()  # Release the video capture object
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
