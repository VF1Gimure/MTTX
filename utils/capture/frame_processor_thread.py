import cv2
import numpy as np
from threading import Thread
import time
class FrameProcessorThread:
    def __init__(self, detector_id, webcam, detector):
        self.detector_id = detector_id
        self.webcam = webcam
        self.detector = detector

        self.overlay_frame = None
        self.detection = None
        self.diff = 0.0
        self.stopped = True
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.frame = None
        #self.processing_fps = 0
        #self.last_time = time.time()

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while not self.stopped:
            start = time.time()
            self.frame = self.webcam.read()
            #self.frame = cv2.resize(self.frame, (640, 640))
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.detection = self.detector.detect(frame_rgb)
            self.overlay_frame = np.zeros_like(self.frame) # can be made into a default
            self.diff = time.time() - start
            # Calculate processing FPS
            #current_time = time.time()
            #self.processing_fps = 1 / (current_time - self.last_time)
            #self.last_time = current_time

    def read(self):
        if self.detection is not None:
            self.open()
        else:
            self.overlay_frame = np.zeros_like(self.frame)
        return self.overlay_frame, self.detection, self.frame, self.diff

    def stop(self):
        """Stops the face processing  thread."""
        self.stopped = True
        self.t.join()

    def open(self):
            boxes, confidences = self.detection
            if boxes is not None:
                for box, confidence in zip(boxes, confidences):
                    if confidence is None:
                        continue
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(self.overlay_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(self.overlay_frame, f'{confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
