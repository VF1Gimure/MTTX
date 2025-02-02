import cv2
from threading import Thread


class cv2VideoCapture_thread:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id
        # self.transformer = transformer
        self.vcap = cv2.VideoCapture(self.stream_id)

        if not self.vcap.isOpened():
            print("[Exiting]: Error accessing video stream.")
            exit(0)

        self.grabbed, self.frame = self.vcap.read()
        if not self.grabbed:
            print("[Exiting]: No more frames to read")
            exit(0)

        self.stopped = True
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while not self.stopped:
            grabbed, frame = self.vcap.read()
            if not grabbed:
                print("[Exiting]: No more frames to read")
                self.stop()
                break

            self.grabbed = grabbed
            self.frame = frame

        self.vcap.release()  # Release capture when stopping

    def read(self):
        return self.frame

    def stop(self):
        """Stops the video capture thread."""
        self.stopped = True
        self.t.join()

