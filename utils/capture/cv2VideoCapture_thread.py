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

class cv2_video_file_capture:
    def __init__(self, video_path):
        self.video_path = video_path
        self.vcap = cv2.VideoCapture(self.video_path)

        if not self.vcap.isOpened():
            print(f"[Exiting]: Error accessing video file: {self.video_path}")
            exit(0)

        self.grabbed, self.frame = self.vcap.read()
        if not self.grabbed:
            print("[Exiting]: No frames in video file")
            exit(0)

        self.stopped = False
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.t.start()

    def update(self):
        """Continuously read frames from the video file until stopped."""
        while not self.stopped:
            grabbed, frame = self.vcap.read()
            if not grabbed:
                print("[INFO]: Video file finished.")
                self.stopped = True  # Stop loop when video ends
                break

            self.grabbed = grabbed
            self.frame = frame

        self.vcap.release()

    def read(self):
        return self.frame if self.grabbed else None

    def stop(self):
        """Stops the video capture thread."""
        self.stopped = True
        self.t.join()
