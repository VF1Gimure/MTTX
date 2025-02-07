import cv2
from threading import Thread
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
        while not self.stopped:
            grabbed, frame = self.vcap.read()
            if not grabbed:
                print("[INFO]: Video file finished.")
                self.stop()
                break

            self.grabbed = grabbed
            self.frame = frame

        self.vcap.release()  # Release capture when stopping

    def read(self):
        return self.frame if self.grabbed else None

    def stop(self):
        """Stops the video capture thread."""
        self.stopped = True
        self.t.join()
