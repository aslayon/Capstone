# stream.py
import cv2
import threading

class VideoStreamThread:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.grab():
                ret, frame = self.cap.retrieve()
                if ret:
                    with self.lock:
                        self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
