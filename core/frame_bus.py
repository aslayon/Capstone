# core/frame_bus.py
import threading

class FrameBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def publish(self, frame):
        # frame: BGR ndarray
        with self._lock:
            self._frame = frame.copy()

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

BUS = FrameBus()  # 싱글톤처럼 써도 됨
