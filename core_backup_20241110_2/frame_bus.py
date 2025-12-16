# core/frame_bus.py
# 🚀 최적화 버전 - FPS 향상

import threading

class FrameBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0  # 프레임 변경 감지용

    def publish(self, frame):
        """
        프레임 발행 (원본 참조 저장)
        
        주의: frame은 publish 후 수정하면 안 됨!
        pipeline.py에서 BUS.publish() 직전에 복사본 만들기
        """
        with self._lock:
            self._frame = frame  # ← copy() 제거! 참조만 저장
            self._frame_id += 1

    def latest(self):
        """
        최신 프레임 반환 (원본 참조)
        
        주의: 반환된 frame은 읽기 전용으로 사용!
        """
        with self._lock:
            return self._frame  # ← copy() 제거! 참조만 반환

    def get_frame_id(self):
        """프레임 변경 감지용 ID"""
        with self._lock:
            return self._frame_id

# 싱글톤
BUS = FrameBus()