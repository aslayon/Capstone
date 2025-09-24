# mouse_handler.py
# 마우스 콜백 분리
import cv2

class MouseSelector:
    """
    - 윈도우 이름/스케일/트래커 참조를 받아 클릭시 select_track_by_point 호출
    - selected_id 토글은 tracker_test.py 내부 로직을 그대로 사용
    """
    def __init__(self, window_name: str, tracker, get_frame_shape_callable, get_scale_callable):
        self.window_name = window_name
        self.tracker = tracker
        self.get_frame_shape = get_frame_shape_callable  # () -> (h, w)
        self.get_scale = get_scale_callable              # () -> scale (float)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        fh, fw = self.get_frame_shape()
        scale = self.get_scale() or 1.0
        if not fh or not fw:
            return
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        try:
            self.tracker.select_track_by_point(orig_x, orig_y)
        except Exception as e:
            print("[WARN] select_track_by_point failed:", e)
