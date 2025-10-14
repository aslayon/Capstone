# core/mouse_tri.py
# tri 윈도우에서 클릭으로 좌/중/우 CCTV 전환
    
import cv2

class TriClickHandler:
    """
    tri 윈도우(2160x480 등)에서 클릭 위치로 스위칭:
    - 좌 클릭 → left CCTV
    - 중 클릭 → 유지
    - 우 클릭 → right CCTV
    """
    def __init__(self, window_name, switcher, get_disp_scale, get_total_w_callable):
        self.win = window_name
        self.switcher = switcher
        self.get_scale = get_disp_scale         # () -> float
        self.get_total_w = get_total_w_callable # () -> int
        cv2.setMouseCallback(self.win, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        scale = self.get_scale() or 1.0
        total_w = self.get_total_w() or 2160
        orig_x = int(x / scale)
        self.switcher.on_triple_click(orig_x, total_w)
