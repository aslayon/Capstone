# reid를 위한 히스토그램 특징 보관/비교


# core/reid_bank.py
from collections import deque
import cv2
import numpy as np

class ReIDBank:
    """선택 차량의 최근 히스토그램 특징을 보관/비교 (HSV HS 2D hist)"""
    def __init__(self, maxlen=10, h_bins=25, s_bins=30):
        self.items = deque(maxlen=maxlen)
        self.h_bins = h_bins
        self.s_bins = s_bins

    @staticmethod
    def _crop(frame, box, pad=4):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w-1, x2 + pad); y2 = min(h-1, y2 + pad)
        if x2 <= x1 or y2 <= y1: return None
        return frame[y1:y2, x1:x2]

    def _hist_hs(self, bgr):
        # ① 노이즈 완화
        bgr = cv2.GaussianBlur(bgr, (3,3), 0)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # ② 중앙 마스크(배경 영향 감소)
        H, W = hsv.shape[:2]
        mask = np.zeros((H,W), np.uint8)
        cx, cy = W//2, H//2
        cv2.ellipse(mask, (cx,cy), (W//3, H//3), 0, 0, 360, 255, -1)
        # ③ HS 2D hist
        hist = cv2.calcHist([hsv], [0, 1], mask,
                            [self.h_bins, self.s_bins],
                            [0,180, 0,256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def add_from_frame(self, frame, box, also_return_hist=False):
        crop = self._crop(frame, box)
        if crop is None: return None if also_return_hist else False
        h = self._hist_hs(crop)
        self.items.append(h)
        return h if also_return_hist else True

    @staticmethod
    def bhattacharyya(h1, h2):
        # OpenCV Hellinger ≈ Bhattacharyya
        h1 = h1.reshape(-1,1).astype(np.float32)
        h2 = h2.reshape(-1,1).astype(np.float32)
        return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    def score_to_gallery(self, frame, box):
        """갤러리와의 최소 거리(낮을수록 유사), 갤러리가 비면 None"""
        if not self.items: return None
        crop = self._crop(frame, box)
        if crop is None: return None
        h = self._hist_hs(crop)
        return float(min(self.bhattacharyya(h, g) for g in self.items))
