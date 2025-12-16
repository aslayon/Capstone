# core/reid_bank.py
from collections import deque
import cv2, glob, os
import numpy as np
from typing import List, Optional  # ← 추가

class ReIDBank:
    """선택 차량의 최근 히스토그램 특징을 보관/비교 (HSV HS 2D hist)"""
    def __init__(self, maxlen=10, h_bins=25, s_bins=30):   # ← 5장 고정
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
        bgr = cv2.GaussianBlur(bgr, (3,3), 0)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H, W = hsv.shape[:2]
        mask = np.zeros((H,W), np.uint8)
        cx, cy = W//2, H//2
        cv2.ellipse(mask, (cx,cy), (W//3, H//3), 0, 0, 360, 255, -1)
        hist = cv2.calcHist([hsv], [0, 1], mask,
                            [self.h_bins, self.s_bins],
                            [0,180, 0,256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def add_from_frame(self, frame, box, also_return_hist=False):
        crop = self._crop(frame, box)
        if crop is None: return None if also_return_hist else False
        h = self._hist_hs(crop)
        self.items.append(h)
        return h if also_return_hist else True

    # === 5장 제한 수집 ===
    def size(self) -> int:
        return len(self.items)

    def clear(self):
        self.items.clear()

    def add_from_frame_until_full(self, frame, box) -> bool:
        if self.size() >= self.items.maxlen:
            return False
        return self.add_from_frame(frame, box)

    # === 평균 Bhattacharyya (작을수록 유사) ===
    def _hist_from_frame(self, frame, box) -> Optional[np.ndarray]:
        crop = self._crop(frame, box)
        if crop is None: return None
        return self._hist_hs(crop)

    def avg_bhatta_to_hist(self, h: np.ndarray) -> Optional[float]:
        if not self.items: return None
        dists = [cv2.compareHist(h.reshape(-1,1), g.reshape(-1,1), cv2.HISTCMP_BHATTACHARYYA)
                 for g in self.items]
        return float(np.mean(dists))

    def avg_bhatta_to_box(self, frame, box) -> Optional[float]:
        h = self._hist_from_frame(frame, box)
        if h is None: return None
        return self.avg_bhatta_to_hist(h)

    def is_same_vehicle(self, frame, box, thresh: float) -> bool:
        m = self.avg_bhatta_to_box(frame, box)
        return (m is not None) and (m <= thresh)

    # === 사전 저장 5장 로드 ===
    def load_from_paths(self, paths: List[str], limit: int = None) -> int:
        n = 0
        for p in (paths if limit is None else paths[:limit]):
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None: continue
            self.items.append(self._hist_hs(bgr))
            n += 1
            if self.size() >= self.items.maxlen:
                break
        return n

    def load_last5_from_dir(self, dirpath: str, pattern: str = "*.jpg", clear_first: bool = True) -> int:
        if clear_first:
            self.clear()
        files = sorted(
            glob.glob(os.path.join(dirpath, pattern)),
            key=lambda p: os.path.getmtime(p),
            reverse=True
        )
        # 최신 5장만
        return self.load_from_paths(files, limit=min(5, self.items.maxlen))

    # (유지) 최소거리 스코어가 필요할 때
    @staticmethod
    def bhattacharyya(h1, h2):
        h1 = h1.reshape(-1,1).astype(np.float32)
        h2 = h2.reshape(-1,1).astype(np.float32)
        return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    def score_to_gallery(self, frame, box):
        if not self.items: return None
        crop = self._crop(frame, box)
        if crop is None: return None
        h = self._hist_hs(crop)
        return float(min(self.bhattacharyya(h, g) for g in self.items))
