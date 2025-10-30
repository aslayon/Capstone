# reid_bank_ext.py
from collections import deque
import cv2, numpy as np
from typing import List, Tuple, Optional

class ReIDBank:
    def __init__(self, maxlen=10, h_bins=25, s_bins=30):
        self.items = deque(maxlen=maxlen)
        self.h_bins, self.s_bins = h_bins, s_bins

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
        hist = cv2.calcHist([hsv], [0,1], mask, [self.h_bins, self.s_bins], [0,180, 0,256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)

    def add_from_frame(self, frame, box):
        crop = self._crop(frame, box)
        if crop is None: return False
        self.items.append(self._hist_hs(crop))
        return True

    def add_from_image_path(self, img_path: str) -> bool:
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None: return False
        self.items.append(self._hist_hs(bgr))
        return True

    @staticmethod
    def bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
        return cv2.compareHist(h1.reshape(-1,1), h2.reshape(-1,1), cv2.HISTCMP_BHATTACHARYYA)

    def score_to_gallery(self, frame, box) -> Optional[float]:
        if not self.items: return None
        crop = self._crop(frame, box)
        if crop is None: return None
        h = self._hist_hs(crop)
        return float(min(self.bhattacharyya(h, g) for g in self.items))

    # ===== New: utilities =====
    def save_npz(self, path: str) -> None:
        if len(self.items) == 0:
            np.savez_compressed(path, items=np.empty((0, self.h_bins*self.s_bins), np.float32))
        else:
            np.savez_compressed(path, items=np.stack(self.items, axis=0))

    def load_npz(self, path: str) -> bool:
        try:
            arr = np.load(path)["items"].astype(np.float32)
            self.items.clear()
            for row in arr:
                self.items.append(row)
            return True
        except Exception:
            return False

    def median_hist(self) -> Optional[np.ndarray]:
        if not self.items: return None
        arr = np.stack(self.items, axis=0)  # (N,D)
        return np.median(arr, axis=0).astype(np.float32)

    def merge_decay(self, new_hist: np.ndarray, alpha: float = 0.2):
        """지수감쇠로 최신 히스토그램을 반영 (0<alpha<=1)."""
        med = self.median_hist()
        if med is None:
            self.items.append(new_hist.astype(np.float32))
        else:
            merged = (1 - alpha) * med + alpha * new_hist
            merged = cv2.normalize(merged, None).astype(np.float32)
            self.items.append(merged)

    def score_batch(self, frame, boxes: List[Tuple[int,int,int,int]]) -> List[Optional[float]]:
        scores = []
        for b in boxes:
            scores.append(self.score_to_gallery(frame, b))
        return scores

    def pick_best_across_tri(
        self,
        lf, cf, rf,
        tracks_L: List[Tuple[int,int,int,int,int]],
        tracks_C: List[Tuple[int,int,int,int,int]],
        tracks_R: List[Tuple[int,int,int,int,int]],
        thresh: float = 0.45
    ):
        """
        tracks_*: [(tid, x1,y1,x2,y2), ...]
        반환: (seg, tid, dist) or (None, None, None)  // dist가 thresh보다 작을 때만 유효
        """
        if not self.items: return (None, None, None)
        cand = []
        for seg, frm, tracks in (("L", lf, tracks_L), ("C", cf, tracks_C), ("R", rf, tracks_R)):
            for tid, x1,y1,x2,y2 in tracks:
                d = self.score_to_gallery(frm, (x1,y1,x2,y2))
                if d is not None:
                    cand.append((seg, tid, d))
        if not cand: return (None, None, None)
        seg, tid, dist = min(cand, key=lambda x: x[2])
        return (seg, tid, dist) if dist < thresh else (None, None, None)
