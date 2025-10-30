# core/reid_bank.py
from collections import deque
import cv2, glob, os
import numpy as np
from typing import List, Optional  # ← 추가


def _pre_norm_color(bgr):
        # Gray-world white balance
        b,g,r = cv2.split(bgr.astype(np.float32))
        mean = (b.mean()+g.mean()+r.mean())/3.0 + 1e-6
        b *= mean/(b.mean()+1e-6); g *= mean/(g.mean()+1e-6); r *= mean/(r.mean()+1e-6)
        wb = np.clip(cv2.merge([b,g,r]), 0, 255).astype(np.uint8)
        # 가벼운 감마 보정(어두우면 밝게)
        v = cv2.cvtColor(wb, cv2.COLOR_BGR2HSV)[:,:,2].mean()
        gamma = 0.9 if v > 110 else (0.75 if v > 80 else 0.65)
        lut = np.array([(i/255.0)**gamma*255 for i in range(256)], np.uint8)
        return cv2.LUT(wb, lut)


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

    # ReIDBank 내부

    def set_origin(self, seg: str = None, cam: str = None):
        """최초 포착 화면/카메라 메타를 저장. 이미 값이 있으면 유지."""
        if not hasattr(self, "origin_seg"):
            self.origin_seg = None
        if not hasattr(self, "origin_cam"):
            self.origin_cam = None
        if self.origin_seg is None and seg is not None:
            self.origin_seg = str(seg)
        if self.origin_cam is None and cam is not None:
            self.origin_cam = str(cam)

    
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
        self.items_band5.clear()
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

    
    # ====== ReIDBank 안에 추가 ======

    def _ensure_band5_list(self):
        if not hasattr(self, "items_band5"):
            from collections import deque
            self.items_band5 = deque(maxlen=5)  # 5장 고정

    @staticmethod
    def _band_crops_equal(bgr, n=5, min_band_h=10):
        """세로로 n등분(기본 5밴드). 얇아지면 None 반환."""
        h, w = bgr.shape[:2]
        band_h = h // n
        if band_h < min_band_h:
            return [None] * n
        bands = []
        y = 0
        for i in range(n):
            y2 = h if i == n-1 else (y + band_h)
            bands.append(bgr[y:y2, :])
            y = y2
        return bands

    def add_from_frame_banded5(self, frame, box, pad=4, origin_seg=None, origin_cam=None):
        """bbox를 5밴드로 나눠 HS hist 저장. 한 샘플 = [h0..h4]."""
        
        try:
            self.set_origin(origin_seg, origin_cam)
        except Exception:
            pass
        
        self._ensure_band5_list()
        # ✅ 이미 꽉 찼으면 즉시 반환 (로그/성능 방지)
        if len(self.items_band5) >= self.items_band5.maxlen:
            return False

        crop = self._crop(frame, box, pad=pad)
        if crop is None:
            return False
        #crop = _pre_norm_color(crop)
        crop = cv2.GaussianBlur(crop, (3,3), 0)

        bands = self._band_crops_equal(crop, n=5, min_band_h=10)
        hists = []
        for band in bands:
            if band is None or band.size == 0:
                hists.append(None); continue
            hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0,1], None,
                                [self.h_bins, self.s_bins],
                                [0,180, 0,256])
            hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
            hists.append(hist)

        if not any(h is not None for h in hists):
            return False

        self.items_band5.append(hists)  # ✅ deque(maxlen=5)라 5장 초과 불가
        return True


    def size5(self) -> int:
        """5밴드 갤러리 샘플 개수"""
        return len(getattr(self, "items_band5", []))

    def avg_bhatta_to_box_banded5(self, frame, box, pad=4):
        """
        쿼리 bbox를 5밴드로 나눠 갤러리(5밴드)와 밴드별 Bhattacharyya 평균을 계산.
        반환: (d_mean, [d0,d1,d2,d3,d4])  // d는 0~1, 작을수록 유사. 유효밴드만 평균.
        """
        if not hasattr(self, "items_band5") or len(self.items_band5) == 0:
            return None, None

        crop = self._crop(frame, box, pad=pad)
        if crop is None:
            return None, None
        #crop = _pre_norm_color(crop)
        crop = cv2.GaussianBlur(crop, (3,3), 0)
        q_bands = self._band_crops_equal(crop, n=5, min_band_h=10)

        # 쿼리 밴드 히스토그램들
        q_hists = []
        for band in q_bands:
            if band is None or band.size == 0:
                q_hists.append(None)
                continue
            hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0,1], None,
                                [self.h_bins, self.s_bins],
                                [0,180, 0,256])
            hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
            q_hists.append(hist)

        # 밴드별 평균 거리
        per_band = []
        for bi in range(5):
            qh = q_hists[bi]
            if qh is None:
                per_band.append(None); continue
            # 이 밴드의 갤러리 히스토그램들 (샘플별 리스트에서 bi번째 모아서)
            bank_hs = [sample[bi] for sample in self.items_band5 if sample[bi] is not None]
            if not bank_hs:
                per_band.append(None); continue
            ds = [cv2.compareHist(qh.reshape(-1,1), gh.reshape(-1,1), cv2.HISTCMP_BHATTACHARYYA)
                for gh in bank_hs]
            per_band.append(float(np.mean(ds)))  # 이 밴드의 평균 거리

        # 유효 밴드만 평균
        valid = [d for d in per_band if d is not None]
        if not valid:
            return None, per_band
        d_mean = float(np.mean(valid))
        return d_mean, per_band

    def _ensure_band5_list(self):
        """5밴드 히스토그램 보관 덱. 항상 5장으로 고정."""
        if not hasattr(self, "items_band5"):
            from collections import deque
            self.items_band5 = deque(maxlen=5)   # ✅ self.items.maxlen 말고, 무조건 5장

    
    
    
    def _ensure_band_lists(self):
        if not hasattr(self, "items_top"):  # 최초 한 번만 생성
            from collections import deque
            self.items_top = deque(maxlen=self.items.maxlen)
            self.items_mid = deque(maxlen=self.items.maxlen)
            self.items_bot = deque(maxlen=self.items.maxlen)

    def add_from_frame_banded(self, frame, box, pad=4, h_bins=25, s_bins=30):
        """밴드별 HS 히스토그램을 갤러리에 저장."""
        self._ensure_band_lists()
        crop = self._crop(frame, box, pad=pad)
        if crop is None:
            return False
        # (선택) 간단 전처리: 살짝 블러만 유지
        crop = cv2.GaussianBlur(crop, (3,3), 0)

        bands = self._band_crops(crop, (0.3, 0.4, 0.3))
        top, mid, bot = bands
        def hist_if(img):
            if img is None: return None
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0,1], None, [h_bins, s_bins], [0,180, 0,256])
            return cv2.normalize(hist, hist).flatten().astype(np.float32)

        h_top = hist_if(top); h_mid = hist_if(mid); h_bot = hist_if(bot)

        # 밝기(too dark/bright) 필터 — V 평균이 너무 낮거나 높으면 제외
        def valid(img):
            if img is None: return False
            v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2].mean()
            return 30 < v < 245

        if h_top is not None and valid(top): self.items_top.append(h_top)
        if h_mid is not None and valid(mid): self.items_mid.append(h_mid)
        if h_bot is not None and valid(bot): self.items_bot.append(h_bot)
        # 최소 하나는 들어가야 성공으로 간주
        return any([h_top is not None, h_mid is not None, h_bot is not None])

    def avg_bhatta_to_box_banded(self, frame, box, pad=4):
        """밴드 가중합 Bhattacharyya 평균(작을수록 유사)."""
        if not hasattr(self, "items_top"):
            return None
        crop = self._crop(frame, box, pad=pad)
        if crop is None:
            return None
        crop = cv2.GaussianBlur(crop, (3,3), 0)
        top, mid, bot = self._band_crops(crop, (0.3, 0.4, 0.3))
        def hist_if(img):
            if img is None: return None
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0,1], None, [self.h_bins, self.s_bins], [0,180, 0,256])
            return cv2.normalize(hist, hist).flatten().astype(np.float32)

        q_top = hist_if(top); q_mid = hist_if(mid); q_bot = hist_if(bot)

        bank_lists = [list(self.items_top), list(self.items_mid), list(self.items_bot)]
        query_list = [q_top, q_mid, q_bot]
        # 상/중/하 가중치 (유리 영향 낮추기 위해 하단↑)
        return self._banded_hist_score(bank_lists, query_list, weights=(0.1, 0.3, 0.6))
    
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
    
    @staticmethod
    def _band_crops(bgr, bands=(0.3, 0.4, 0.3), min_band_h=12):
        """상/중/하 비율로 밴드 잘라 반환 [top, mid, bot]. 너무 얇으면 None."""
        h, w = bgr.shape[:2]
        t, m, b = bands
        h_t = int(round(h * t))
        h_m = int(round(h * m))
        h_b = h - h_t - h_m
        if min(h_t, h_m, h_b) < min_band_h:  # 너무 작은 박스 방지
            return [None, None, None]
        y0 = 0
        top = bgr[y0:y0+h_t, :]
        mid = bgr[y0+h_t:y0+h_t+h_m, :]
        bot = bgr[y0+h_t+h_m:h, :]
        return [top, mid, bot]

    @staticmethod
    def _banded_hist_score(bank_hist_list, query_hist_list, weights=(0.1, 0.3, 0.6)):
        """밴드별 Bhattacharyya 평균(작을수록 유사)을 가중합으로 결합."""
        # bank_hist_list / query_hist_list: [[top_hists...],[mid_hists...],[bot_hists...]]
        def bhatta(a, b):
            return cv2.compareHist(a.reshape(-1,1), b.reshape(-1,1), cv2.HISTCMP_BHATTACHARYYA)

        total = 0.0
        wsum = 0.0
        for i, w in enumerate(weights):
            qh = query_hist_list[i]
            bank = bank_hist_list[i]
            if qh is None or not bank:
                continue
            # 갤러리 각 히스토그램과 비교 후 평균(혹은 최소) — 평균이 안정적
            ds = [bhatta(qh, bh) for bh in bank]
            d_mean = float(np.mean(ds))
            total += w * d_mean
            wsum += w
        if wsum == 0.0:
            return None
        return total / wsum  # 작을수록 유사

