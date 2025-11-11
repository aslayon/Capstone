# core/reid_bank.py (화이트닝 강화 버전)
from collections import deque
import cv2, glob, os
import numpy as np
from typing import List, Optional
from core.cam_stats import _CAMSTATS, build_fp, _extract_color_feat, _extract_shape_feat

def _pre_norm_color(bgr):
    """Gray-world white balance + 적응형 감마 보정"""
    # Gray-world white balance
    b, g, r = cv2.split(bgr.astype(np.float32))
    mean = (b.mean() + g.mean() + r.mean()) / 3.0 + 1e-6
    b *= mean / (b.mean() + 1e-6)
    g *= mean / (g.mean() + 1e-6)
    r *= mean / (r.mean() + 1e-6)
    wb = np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)
    
    # 적응형 감마 보정 (어두우면 밝게)
    v = cv2.cvtColor(wb, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
    gamma = 0.9 if v > 110 else (0.75 if v > 80 else 0.65)
    lut = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], np.uint8)
    return cv2.LUT(wb, lut)


def _cam_normalize_color(bgr, cam_id):
    """
    ✅ 카메라별 통계 기반 색상 정규화 (화이트닝)
    _CAMSTATS를 활용하여 카메라 간 색상 차이를 보정
    """
    if cam_id is None:
        return bgr
    
    # BGR 각 채널을 평탄화하여 벡터로
    b, g, r = cv2.split(bgr.astype(np.float32))
    h, w = bgr.shape[:2]
    
    # 채널별 평균/표준편차 계산
    vec = np.array([b.mean(), g.mean(), r.mean()], dtype=np.float32)
    
    # _CAMSTATS로 화이트닝
    try:
        normed_vec = _CAMSTATS.whiten(cam_id, vec)
        
        # 정규화된 값을 다시 이미지에 적용 (단순 스케일링)
        if normed_vec is not None and len(normed_vec) == 3:
            # 원본 대비 정규화 비율 계산
            scale_b = normed_vec[0] / (vec[0] + 1e-6)
            scale_g = normed_vec[1] / (vec[1] + 1e-6)
            scale_r = normed_vec[2] / (vec[2] + 1e-6)
            
            # 각 채널에 스케일 적용 (과도한 변화 방지)
            scale_b = np.clip(scale_b, 0.5, 2.0)
            scale_g = np.clip(scale_g, 0.5, 2.0)
            scale_r = np.clip(scale_r, 0.5, 2.0)
            
            b = np.clip(b * scale_b, 0, 255)
            g = np.clip(g * scale_g, 0, 255)
            r = np.clip(r * scale_r, 0, 255)
            
            return cv2.merge([b, g, r]).astype(np.uint8)
    except Exception as e:
        pass  # 실패 시 원본 반환
    
    return bgr


def _advanced_pre_norm_color(bgr, cam_id=None):
    """
    ✅ 강화된 색상 정규화: Gray-world + 카메라별 화이트닝 + 감마 보정
    """
    # 1) Gray-world white balance
    b, g, r = cv2.split(bgr.astype(np.float32))
    mean = (b.mean() + g.mean() + r.mean()) / 3.0 + 1e-6
    b *= mean / (b.mean() + 1e-6)
    g *= mean / (g.mean() + 1e-6)
    r *= mean / (r.mean() + 1e-6)
    wb = np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)
    
    # 2) 카메라별 화이트닝 (선택적)
    if cam_id is not None:
        wb = _cam_normalize_color(wb, cam_id)
    
    # 3) 적응형 감마 보정
    v = cv2.cvtColor(wb, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
    gamma = 0.9 if v > 110 else (0.75 if v > 80 else 0.65)
    lut = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], np.uint8)
    
    return cv2.LUT(wb, lut)


class ReIDBank:
    """선택 차량의 최근 히스토그램 특징을 보관/비교 (HSV HS 2D hist)"""
    def __init__(self, maxlen=10, h_bins=25, s_bins=30):
        self.items = deque(maxlen=maxlen)
        self.h_bins = h_bins
        self.s_bins = s_bins

        
        self.items_band5 = deque(maxlen=5) 
    @staticmethod
    def _crop(frame, box, pad=4):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _crop_center(self, frame, box, pad=4, center_ratio=0.75):
        """
        bbox에서 중앙 영역만 추출 (배경 제거)
        
        Args:
            frame: 입력 프레임
            box: (x1, y1, x2, y2)
            pad: 패딩
            center_ratio: 중앙 영역 비율 (0.75 = 75% 영역만 사용)
                        1.0 = 전체 bbox 사용
                        0.6 = 중앙 60%만 사용
        
        Returns:
            중앙 영역 crop
        """
        x1, y1, x2, y2 = map(int, box[:4])
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame.shape[1], x2 + pad)
        y2 = min(frame.shape[0], y2 + pad)
        
        # ✅ 중앙 영역만 추출
        w = x2 - x1
        h = y2 - y1
        
        # 중앙 영역 계산
        margin_w = int(w * (1 - center_ratio) / 2)
        margin_h = int(h * (1 - center_ratio) / 2)
        
        cx1 = x1 + margin_w
        cy1 = y1 + margin_h
        cx2 = x2 - margin_w
        cy2 = y2 - margin_h
        
        # 최소 크기 보장
        if cx2 - cx1 < 20 or cy2 - cy1 < 20:
            # 너무 작으면 전체 bbox 사용
            return frame[y1:y2, x1:x2]
        
        return frame[cy1:cy2, cx1:cx2]



    def _hist_hs(self, bgr):
        bgr = cv2.GaussianBlur(bgr, (3, 3), 0)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H, W = hsv.shape[:2]
        mask = np.zeros((H, W), np.uint8)
        cx, cy = W // 2, H // 2
        cv2.ellipse(mask, (cx, cy), (W // 3, H // 3), 0, 0, 360, 255, -1)
        hist = cv2.calcHist([hsv], [0, 1], mask,
                            [self.h_bins, self.s_bins],
                            [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

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
        if crop is None:
            return None if also_return_hist else False
        h = self._hist_hs(crop)
        self.items.append(h)
        return h if also_return_hist else True

    def size(self) -> int:
        return len(self.items)

    def clear(self):
        if hasattr(self, "items_band5"):
            self.items_band5.clear()
        self.items.clear()

    def add_from_frame_until_full(self, frame, box) -> bool:
        if self.size() >= self.items.maxlen:
            return False
        return self.add_from_frame(frame, box)

    def _hist_from_frame(self, frame, box) -> Optional[np.ndarray]:
        crop = self._crop(frame, box)
        if crop is None:
            return None
        return self._hist_hs(crop)

    def avg_bhatta_to_hist(self, h: np.ndarray) -> Optional[float]:
        if not self.items:
            return None
        dists = [cv2.compareHist(h.reshape(-1, 1), g.reshape(-1, 1), cv2.HISTCMP_BHATTACHARYYA)
                 for g in self.items]
        return float(np.mean(dists))

    def avg_bhatta_to_box(self, frame, box) -> Optional[float]:
        h = self._hist_from_frame(frame, box)
        if h is None:
            return None
        return self.avg_bhatta_to_hist(h)

    def is_same_vehicle(self, frame, box, thresh: float) -> bool:
        m = self.avg_bhatta_to_box(frame, box)
        return (m is not None) and (m <= thresh)

    def load_from_paths(self, paths: List[str], limit: int = None) -> int:
        n = 0
        for p in (paths if limit is None else paths[:limit]):
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
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
        return self.load_from_paths(files, limit=min(5, self.items.maxlen))

    def _ensure_band5_list(self):
        """5밴드 히스토그램 보관 덱. 항상 5장으로 고정."""
        if not hasattr(self, "items_band5"):
            self.items_band5 = deque(maxlen=5)

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
            y2 = h if i == n - 1 else (y + band_h)
            bands.append(bgr[y:y2, :])
            y = y2
        return bands

    def add_from_frame_banded5(self, frame, box, pad=4, origin_seg=None, origin_cam=None, cam_id=None, use_whitening=True):
        """
        ✅ 화이트닝 강화 버전
        bbox를 5밴드로 나눠 HS hist 저장. 한 샘플 = [h0..h4].
        
        Args:
            cam_id: 카메라 ID (화이트닝용, origin_cam과 동일하게 사용 가능)
            use_whitening: True면 카메라별 화이트닝 적용
        """
        try:
            self.set_origin(origin_seg, origin_cam)
        except Exception:
            pass
        
        self._ensure_band5_list()
        
        # 이미 꽉 찼으면 즉시 반환
        if len(self.items_band5) >= self.items_band5.maxlen:
            return False

        crop = self._crop(frame, box, pad=pad)
        if crop is None:
            return False
        
        # ✅ 강화된 색상 정규화 적용
        if use_whitening:
            # cam_id가 없으면 origin_cam 사용
            effective_cam_id = cam_id if cam_id is not None else origin_cam
            crop = _advanced_pre_norm_color(crop, cam_id=effective_cam_id)
        else:
            # 기존 방식
            crop = _pre_norm_color(crop)
        
        crop = cv2.GaussianBlur(crop, (3, 3), 0)

        bands = self._band_crops_equal(crop, n=5, min_band_h=10)
        hists = []
        for band in bands:
            if band is None or band.size == 0:
                hists.append(None)
                continue
            hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None,
                                [self.h_bins, self.s_bins],
                                [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
            hists.append(hist)

        if not any(h is not None for h in hists):
            return False

        self.items_band5.append(hists)
        
        # ✅ 카메라 통계 업데이트 (선택적)
        if use_whitening and effective_cam_id is not None:
            try:
                # 전체 crop의 색상 통계를 _CAMSTATS에 업데이트
                vec = np.array([
                    crop[:, :, 0].mean(),  # B
                    crop[:, :, 1].mean(),  # G
                    crop[:, :, 2].mean()   # R
                ], dtype=np.float32)
                _CAMSTATS.update(effective_cam_id, vec)
            except Exception:
                pass
        
        return True

    def add_from_frame_banded5_improved(self, frame, box, pad=4, center_ratio=0.75, 
                                    origin_seg=None, origin_cam=None, cam_id=None, 
                                    use_whitening=True):
        """
        ✅ 배경 제거 버전
        
        Args:
            center_ratio: 0.6~1.0 사이 값
                        0.75 = 중앙 75% 영역만 사용 (권장)
                        0.6 = 중앙 60% (배경 최대 제거)
                        1.0 = 전체 bbox (기존 방식)
        """
        try:
            self.set_origin(origin_seg, origin_cam)
        except Exception:
            pass
        
        self._ensure_band5_list()
        
        if len(self.items_band5) >= self.items_band5.maxlen:
            return False

        # ✅ 중앙 영역만 crop
        crop = self._crop_center(frame, box, pad=pad, center_ratio=center_ratio)
        if crop is None or crop.size == 0:
            return False
        
        # 나머지는 기존과 동일
        if use_whitening:
            effective_cam_id = cam_id if cam_id is not None else origin_cam
            crop = _advanced_pre_norm_color(crop, cam_id=effective_cam_id)
        else:
            crop = _pre_norm_color(crop)
        
        crop = cv2.GaussianBlur(crop, (3, 3), 0)

        bands = self._band_crops_equal(crop, n=5, min_band_h=10)
        hists = []
        for band in bands:
            if band is None or band.size == 0:
                hists.append(None)
                continue
            hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None,
                                [self.h_bins, self.s_bins],
                                [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
            hists.append(hist)

        if not any(h is not None for h in hists):
            return False

        self.items_band5.append(hists)
        return True


    def size5(self) -> int:
        """5밴드 갤러리 샘플 개수"""
        return len(getattr(self, "items_band5", []))

    def avg_bhatta_to_box_banded5(self, frame, box, pad=4, cam_id=None, use_whitening=True):
        """
        ✅ 화이트닝 강화 버전
        쿼리 bbox를 5밴드로 나눠 갤러리(5밴드)와 밴드별 Bhattacharyya 평균을 계산.
        
        반환: (d_mean, [d0,d1,d2,d3,d4])
              d는 0~1, 작을수록 유사. 유효밴드만 평균.
        """
        if not hasattr(self, "items_band5") or len(self.items_band5) == 0:
            return None, None

        crop = self._crop(frame, box, pad=pad)
        if crop is None:
            return None, None
        
        # ✅ 강화된 색상 정규화 적용
        if use_whitening:
            # cam_id가 없으면 origin_cam 사용
            effective_cam_id = cam_id if cam_id is not None else getattr(self, 'origin_cam', None)
            crop = _advanced_pre_norm_color(crop, cam_id=effective_cam_id)
        else:
            crop = _pre_norm_color(crop)
        
        crop = cv2.GaussianBlur(crop, (3, 3), 0)
        q_bands = self._band_crops_equal(crop, n=5, min_band_h=10)

        # 쿼리 밴드 히스토그램들
        q_hists = []
        for band in q_bands:
            if band is None or band.size == 0:
                q_hists.append(None)
                continue
            hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None,
                                [self.h_bins, self.s_bins],
                                [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
            q_hists.append(hist)

        # 밴드별 평균 거리
        per_band = []
        for bi in range(5):
            qh = q_hists[bi]
            if qh is None:
                per_band.append(None)
                continue
            bank_hs = [sample[bi] for sample in self.items_band5 if sample[bi] is not None]
            if not bank_hs:
                per_band.append(None)
                continue
            ds = [cv2.compareHist(qh.reshape(-1, 1), gh.reshape(-1, 1), cv2.HISTCMP_BHATTACHARYYA)
                  for gh in bank_hs]
            per_band.append(float(np.mean(ds)))

        # 유효 밴드만 평균
        valid = [d for d in per_band if d is not None]
        if not valid:
            return None, per_band
        d_mean = float(np.mean(valid))
        return d_mean, per_band

    def avg_bhatta_to_box_banded5_improved(self, frame, box, pad=4, center_ratio=0.75,
                                        cam_id=None, use_whitening=True):
        """
        ✅ 배경 제거 버전
        """
        if not hasattr(self, 'items_band5') or len(self.items_band5) == 0:
            return None, None

        # ✅ 중앙 영역만 crop
        crop = self._crop_center(frame, box, pad=pad, center_ratio=center_ratio)
        if crop is None or crop.size == 0:
            return None, None
        
        # 나머지는 기존과 동일
        if use_whitening:
            effective_cam_id = cam_id if cam_id is not None else getattr(self, 'origin_cam', None)
            crop = _advanced_pre_norm_color(crop, cam_id=effective_cam_id)
        else:
            crop = _pre_norm_color(crop)
        
        crop = cv2.GaussianBlur(crop, (3, 3), 0)

        bands = self._band_crops_equal(crop, n=5, min_band_h=10)
        curr_hists = []
        for band in bands:
            if band is None or band.size == 0:
                curr_hists.append(None)
                continue
            hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None,
                                [self.h_bins, self.s_bins],
                                [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
            curr_hists.append(hist)

        if not any(h is not None for h in curr_hists):
            return None, None

        # 갤러리와 비교
        dists = []
        for gallery_item in self.items_band5:
            band_dists = []
            for i in range(5):
                gh = gallery_item[i]
                ch = curr_hists[i]
                if gh is None or ch is None:
                    continue
                d = _bhattacharyya_distance(gh, ch)
                band_dists.append(d)
            
            if band_dists:
                dists.append(np.mean(band_dists))
        
        if not dists:
            return None, None
        
        return np.mean(dists), np.std(dists)


    # 기존 3밴드 메서드들 (호환성 유지)
    def _ensure_band_lists(self):
        if not hasattr(self, "items_top"):
            self.items_top = deque(maxlen=self.items.maxlen)
            self.items_mid = deque(maxlen=self.items.maxlen)
            self.items_bot = deque(maxlen=self.items.maxlen)

    def add_from_frame_banded(self, frame, box, pad=4, h_bins=25, s_bins=30):
        """밴드별 HS 히스토그램을 갤러리에 저장."""
        self._ensure_band_lists()
        crop = self._crop(frame, box, pad=pad)
        if crop is None:
            return False
        crop = cv2.GaussianBlur(crop, (3, 3), 0)

        bands = self._band_crops(crop, (0.3, 0.4, 0.3))
        top, mid, bot = bands

        def hist_if(img):
            if img is None:
                return None
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
            return cv2.normalize(hist, hist).flatten().astype(np.float32)

        h_top = hist_if(top)
        h_mid = hist_if(mid)
        h_bot = hist_if(bot)

        def valid(img):
            if img is None:
                return False
            v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
            return 30 < v < 245

        if h_top is not None and valid(top):
            self.items_top.append(h_top)
        if h_mid is not None and valid(mid):
            self.items_mid.append(h_mid)
        if h_bot is not None and valid(bot):
            self.items_bot.append(h_bot)
        return any([h_top is not None, h_mid is not None, h_bot is not None])

    def avg_bhatta_to_box_banded(self, frame, box, pad=4):
        """밴드 가중합 Bhattacharyya 평균(작을수록 유사)."""
        if not hasattr(self, "items_top"):
            return None
        crop = self._crop(frame, box, pad=pad)
        if crop is None:
            return None
        crop = cv2.GaussianBlur(crop, (3, 3), 0)
        top, mid, bot = self._band_crops(crop, (0.3, 0.4, 0.3))

        def hist_if(img):
            if img is None:
                return None
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [self.h_bins, self.s_bins], [0, 180, 0, 256])
            return cv2.normalize(hist, hist).flatten().astype(np.float32)

        q_top = hist_if(top)
        q_mid = hist_if(mid)
        q_bot = hist_if(bot)

        bank_lists = [list(self.items_top), list(self.items_mid), list(self.items_bot)]
        query_list = [q_top, q_mid, q_bot]
        return self._banded_hist_score(bank_lists, query_list, weights=(0.1, 0.3, 0.6))

    @staticmethod
    def bhattacharyya(h1, h2):
        h1 = h1.reshape(-1, 1).astype(np.float32)
        h2 = h2.reshape(-1, 1).astype(np.float32)
        return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    def score_to_gallery(self, frame, box):
        if not self.items:
            return None
        crop = self._crop(frame, box)
        if crop is None:
            return None
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
        if min(h_t, h_m, h_b) < min_band_h:
            return [None, None, None]
        y0 = 0
        top = bgr[y0:y0 + h_t, :]
        mid = bgr[y0 + h_t:y0 + h_t + h_m, :]
        bot = bgr[y0 + h_t + h_m:h, :]
        return [top, mid, bot]

    @staticmethod
    def _banded_hist_score(bank_hist_list, query_hist_list, weights=(0.1, 0.3, 0.6)):
        """밴드별 Bhattacharyya 평균(작을수록 유사)을 가중합으로 결합."""

        def bhatta(a, b):
            return cv2.compareHist(a.reshape(-1, 1), b.reshape(-1, 1), cv2.HISTCMP_BHATTACHARYYA)

        total = 0.0
        wsum = 0.0
        for i, w in enumerate(weights):
            qh = query_hist_list[i]
            bank = bank_hist_list[i]
            if qh is None or not bank:
                continue
            ds = [bhatta(qh, bh) for bh in bank]
            d_mean = float(np.mean(ds))
            total += w * d_mean
            wsum += w
        if wsum == 0.0:
            return None
        return total / wsum
    
    
def _bhattacharyya_distance(h1, h2):
    """바타차야 거리 계산"""
    h1 = h1 / (h1.sum() + 1e-10)
    h2 = h2 / (h2.sum() + 1e-10)
    bc = np.sum(np.sqrt(h1 * h2))
    return np.sqrt(max(0.0, 1.0 - bc))