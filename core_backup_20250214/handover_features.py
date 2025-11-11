# handover_features.py
import os, json
import cv2
import numpy as np
from pathlib import Path
from core.config import CAM_STATS_PATH
from collections import defaultdict, deque

# ----------------------------
# A1. 카메라별 러닝 통계 (μ, σ) 저장/로드
# ----------------------------
class CamStats:
    def __init__(self, save_path: str | Path = CAM_STATS_PATH, momentum=0.1):
        self.path = Path(save_path)
        self.momentum = float(momentum)
        self.mu = {}   # cam_id -> np.array(D,)
        self.sigma = {}# cam_id -> np.array(D,)

        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for k,v in data.get("mu", {}).items():
                    self.mu[k] = np.array(v, dtype=np.float32)
                for k,v in data.get("sigma", {}).items():
                    self.sigma[k] = np.array(v, dtype=np.float32)
            except Exception:
                pass

    def _init_if_needed(self, cam_id, vec):
        if cam_id not in self.mu:
            self.mu[cam_id] = vec.astype(np.float32).copy()
            self.sigma[cam_id] = np.ones_like(vec, dtype=np.float32)

    def update(self, cam_id, vec):
        vec = vec.astype(np.float32)
        self._init_if_needed(cam_id, vec)
        # EMA 방식 러닝 통계
        m = self.momentum
        self.mu[cam_id] = (1-m)*self.mu[cam_id] + m*vec
        # 러닝 분산 추정 (간단화)
        delta = vec - self.mu[cam_id]
        self.sigma[cam_id] = (1-m)*self.sigma[cam_id] + m*np.abs(delta)

    def whiten(self, cam_id, vec):
        vec = vec.astype(np.float32)
        self._init_if_needed(cam_id, vec)
        return (vec - self.mu[cam_id]) / (self.sigma[cam_id] + 1e-6)

    def save(self):
        data = {
            "mu":    {k: v.tolist() for k,v in self.mu.items()},
            "sigma": {k: v.tolist() for k,v in self.sigma.items()},
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------
# A2. 저비용 특징 추출
#   - HSV(H,S), LAB(a,b) hist
#   - (옵션) Hu moments 약간
# ----------------------------
def _crop_safe(img, xyxy):
    h, w = img.shape[:2]
    x1,y1,x2,y2 = map(int, xyxy)
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1+1 or y2 <= y1+1:
        return None
    return img[y1:y2, x1:x2]

def _hist_1ch(ch, bins=32):
    hist = cv2.calcHist([ch],[0],None,[bins],[0,256]).ravel()
    hist = hist / (hist.sum()+1e-6)
    return hist.astype(np.float32)

def extract_color_feature(img_bgr, bins=32, use_clahe=True):
    # 조명 내성 보강 (V 채널 CLAHE 권장)
    if use_clahe:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v = clahe.apply(v)
        hsv = cv2.merge([h,s,v])
        img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    h,s,_ = cv2.split(hsv)
    _,a,b = cv2.split(lab)

    h_hist = _hist_1ch(h, bins)
    s_hist = _hist_1ch(s, bins)
    a_hist = _hist_1ch(a, bins)
    b_hist = _hist_1ch(b, bins)
    return np.concatenate([h_hist, s_hist, a_hist, b_hist], axis=0)  # 4*bins

def extract_shape_feature(img_bgr, take_hu=3):
    # 간단 이진화 후 Hu moments
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 1.0)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    hu = cv2.HuMoments(cv2.moments(thr)).flatten()
    # log scale 안정화
    hu = -np.sign(hu)*np.log10(np.abs(hu) + 1e-9)
    hu = hu[:take_hu].astype(np.float32)  # 2~3D만 사용 권장
    # 비율 한 개 추가
    h, w = img_bgr.shape[:2]
    ratio = np.array([w/(h+1e-6)], dtype=np.float32)
    return np.concatenate([hu, ratio], axis=0)  # (take_hu + 1,)

def build_frame_feature(img_bgr, bins=32, with_shape=True):
    col = extract_color_feature(img_bgr, bins=bins, use_clahe=True)       # 4*bins
    if with_shape:
        shp = extract_shape_feature(img_bgr, take_hu=3)                   # 4D
        return np.concatenate([col, shp], axis=0)                          # 4*bins + 4
    return col


# ----------------------------
# A3. 트랙 → 핑거프린트 (중앙값 풀링 + 카메라 표준화)
# ----------------------------
def build_fingerprint(frames, bboxes, cam_id, cam_stats: CamStats,
                      bins=32, with_shape=True, use_whitening=True):
    """
    frames: List[np.ndarray HxWx3 BGR]  (N프레임)
    bboxes: List[tuple(x1,y1,x2,y2)]    (frames와 동일 길이)
    """
    feats = []
    for img, xyxy in zip(frames, bboxes):
        crop = _crop_safe(img, xyxy)
        if crop is None: 
            continue
        feats.append(build_frame_feature(crop, bins=bins, with_shape=with_shape))
    if not feats:
        return None

    feats = np.stack(feats, axis=0)           # [T, D]
    fp_raw = np.median(feats, axis=0).astype(np.float32)   # 중앙값 풀링

    # 러닝 통계 업데이트(원하면 exit/entry 시점에만 업데이트)
    cam_stats.update(cam_id, fp_raw)

    fp = cam_stats.whiten(cam_id, fp_raw) if use_whitening else fp_raw
    return fp, fp_raw


# ----------------------------
# A4. 유사도 & 이중 게이트
# ----------------------------
def cosine_sim(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6: return 0.0
    return float(np.dot(a,b) / (na*nb))

def bhattach_dist(p, q):
    # hist-like 부분에 더 적합하지만 여기서는 전체 벡터에도 사용 가능
    # (음수 없도록 ReLU; 안정화)
    p = np.maximum(0, p); q = np.maximum(0, q)
    p = p / (p.sum()+1e-6); q = q / (q.sum()+1e-6)
    bc = np.sum(np.sqrt(p*q))
    return float(np.clip(np.sqrt(max(0.0, 1.0 - bc)), 0.0, 1.0))

def compare_fingerprints(fp_src, fp_tgt, cos_th=0.85, bh_th=0.25, alpha=0.6):
    cos = cosine_sim(fp_src, fp_tgt)
    bh  = bhattach_dist(fp_src, fp_tgt)
    # 종합 점수(참고) 및 이중게이트 판정
    score = alpha*cos + (1-alpha)*(1.0 - bh)
    ok = (cos >= cos_th) or (bh <= bh_th)
    return {"cos":cos, "bh":bh, "score":score, "ok":ok}
