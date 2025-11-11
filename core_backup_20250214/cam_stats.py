# ==== Lightweight handover features & camera-wise whitening ====
import json
from pathlib import Path
from core.config import CAM_STATS_PATH
import numpy as np
import cv2

class _CamStats:
    def __init__(self, save_path: str | Path = CAM_STATS_PATH, momentum=0.1):
        self.path = Path(save_path)
        self.m = float(momentum)
        self.mu, self.sigma = {}, {}
        if self.path.is_file():
            try:
                j = json.loads(self.path.read_text(encoding="utf-8"))
                self.mu    = {k: np.array(v, np.float32) for k,v in j.get("mu", {}).items()}
                self.sigma = {k: np.array(v, np.float32) for k,v in j.get("sigma", {}).items()}
            except Exception:
                pass
    def _init(self, cam_id, vec):
        if cam_id not in self.mu:
            self.mu[cam_id]    = vec.astype(np.float32).copy()
            self.sigma[cam_id] = np.ones_like(vec, np.float32)
    def update(self, cam_id, vec):
        v = vec.astype(np.float32); self._init(cam_id, v)
        m = self.m
        self.mu[cam_id]    = (1-m)*self.mu[cam_id] + m*v
        self.sigma[cam_id] = (1-m)*self.sigma[cam_id] + m*np.abs(v - self.mu[cam_id])
    def whiten(self, cam_id, vec):
        v = vec.astype(np.float32); self._init(cam_id, v)
        return (v - self.mu[cam_id]) / (self.sigma[cam_id] + 1e-6)
    def save(self):
        j = {
            "mu": {k: v.tolist() for k,v in self.mu.items()},
            "sigma": {k: v.tolist() for k,v in self.sigma.items()},
        }
        self.path.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")




_CAMSTATS = _CamStats(CAM_STATS_PATH, momentum=0.1)

def _crop_safe(img, xyxy):
    h, w = img.shape[:2]; x1,y1,x2,y2 = map(int, xyxy)
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1+1 or y2 <= y1+1: return None
    return img[y1:y2, x1:x2]

def _hist1(ch, bins=32):
    h = cv2.calcHist([ch],[0],None,[bins],[0,256]).ravel()
    h = h / (h.sum()+1e-6)
    return h.astype(np.float32)

def _extract_color_feat(bgr, bins=32):
    # 조명 내성: V만 CLAHE
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    v = cv2.createCLAHE(2.0,(8,8)).apply(v)
    hsv = cv2.merge([h,s,v]); bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    h,s,_ = cv2.split(hsv); _,a,b = cv2.split(lab)
    return np.concatenate([_hist1(h,bins), _hist1(s,bins), _hist1(a,bins), _hist1(b,bins)], 0)  # 4*bins

def _extract_shape_feat(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g,(3,3),1.0)
    thr = cv2.threshold(g,0,255,cv2.THRESH_OTSU)[1]
    hu = cv2.HuMoments(cv2.moments(thr)).flatten()
    hu = -np.sign(hu)*np.log10(np.abs(hu)+1e-9)
    h,w = bgr.shape[:2]; ratio = np.array([w/(h+1e-6)], np.float32)
    return np.concatenate([hu[:3].astype(np.float32), ratio], 0)   # 4D

def build_fp(img, box, cam_id, bins=32, with_shape=True, do_whiten=True):
    crop = _crop_safe(img, box)
    if crop is None: return None, None
    col = _extract_color_feat(crop, bins=bins)              # 4*bins
    feat = col if not with_shape else np.concatenate([col, _extract_shape_feat(crop)], 0)  # 4*bins + 4
    _CAMSTATS.update(cam_id, feat)
    fp = _CAMSTATS.whiten(cam_id, feat) if do_whiten else feat
    return fp, feat

def _cos(a,b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na<1e-6 or nb<1e-6: return 0.0
    return float(np.dot(a,b)/(na*nb))

def _bhattacharyya(p, q):
    p = np.maximum(0,p); q = np.maximum(0,q)
    p = p/(p.sum()+1e-6); q = q/(q.sum()+1e-6)
    bc = np.sum(np.sqrt(p*q))
    return float(np.clip(np.sqrt(max(0.0, 1.0-bc)), 0.0, 1.0))   # 0~1, 작을수록 유사
