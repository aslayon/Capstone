import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import itertools
import os, json
from pathlib import Path

def _load_ultralytics_bytetrack_defaults():
    """
    1) 로컬/패키지의 bytetrack.yaml 찾아 로드
    2) 없으면 Ultralytics 퍼블릭 기본값으로 폴백
       (track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6,
        track_buffer=30, match_thresh=0.8)
    """
    import yaml

    # 후보 경로들: 프로젝트 루트/현재 작업 디렉토리/ultralytics 설치 경로
    candidates = [
        Path("bytetrack.yaml"),
        Path("ultralytics/cfg/trackers/bytetrack.yaml"),
        Path(os.path.dirname(__file__)) / "bytetrack.yaml",
    ]
    # site-packages 내 ultralytics 경로도 시도
    try:
        import ultralytics, inspect
        upath = Path(inspect.getfile(ultralytics)).parent / "cfg" / "trackers" / "bytetrack.yaml"
        candidates.append(upath)
    except Exception:
        pass

    for p in candidates:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                # 키 매핑(필요한 것만 추출)
                return {
                    "track_high_thresh": float(data.get("track_high_thresh", 0.5)),
                    "track_low_thresh":  float(data.get("track_low_thresh", 0.1)),
                    "new_track_thresh":  float(data.get("new_track_thresh", 0.6)),
                    "track_buffer":      int(data.get("track_buffer", 30)),
                    "match_thresh":      float(data.get("match_thresh", 0.8)),
                }
            except Exception:
                continue

    # 폴백: Ultralytics 공개 기본값
    return {
        "track_high_thresh": 0.5,
        "track_low_thresh":  0.1,
        "new_track_thresh":  0.6,
        "track_buffer":      30,
        "match_thresh":      0.8,
    }


# 각 트래커별 독립적인 선택된 ID 관리
class MultiTracker:
    def __init__(self, max_age=30, iou_threshold=0.3, min_hits=1, show_newborn=True,
                track_high_thresh=None, track_low_thresh=None, new_track_thresh=None,
                match_thresh=None, track_buffer=None):

        # 퍼블릭 디폴트 로드
        defaults = _load_ultralytics_bytetrack_defaults()

        self.track_high_thresh = float(track_high_thresh if track_high_thresh is not None else defaults["track_high_thresh"])
        self.track_low_thresh  = float(track_low_thresh  if track_low_thresh  is not None else defaults["track_low_thresh"])
        self.new_track_thresh  = float(new_track_thresh  if new_track_thresh  is not None else defaults["new_track_thresh"])
        self.track_buffer      = int  (track_buffer      if track_buffer      is not None else defaults["track_buffer"])
        self.match_thresh      = float(match_thresh      if match_thresh      is not None else defaults["match_thresh"])

        # 나머지 기존 초기화 그대로...
        self.max_age = int(max_age)
        self.iou_threshold = float(iou_threshold)
        self.min_hits = int(min_hits)
        self.show_newborn = bool(show_newborn)
        self.tracks = []
        self._last_results = []
        self._last_tlbr_by_id = {}
        self.selected_id = None
        self._frame_id = 0

    def update(self, detections):
        self.newborn = {}
        predicted = [track.predict() for track in self.tracks]
        matched, unmatched_dets, unmatched_tracks = self._match(detections, predicted)

        # 매칭된 트랙 업데이트
        for det_idx, track_idx in matched:
            self.tracks[track_idx].update(detections[det_idx][:4])

        # 매칭 안 된 det는 새 트랙으로 즉시 생성
        for idx in unmatched_dets:
            t = Track(detections[idx][:4])
            self.tracks.append(t)
            if self.show_newborn:
                x1,y1,x2,y2 = t.get_bbox()
                self.newborn[t.id] = (x1,y1,x2,y2)  # 같은 프레임에 바로 그림

        # 오래된 트랙 정리
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        # 결과 구성: min_hits 충족 or (show_newborn이고 hits==1인 신생 트랙)
        results = []
        for t in self.tracks:
            if t.time_since_update == 0:
                if t.hits >= self.min_hits or (self.show_newborn and t.hits == 1):
                    x1, y1, x2, y2 = t.get_bbox()
                    results.append((t.id, x1, y1, x2, y2))

        return results

    def predict_only(self):
        """탐지 없이 예측만 수행 (부드러운 움직임을 위해)"""
        for track in self.tracks:
            track.predict()
        
        # 너무 오래된 트랙 제거
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        
        # 모든 트랙의 현재 위치 반환
        results = []
        for t in self.tracks:
            x1, y1, x2, y2 = t.get_bbox()
            results.append((t.id, x1, y1, x2, y2))
        
        return results

    def _match(self, detections, predicted):
        if len(predicted) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(predicted)))

        iou_matrix = np.zeros((len(detections), len(predicted)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, pred in enumerate(predicted):
                iou_matrix[d, t] = self._iou(det[:4], self._convert_to_bbox(pred))

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = list(zip(*matched_indices))

        unmatched_dets = list(set(range(len(detections))) - {m[0] for m in matched_indices})
        unmatched_tracks = list(set(range(len(predicted))) - {m[1] for m in matched_indices})

        matches = [m for m in matched_indices if iou_matrix[m[0], m[1]] >= self.iou_threshold]
        return matches, unmatched_dets, unmatched_tracks

    def _convert_to_bbox(self, pred):
        cx, cy, s, r = pred
        val = max(0.0, s * r)        # ✅ 음수 방지
        w = max(1e-6, np.sqrt(val))  # sqrt 전에 clamp
        h = max(1e-6, s / w if w > 1e-12 else 1e-6)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return (x1, y1, x2, y2)


    def _iou(self, bb_test, bb_gt):
        xx1 = max(bb_test[0], bb_gt[0])
        yy1 = max(bb_test[1], bb_gt[1])
        xx2 = min(bb_test[2], bb_gt[2])
        yy2 = min(bb_test[3], bb_gt[3])
        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        inter = w * h
        area1 = max(1e-6, (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]))
        area2 = max(1e-6, (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]))
        return inter / (area1 + area2 - inter + 1e-6)

    def select_track_by_point(self, x, y):
        """특정 좌표의 차량을 선택/해제"""
        if x < 0 or y < 0:  # 강제 해제
            if self.selected_id is not None:
                print(f"[INFO] 차량 선택 해제됨 (이전 ID: {self.selected_id})")
                self.selected_id = None
            return
            
        for track in self.tracks:
            if track.contains_point(x, y):
                if self.selected_id == track.id:
                    self.selected_id = None
                    print(f"[INFO] 관심 차량 해제됨 (ID: {track.id})")
                else:
                    self.selected_id = track.id
                    print(f"[INFO] 관심 차량 선택됨 (ID: {track.id})")
                return
        
        # 클릭한 위치에 차량이 없으면 선택 해제
        if self.selected_id is not None:
            print("[INFO] 빈 공간 클릭으로 차량 선택 해제됨")
            self.selected_id = None
        
    def get_selected_bbox(self):
        """선택된 차량의 bbox 반환"""
        for track in self.tracks:
            if track.id == self.selected_id:
                return track.get_bbox()
        return None



class Track:
    _id_iter = itertools.count()

    def __init__(self, bbox):
        self.id = next(self._id_iter)
        self.kf = self._init_kalman_filter(bbox)
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 0
        self.age = 0
        self.bbox = bbox

    def _init_kalman_filter(self, bbox):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        kf.R = np.diag([0.1, 0.1, 2.0, 1.0])     # :contentReference[oaicite:4]{index=4} 의 튜닝 참고
        # 초기 공분산: 속도항/크기항을 과도하게 크게 두지 않기
        kf.P = np.diag([50., 50., 200., 100., 100., 100., 100.])
        # 프로세스 노이즈: 약간 키워 초기에 빨리 따라붙게
        kf.Q = np.diag([1.0, 1.0, 5.0, 2.0, 3.0, 3.0, 2.0])

        # === 초기 상태 설정 ===
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = max(1e-6, bbox[2] - bbox[0])
        h = max(1e-6, bbox[3] - bbox[1])
        s = w * h
        r = w / h
        kf.x[:4] = np.array([[cx], [cy], [s], [r]])
        return kf



    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:4].flatten()

    def update(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        self.kf.update(np.array([cx, cy, s, r]))
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.bbox = bbox

    def get_bbox(self):
        cx, cy, s, r = self.kf.x[:4].flatten()
        val = max(0.0, s * r)        # ✅ 음수 방지
        w = max(1e-6, np.sqrt(val))
        h = max(1e-6, s / w if w > 1e-12 else 1e-6)
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return (x1, y1, x2, y2)


    def contains_point(self, x, y):
        x1, y1, x2, y2 = self.get_bbox()
        return x1 <= x <= x2 and y1 <= y <= y2


def check_boundary_event(bbox, frame_width, frame_height, margin=0.05):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if (cx < frame_width * margin or cx > frame_width * (1 - margin) or
        cy < frame_height * margin or cy > frame_height * (1 - margin)):
        return True
    return False