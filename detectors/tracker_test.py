import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import itertools

# 각 트래커별 독립적인 선택된 ID 관리
class MultiTracker:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.selected_id = None  # 인스턴스별 선택된 ID

    def update(self, detections):
        predicted = [track.predict() for track in self.tracks]
        matched, unmatched_dets, unmatched_tracks = self._match(detections, predicted)

        for det_idx, track_idx in matched:
            self.tracks[track_idx].update(detections[det_idx][:4])

        for idx in unmatched_dets:
            self.tracks.append(Track(detections[idx][:4]))

        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        results = []
        for t in self.tracks:
            if t.time_since_update == 0:
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

        # state: [cx, cy, s, r, vx, vy, vs]
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # cx += vx
            [0, 1, 0, 0, 0, 1, 0],  # cy += vy
            [0, 0, 1, 0, 0, 0, 1],  # s  += vs
            [0, 0, 0, 1, 0, 0, 0],  # r  (정적)
            [0, 0, 0, 0, 1, 0, 0],  # vx
            [0, 0, 0, 0, 0, 1, 0],  # vy
            [0, 0, 0, 0, 0, 0, 1],  # vs
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # cx
            [0, 1, 0, 0, 0, 0, 0],  # cy
            [0, 0, 1, 0, 0, 0, 0],  # s
            [0, 0, 0, 1, 0, 0, 0],  # r
        ])

        # ---- R: 관측(디텍션) 노이즈 공분산 -> YOLO를 '더' 믿도록 감소
        # (픽셀 단위/스케일(ratio) 단위가 다르니 항목별로 다르게)
        # cx, cy, s, r 순서
        kf.R = np.diag([0.1, 0.1, 1.0, 1.0]).astype(float)

        # ---- P: 초기 공분산(초기 불확실성). 너무 크면 출렁, 너무 작으면 딱딱
        kf.P = np.diag([50., 50., 400., 10., 50., 50., 100.]).astype(float)

        # ---- Q: 프로세스(모션) 노이즈 공분산 -> 예측을 '유연하게' (차량은 크게)
        # 위치, 크기 변화(속도), 속도 자체에 더 큰 노이즈를 준다
        q_pos = 1.0   # cx, cy
        q_s   = 1.0   # s
        q_r   = 0.1  # r (종횡비는 상대적으로 천천히 변함)
        q_vel = 6.0   # vx, vy
        q_vs  = 3.0   # vs
        kf.Q = np.diag([q_pos, q_pos, q_s, q_r, q_vel, q_vel, q_vs]).astype(float)

        # ---- 초기 상태 셋업
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w  = max(1e-6, (bbox[2] - bbox[0]))
        h  = max(1e-6, (bbox[3] - bbox[1]))
        s  = w * h
        r  = w / h

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