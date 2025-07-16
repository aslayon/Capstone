import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import itertools

selected_id = None  # 사용자 선택 ID (외부에서 접근 가능)


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
        kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]])
        kf.R *= 10.
        kf.P *= 500.
        kf.Q *= 0.01
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
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
        w = max(1e-6, np.sqrt(s * r))
        h = max(1e-6, s / w)
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return (x1, y1, x2, y2)

    def contains_point(self, x, y):
        x1, y1, x2, y2 = self.get_bbox()
        return x1 <= x <= x2 and y1 <= y <= y2


class MultiTracker:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []

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
                if selected_id is None or t.id == selected_id:
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
        w = max(1e-6, np.sqrt(s * r))
        h = max(1e-6, s / w)
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
        global selected_id
        for track in self.tracks:
            if track.contains_point(x, y):
                selected_id = track.id
                print(f"[INFO] 관심 차량 선택됨: ID={selected_id}")
                return
        print("[INFO] 클릭한 위치에 차량이 없습니다.")
