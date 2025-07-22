import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import itertools
import time

class PersistentTrack:
    """지속적 예측이 가능한 트랙"""
    
    _id_iter = itertools.count()

    def __init__(self, bbox):
        self.id = next(self._id_iter)
        self.kf = self._init_kalman_filter(bbox)
        
        # 기본 상태
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 0
        self.age = 0
        self.bbox = bbox
        
        # 지속성을 위한 추가 속성
        self.detection_history = []  # 탐지 히스토리
        self.prediction_history = []  # 예측 히스토리
        self.last_detection_time = time.time()
        self.confidence_score = 1.0  # 트랙 신뢰도
        
        # 상태 관리
        self.state = "DETECTED"  # DETECTED, PREDICTING, LOST
        self.max_prediction_time = 5.0  # 5초까지 예측 유지

    def _init_kalman_filter(self, bbox):
        """칼만 필터 초기화 (속도 추정 향상)"""
        kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # 상태 전이 행렬 (위치 + 속도)
        kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1]])
        
        # 관측 행렬
        kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]])
        
        # 노이즈 설정 (예측 성능 향상)
        kf.R *= 10.   # 관측 노이즈
        kf.P *= 500.  # 초기 불확실성
        kf.Q *= 0.01  # 프로세스 노이즈
        
        # 초기 상태 설정
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        
        kf.x[:4] = np.array([[cx], [cy], [s], [r]])
        
        return kf

    def predict(self):
        """예측 수행 (상태 관리 포함)"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
        # 예측 히스토리 저장
        predicted_bbox = self.get_bbox()
        self.prediction_history.append({
            'bbox': predicted_bbox,
            'timestamp': time.time(),
            'confidence': self.confidence_score
        })
        
        # 예측 히스토리 크기 제한
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-50:]
        
        # 상태 업데이트
        time_since_detection = time.time() - self.last_detection_time
        
        if time_since_detection > self.max_prediction_time:
            self.state = "LOST"
            self.confidence_score *= 0.95  # 신뢰도 감소
        elif self.time_since_update > 0:
            self.state = "PREDICTING"
            self.confidence_score *= 0.98  # 약간 감소
        else:
            self.state = "DETECTED"
            self.confidence_score = min(1.0, self.confidence_score * 1.02)  # 회복
        
        return self.kf.x[:4].flatten()

    def update(self, bbox):
        """실제 탐지로 업데이트"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        
        self.kf.update(np.array([cx, cy, s, r]))
        
        # 상태 업데이트
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.bbox = bbox
        self.last_detection_time = time.time()
        self.state = "DETECTED"
        self.confidence_score = min(1.0, self.confidence_score * 1.05)
        
        # 탐지 히스토리 저장
        self.detection_history.append({
            'bbox': bbox,
            'timestamp': time.time()
        })
        
        # 히스토리 크기 제한
        if len(self.detection_history) > 50:
            self.detection_history = self.detection_history[-25:]

    def get_bbox(self):
        """현재 bbox 반환 (예측 또는 실제)"""
        cx, cy, s, r = self.kf.x[:4].flatten()
        w = max(1e-6, np.sqrt(s * r))
        h = max(1e-6, s / w)
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return (x1, y1, x2, y2)

    def get_velocity(self):
        """현재 속도 반환"""
        if len(self.kf.x) >= 7:
            vx, vy = self.kf.x[4], self.kf.x[5]
            return (float(vx), float(vy))
        return (0, 0)

    def is_valid(self):
        """트랙이 유효한지 확인"""
        time_since_detection = time.time() - self.last_detection_time
        return (time_since_detection < self.max_prediction_time and 
                self.confidence_score > 0.1)

    def contains_point(self, x, y):
        """점이 bbox 안에 있는지 확인"""
        x1, y1, x2, y2 = self.get_bbox()
        return x1 <= x <= x2 and y1 <= y <= y2


class PersistentMultiTracker:
    """지속적 예측이 가능한 멀티 트래커"""
    
    def __init__(self, max_age=150, iou_threshold=0.3):  # max_age 늘림 (5초)
        self.max_age = max_age  # 150프레임 = 약 5초
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.selected_id = None
        
        # 재연결 기능
        self.lost_tracks = []  # 잃어버린 트랙들
        self.reactivation_threshold = 0.5  # 재활성화 IoU 임계값

    def update(self, detections):
        """트래커 업데이트 (지속적 예측 포함)"""
        # 1. 모든 트랙 예측 수행
        predicted = []
        for track in self.tracks:
            pred = track.predict()
            predicted.append(pred)
        
        # 2. 탐지와 예측 매칭
        matched, unmatched_dets, unmatched_tracks = self._match(detections, predicted)
        
        # 3. 매칭된 트랙 업데이트
        for det_idx, track_idx in matched:
            self.tracks[track_idx].update(detections[det_idx][:4])
        
        # 4. 새로운 탐지에 대한 새 트랙 생성
        for idx in unmatched_dets:
            self.tracks.append(PersistentTrack(detections[idx][:4]))
        
        # 5. 매칭되지 않은 트랙 처리 (삭제하지 않고 예측 상태로 유지)
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            # 여기서 삭제하지 않음! 계속 예측만 수행
        
        # 6. 유효하지 않은 트랙만 제거
        valid_tracks = []
        for track in self.tracks:
            if track.is_valid():
                valid_tracks.append(track)
            else:
                # 완전히 잃어버린 트랙은 lost_tracks로 이동
                self.lost_tracks.append(track)
                print(f"🔍 트랙 ID{track.id} 분실됨 (마지막 위치: {track.get_bbox()})")
        
        self.tracks = valid_tracks
        
        # 7. lost_tracks 정리 (너무 오래된 것들 제거)
        current_time = time.time()
        self.lost_tracks = [t for t in self.lost_tracks 
                           if current_time - t.last_detection_time < 30.0]  # 30초 후 완전 삭제
        
        # 8. 결과 반환 (상태별로 구분)
        results = []
        for track in self.tracks:
            x1, y1, x2, y2 = track.get_bbox()
            results.append((track.id, x1, y1, x2, y2, track.state, track.confidence_score))
        
        return results

    def predict_only(self):
        """탐지 없이 예측만 수행"""
        for track in self.tracks:
            track.predict()
        
        results = []
        for track in self.tracks:
            if track.is_valid():
                x1, y1, x2, y2 = track.get_bbox()
                results.append((track.id, x1, y1, x2, y2, track.state, track.confidence_score))
        
        return results

    def try_reactivate_lost_tracks(self, detections):
        """잃어버린 트랙 재활성화 시도"""
        if not self.lost_tracks or not detections:
            return []
        
        reactivated = []
        
        for detection in detections:
            det_bbox = detection[:4]
            
            best_match = None
            best_iou = 0
            
            for lost_track in self.lost_tracks:
                # 잃어버린 트랙의 예측 위치와 비교
                lost_track.predict()
                predicted_bbox = lost_track.get_bbox()
                
                iou = self._calculate_iou(det_bbox, predicted_bbox)
                
                if iou > best_iou and iou > self.reactivation_threshold:
                    best_iou = iou
                    best_match = lost_track
            
            if best_match:
                # 재활성화
                best_match.update(det_bbox)
                self.tracks.append(best_match)
                self.lost_tracks.remove(best_match)
                
                reactivated.append(best_match.id)
                print(f"🎯 트랙 ID{best_match.id} 재활성화! (IoU: {best_iou:.3f})")
        
        return reactivated

    def _match(self, detections, predicted):
        """탐지와 예측 매칭"""
        if len(predicted) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(predicted)))

        iou_matrix = np.zeros((len(detections), len(predicted)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, pred in enumerate(predicted):
                predicted_bbox = self._convert_to_bbox(pred)
                iou_matrix[d, t] = self._calculate_iou(det[:4], predicted_bbox)

        # Hungarian 알고리즘으로 최적 매칭
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = list(zip(*matched_indices))

        unmatched_dets = list(set(range(len(detections))) - {m[0] for m in matched_indices})
        unmatched_tracks = list(set(range(len(predicted))) - {m[1] for m in matched_indices})

        # IoU 임계값 이상인 매칭만 유지
        matches = [m for m in matched_indices if iou_matrix[m[0], m[1]] >= self.iou_threshold]
        
        return matches, unmatched_dets, unmatched_tracks

    def _convert_to_bbox(self, pred):
        """예측 상태를 bbox로 변환"""
        cx, cy, s, r = pred
        w = max(1e-6, np.sqrt(s * r))
        h = max(1e-6, s / w)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return (x1, y1, x2, y2)

    def _calculate_iou(self, bb_test, bb_gt):
        """IoU 계산"""
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
        """점 클릭으로 트랙 선택"""
        if x < 0 or y < 0:  # 강제 해제
            if self.selected_id is not None:
                print(f"[INFO] 트랙 선택 해제됨 (이전 ID: {self.selected_id})")
                self.selected_id = None
            return
            
        for track in self.tracks:
            if track.contains_point(x, y):
                if self.selected_id == track.id:
                    self.selected_id = None
                    print(f"[INFO] 트랙 해제됨 (ID: {track.id})")
                else:
                    self.selected_id = track.id
                    print(f"[INFO] 트랙 선택됨 (ID: {track.id}, 상태: {track.state})")
                return
        
        # 클릭 위치에 트랙이 없으면 선택 해제
        if self.selected_id is not None:
            print("[INFO] 빈 공간 클릭으로 트랙 선택 해제됨")
            self.selected_id = None

    def get_selected_bbox(self):
        """선택된 트랙의 bbox 반환"""
        for track in self.tracks:
            if track.id == self.selected_id:
                return track.get_bbox()
        return None

    def get_track_info(self, track_id):
        """트랙 상세 정보 반환"""
        for track in self.tracks:
            if track.id == track_id:
                return {
                    'id': track.id,
                    'state': track.state,
                    'confidence': track.confidence_score,
                    'velocity': track.get_velocity(),
                    'time_since_detection': time.time() - track.last_detection_time,
                    'total_detections': len(track.detection_history),
                    'predictions_count': len(track.prediction_history)
                }
        return None


# 사용 예시
def main_with_persistent_tracking():
    """지속적 예측이 포함된 메인 루프"""
    
    tracker = PersistentMultiTracker(max_age=150, iou_threshold=0.3)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # YOLO 탐지
        detections = get_vehicle_detections(frame, vehicle_classes=['car', 'truck'])
        
        # 잃어버린 트랙 재활성화 시도
        reactivated = tracker.try_reactivate_lost_tracks(detections)
        if reactivated:
            print(f"🎉 재활성화된 트랙들: {reactivated}")
        
        # 트래킹 수행
        tracks = tracker.update(detections)
        
        # 시각화 (상태별 색상)
        for track_id, x1, y1, x2, y2, state, confidence in tracks:
            # 상태별 색상
            if state == "DETECTED":
                color = (0, 255, 0)  # 초록색
            elif state == "PREDICTING":
                color = (0, 255, 255)  # 노란색
            elif state == "LOST":
                color = (0, 0, 255)  # 빨간색
            else:
                color = (128, 128, 128)  # 회색
            
            # 선택된 트랙은 더 두껍게
            thickness = 4 if track_id == tracker.selected_id else 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # 라벨 (상태 + 신뢰도 포함)
            label = f"ID{track_id} {state[:4]} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow("Persistent Tracking", frame)
        
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break