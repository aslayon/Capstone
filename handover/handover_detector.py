import cv2
import time
import math
import numpy as np
from collections import deque

class HandoverDetector:
    """핸드오버 상황 감지 및 관리"""
    
    def __init__(self):
        # 핸드오버 감지 파라미터
        self.prediction_timeout = 3.0  # 3초 이상 예측만 지속되면 핸드오버 고려
        self.confidence_threshold = 0.3  # 신뢰도 임계값
        self.boundary_margin = 0.2  # 화면 경계 마진 (20%)
        
        # 핸드오버 이력 관리
        self.handover_candidates = {}  # track_id -> 후보 정보
        self.handover_history = deque(maxlen=100)  # 최근 핸드오버 이력
        
        # 화면 크기 (동적으로 업데이트)
        self.frame_width = 0
        self.frame_height = 0
        
    def update_frame_size(self, frame):
        """프레임 크기 업데이트"""
        self.frame_height, self.frame_width = frame.shape[:2]
    
    def check_handover_conditions(self, track_info, current_time):
        """핸드오버 조건 확인"""
        track_id = track_info['id']
        state = track_info['state']
        confidence = track_info['confidence']
        time_since_detection = track_info['time_since_detection']
        predicted_bbox = track_info['bbox']
        velocity = track_info.get('velocity', (0, 0))
        
        # 조건 체크 결과 저장
        conditions = {
            'is_predicting': False,
            'timeout_exceeded': False, 
            'low_confidence': False,
            'near_boundary': False,
            'moving_outward': False
        }
        
        # 1. 예측 상태인가?
        if state == "PREDICTING":
            conditions['is_predicting'] = True
        
        # 2. 예측 시간 초과?
        if time_since_detection > self.prediction_timeout:
            conditions['timeout_exceeded'] = True
        
        # 3. 신뢰도 낮음?
        if confidence < self.confidence_threshold:
            conditions['low_confidence'] = True
        
        # 4. 화면 경계 근처?
        if self.frame_width > 0 and self.frame_height > 0:
            boundary_info = self._check_boundary_proximity(predicted_bbox)
            conditions['near_boundary'] = boundary_info['is_near']
            conditions['boundary_direction'] = boundary_info['direction']
        
        # 5. 화면 밖으로 이동 중?
        if self.frame_width > 0 and self.frame_height > 0:
            outward_info = self._check_outward_movement(predicted_bbox, velocity)
            conditions['moving_outward'] = outward_info['is_moving_out']
            conditions['exit_direction'] = outward_info['direction']
        
        return conditions
    
    def _check_boundary_proximity(self, bbox):
        """화면 경계 근처인지 확인"""
        if self.frame_width <= 0 or self.frame_height <= 0:
            return {'is_near': False, 'direction': None}
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 정규화된 좌표
        norm_x = center_x / self.frame_width
        norm_y = center_y / self.frame_height
        
        margin = self.boundary_margin
        directions = []
        
        if norm_x < margin:
            directions.append('left')
        if norm_x > (1 - margin):
            directions.append('right')
        if norm_y < margin:
            directions.append('top')
        if norm_y > (1 - margin):
            directions.append('bottom')
        
        return {
            'is_near': len(directions) > 0,
            'direction': directions[0] if directions else None,
            'all_directions': directions,
            'distance_to_edge': min(norm_x, 1-norm_x, norm_y, 1-norm_y)
        }
    
    def _check_outward_movement(self, bbox, velocity):
        """화면 밖으로 향하는 움직임인지 확인"""
        if self.frame_width <= 0 or self.frame_height <= 0:
            return {'is_moving_out': False, 'direction': None}
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        vx, vy = velocity
        
        # 속도 크기가 너무 작으면 판단 불가
        speed = math.sqrt(vx*vx + vy*vy)
        if speed < 1.0:  # 픽셀/프레임
            return {'is_moving_out': False, 'direction': None}
        
        # 정규화된 좌표와 속도
        norm_x = center_x / self.frame_width
        norm_y = center_y / self.frame_height
        norm_vx = vx / self.frame_width
        norm_vy = vy / self.frame_height
        
        # 경계와 속도 방향 분석
        direction = None
        is_moving_out = False
        
        # 왼쪽 경계로 이동
        if norm_x < 0.3 and norm_vx < -0.001:
            direction = 'left'
            is_moving_out = True
        # 오른쪽 경계로 이동
        elif norm_x > 0.7 and norm_vx > 0.001:
            direction = 'right'
            is_moving_out = True
        # 위쪽 경계로 이동
        elif norm_y < 0.3 and norm_vy < -0.001:
            direction = 'top'
            is_moving_out = True
        # 아래쪽 경계로 이동
        elif norm_y > 0.7 and norm_vy > 0.001:
            direction = 'bottom'  
            is_moving_out = True
        
        return {
            'is_moving_out': is_moving_out,
            'direction': direction,
            'speed': speed,
            'velocity_direction': math.degrees(math.atan2(vy, vx))
        }
    
    def evaluate_handover_probability(self, conditions):
        """핸드오버 확률 계산"""
        score = 0.0
        factors = []
        
        # 각 조건별 가중치
        if conditions['is_predicting']:
            score += 0.3
            factors.append("예측중")
        
        if conditions['timeout_exceeded']:
            score += 0.4
            factors.append("시간초과")
        
        if conditions['low_confidence']:
            score += 0.2
            factors.append("낮은신뢰도")
        
        if conditions['near_boundary']:
            score += 0.3
            factors.append("경계근처")
        
        if conditions['moving_outward']:
            score += 0.5
            factors.append("외향이동")
        
        # 최대 1.0으로 정규화
        probability = min(score, 1.0)
        
        return {
            'probability': probability,
            'factors': factors,
            'is_handover': probability > 0.7  # 70% 이상이면 핸드오버
        }
    
    def register_handover_candidate(self, track_id, track_info, conditions, probability_info):
        """핸드오버 후보로 등록"""
        candidate_info = {
            'track_id': track_id,
            'track_info': track_info,
            'conditions': conditions,
            'probability': probability_info,
            'registered_time': time.time(),
            'status': 'CANDIDATE'  # CANDIDATE -> CONFIRMED -> RESOLVED
        }
        
        self.handover_candidates[track_id] = candidate_info
        
        print(f"🔄 핸드오버 후보 등록: ID{track_id}")
        print(f"   확률: {probability_info['probability']:.2f}")
        print(f"   요인: {', '.join(probability_info['factors'])}")
        print(f"   방향: {conditions.get('exit_direction', '알수없음')}")
        
        return candidate_info
    
    def update_handover_candidate(self, track_id, new_track_info):
        """기존 핸드오버 후보 정보 업데이트"""
        if track_id not in self.handover_candidates:
            return None
        
        candidate = self.handover_candidates[track_id]
        candidate['track_info'] = new_track_info
        candidate['last_updated'] = time.time()
        
        # 상태가 DETECTED로 돌아오면 후보에서 제거
        if new_track_info['state'] == 'DETECTED':
            print(f"✅ 핸드오버 후보 해제: ID{track_id} (재탐지됨)")
            del self.handover_candidates[track_id]
            return None
        
        return candidate
    
    def confirm_handover(self, track_id):
        """핸드오버 확정"""
        if track_id not in self.handover_candidates:
            return None
        
        candidate = self.handover_candidates[track_id]
        candidate['status'] = 'CONFIRMED'
        candidate['confirmed_time'] = time.time()
        
        # 핸드오버 이력에 추가
        handover_record = {
            'track_id': track_id,
            'timestamp': time.time(),
            'track_info': candidate['track_info'].copy(),
            'conditions': candidate['conditions'].copy(),
            'probability': candidate['probability']['probability']
        }
        self.handover_history.append(handover_record)
        
        print(f"🚀 핸드오버 확정: ID{track_id}")
        
        return candidate
    
    def get_handover_direction(self, track_id):
        """핸드오버 방향 반환"""
        if track_id not in self.handover_candidates:
            return None
        
        candidate = self.handover_candidates[track_id]
        conditions = candidate['conditions']
        
        # 우선순위: 이동 방향 > 경계 위치
        if conditions.get('exit_direction'):
            return conditions['exit_direction']
        elif conditions.get('boundary_direction'):
            return conditions['boundary_direction']
        else:
            return None
    
    def cleanup_old_candidates(self, max_age=30.0):
        """오래된 핸드오버 후보 정리"""
        current_time = time.time()
        expired_ids = []
        
        for track_id, candidate in self.handover_candidates.items():
            age = current_time - candidate['registered_time']
            if age > max_age:
                expired_ids.append(track_id)
        
        for track_id in expired_ids:
            print(f"🗑️ 만료된 핸드오버 후보 제거: ID{track_id}")
            del self.handover_candidates[track_id]
        
        return len(expired_ids)
    
    def get_statistics(self):
        """핸드오버 감지 통계"""
        return {
            'active_candidates': len(self.handover_candidates),
            'total_handovers': len(self.handover_history),
            'recent_handovers': len([h for h in self.handover_history 
                                   if time.time() - h['timestamp'] < 60]),
            'candidate_ids': list(self.handover_candidates.keys())
        }


# 사용 예시 함수
def integrate_handover_detection(tracker, frame):
    """트래커와 핸드오버 감지 통합"""
    
    handover_detector = HandoverDetector()
    handover_detector.update_frame_size(frame)
    
    # 트래커에서 모든 트랙 정보 가져오기
    tracks = tracker.update([])  # 예측만 수행
    current_time = time.time()
    
    handover_events = []
    
    for track_id, x1, y1, x2, y2, state, confidence in tracks:
        # 트랙 상세 정보 구성
        track_info = {
            'id': track_id,
            'bbox': (x1, y1, x2, y2),
            'state': state,
            'confidence': confidence,
            'time_since_detection': 0,  # tracker에서 가져와야 함
            'velocity': (0, 0)  # tracker에서 가져와야 함
        }
        
        # 실제로는 tracker.get_track_info(track_id)로 상세 정보 가져오기
        detailed_info = tracker.get_track_info(track_id)
        if detailed_info:
            track_info.update(detailed_info)
        
        # 핸드오버 조건 확인
        conditions = handover_detector.check_handover_conditions(track_info, current_time)
        
        # 핸드오버 확률 계산
        probability_info = handover_detector.evaluate_handover_probability(conditions)
        
        # 핸드오버 판정
        if probability_info['is_handover']:
            if track_id not in handover_detector.handover_candidates:
                # 새로운 핸드오버 후보 등록
                candidate = handover_detector.register_handover_candidate(
                    track_id, track_info, conditions, probability_info
                )
                handover_events.append({
                    'type': 'NEW_CANDIDATE',
                    'track_id': track_id,
                    'direction': handover_detector.get_handover_direction(track_id),
                    'candidate': candidate
                })
            else:
                # 기존 후보 업데이트
                handover_detector.update_handover_candidate(track_id, track_info)
        
        # 확정 조건 확인 (예: 3초 이상 후보 상태)
        if track_id in handover_detector.handover_candidates:
            candidate = handover_detector.handover_candidates[track_id]
            if (current_time - candidate['registered_time'] > 3.0 and 
                candidate['status'] == 'CANDIDATE'):
                
                confirmed = handover_detector.confirm_handover(track_id)
                handover_events.append({
                    'type': 'CONFIRMED',
                    'track_id': track_id,
                    'direction': handover_detector.get_handover_direction(track_id),
                    'candidate': confirmed
                })
    
    # 정리 작업
    handover_detector.cleanup_old_candidates()
    
    return handover_events, handover_detector


if __name__ == "__main__":
    # 테스트 코드
    detector = HandoverDetector()
    
    # 샘플 트랙 정보
    sample_track = {
        'id': 123,
        'bbox': (50, 100, 150, 200),  # 왼쪽 경계 근처
        'state': 'PREDICTING',
        'confidence': 0.2,  # 낮은 신뢰도
        'time_since_detection': 4.0,  # 4초 경과
        'velocity': (-5, 0)  # 왼쪽으로 이동
    }
    
    detector.frame_width = 800
    detector.frame_height = 600
    
    # 핸드오버 조건 확인
    conditions = detector.check_handover_conditions(sample_track, time.time())
    probability = detector.evaluate_handover_probability(conditions)
    
    print("🔍 핸드오버 감지 테스트:")
    print(f"조건: {conditions}")
    print(f"확률: {probability}")
    
    if probability['is_handover']:
        detector.register_handover_candidate(123, sample_track, conditions, probability)
        print(f"방향: {detector.get_handover_direction(123)}")