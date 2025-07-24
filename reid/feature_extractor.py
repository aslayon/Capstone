import cv2
import numpy as np
import time
import math
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity

class VehicleFeatureExtractor:
    """차량 특징 추출기 - 색상, 크기, 형태 기반"""
    
    def __init__(self):
        # 색상 범위 정의 (HSV)
        self.color_ranges = {
            'white': ([0, 0, 200], [180, 30, 255]),
            'black': ([0, 0, 0], [180, 255, 50]),
            'gray': ([0, 0, 50], [180, 30, 200]),
            'red': ([0, 120, 120], [10, 255, 255]),  # 빨간색 1
            'red2': ([170, 120, 120], [180, 255, 255]),  # 빨간색 2
            'blue': ([100, 120, 120], [130, 255, 255]),
            'green': ([40, 120, 120], [80, 255, 255]),
            'yellow': ([15, 120, 120], [35, 255, 255])
        }
        
        # 특징 가중치
        self.feature_weights = {
            'color': 0.4,      # 색상이 가장 중요
            'size': 0.3,       # 크기
            'shape': 0.2,      # 형태
            'position': 0.1    # 위치 (보조적)
        }
        
    def extract_features(self, frame, bbox, class_name="unknown", frame_info=None):
        """차량 bbox에서 종합 특징 추출"""
        x1, y1, x2, y2 = bbox
        
        # bbox 유효성 검사
        if x1 >= x2 or y1 >= y2:
            return None
        
        # 프레임 경계 체크
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # 차량 영역 크롭
        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            return None
        
        try:
            # 각 특징 추출
            color_features = self._extract_color_features(vehicle_crop)
            size_features = self._extract_size_features(bbox, frame.shape)
            shape_features = self._extract_shape_features(vehicle_crop)
            position_features = self._extract_position_features(bbox, frame.shape)
            
            # 특징 벡터 구성
            feature_vector = {
                'color': color_features,
                'size': size_features,
                'shape': shape_features,
                'position': position_features,
                'class_name': class_name,
                'bbox': bbox,
                'timestamp': time.time(),
                'frame_info': frame_info or {}
            }
            
            return feature_vector
            
        except Exception as e:
            print(f"⚠️ 특징 추출 실패: {e}")
            return None
    
    def _extract_color_features(self, crop):
        """색상 특징 추출"""
        # HSV 변환
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # 각 색상별 점수 계산
        color_scores = {}
        total_pixels = crop.shape[0] * crop.shape[1]
        
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            score = cv2.countNonZero(mask) / total_pixels
            color_scores[color_name] = score
        
        # 빨간색 특별 처리 (두 범위 합치기)
        if 'red' in color_scores and 'red2' in color_scores:
            color_scores['red'] = color_scores['red'] + color_scores['red2']
            del color_scores['red2']
        
        # 주요 색상 찾기
        dominant_color = max(color_scores.items(), key=lambda x: x[1])
        
        # 색상 히스토그램 (간단화)
        hist_h = cv2.calcHist([hsv], [0], None, [36], [0, 180])  # Hue
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])  # Saturation
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])  # Value
        
        # 정규화
        hist_h = hist_h.flatten() / total_pixels
        hist_s = hist_s.flatten() / total_pixels  
        hist_v = hist_v.flatten() / total_pixels
        
        return {
            'dominant_color': dominant_color[0],
            'dominant_confidence': dominant_color[1],
            'color_distribution': color_scores,
            'histogram_h': hist_h.tolist(),
            'histogram_s': hist_s.tolist(), 
            'histogram_v': hist_v.tolist(),
            'avg_brightness': np.mean(hsv[:,:,2]) / 255.0
        }
    
    def _extract_size_features(self, bbox, frame_shape):
        """크기 특징 추출"""
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame_shape[:2]
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / max(height, 1)
        
        # 프레임 대비 정규화
        norm_width = width / frame_w
        norm_height = height / frame_h
        norm_area = area / (frame_w * frame_h)
        
        # 크기 카테고리 분류
        if norm_area < 0.01:
            size_category = 'small'
        elif norm_area < 0.05:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        return {
            'width': width,
            'height': height,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'norm_width': norm_width,
            'norm_height': norm_height,
            'norm_area': norm_area,
            'size_category': size_category
        }
    
    def _extract_shape_features(self, crop):
        """형태 특징 추출"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.countNonZero(edges) / (gray.shape[0] * gray.shape[1])
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 윤곽선
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 윤곽선 특징
            contour_area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # 직사각형성
            if perimeter > 0:
                rectangularity = 4 * math.pi * contour_area / (perimeter * perimeter)
            else:
                rectangularity = 0
            
            # 경계 박스와의 비율
            x, y, w, h = cv2.boundingRect(largest_contour)
            bbox_area = w * h
            fill_ratio = contour_area / max(bbox_area, 1)
            
        else:
            rectangularity = 0
            fill_ratio = 0
            contour_area = 0
        
        # 텍스처 특징 (간단한 LBP)
        texture_score = self._calculate_texture_score(gray)
        
        return {
            'edge_density': edge_density,
            'rectangularity': rectangularity,
            'fill_ratio': fill_ratio,
            'texture_score': texture_score,
            'contour_count': len(contours)
        }
    
    def _extract_position_features(self, bbox, frame_shape):
        """위치 특징 추출"""
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame_shape[:2]
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 정규화된 위치
        norm_center_x = center_x / frame_w
        norm_center_y = center_y / frame_h  
        
        # 화면 영역 구분
        if norm_center_x < 0.33:
            x_region = 'left'
        elif norm_center_x < 0.67:
            x_region = 'center'
        else:
            x_region = 'right'
        
        if norm_center_y < 0.33:
            y_region = 'top'
        elif norm_center_y < 0.67:
            y_region = 'middle'
        else:
            y_region = 'bottom'
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'norm_center_x': norm_center_x,
            'norm_center_y': norm_center_y,
            'x_region': x_region,
            'y_region': y_region,
            'distance_from_center': math.sqrt((norm_center_x - 0.5)**2 + (norm_center_y - 0.5)**2)
        }
    
    def _calculate_texture_score(self, gray_image):
        """간단한 텍스처 점수 계산"""
        if gray_image.size == 0:
            return 0
        
        # 표준편차 기반 텍스처 측정
        texture = np.std(gray_image) / 255.0
        return texture
    
    def compare_features(self, features1, features2, debug=False):
        """두 특징 벡터 비교 - 유사도 반환 (0~1)"""
        if not features1 or not features2:
            return 0.0
        
        scores = {}
        
        # 1. 색상 유사도
        color_sim = self._compare_color_features(features1['color'], features2['color'])
        scores['color'] = color_sim
        
        # 2. 크기 유사도  
        size_sim = self._compare_size_features(features1['size'], features2['size'])
        scores['size'] = size_sim
        
        # 3. 형태 유사도
        shape_sim = self._compare_shape_features(features1['shape'], features2['shape'])
        scores['shape'] = shape_sim
        
        # 4. 클래스 일치 보너스
        if features1['class_name'] == features2['class_name']:
            class_bonus = 0.1
        else:
            class_bonus = -0.1
        
        # 가중 평균 계산
        weighted_score = (
            scores['color'] * self.feature_weights['color'] +
            scores['size'] * self.feature_weights['size'] +
            scores['shape'] * self.feature_weights['shape'] +
            class_bonus
        )
        
        final_score = max(0.0, min(1.0, weighted_score))
        
        if debug:
            print(f"🔍 특징 비교 상세:")
            print(f"   색상: {color_sim:.3f}")
            print(f"   크기: {size_sim:.3f}")  
            print(f"   형태: {shape_sim:.3f}")
            print(f"   클래스: {class_bonus:+.1f}")
            print(f"   최종: {final_score:.3f}")
        
        return final_score
    
    def _compare_color_features(self, color1, color2):
        """색상 특징 비교"""
        # 주요 색상 일치도
        if color1['dominant_color'] == color2['dominant_color']:
            dominant_match = 0.7 + 0.3 * min(
                color1['dominant_confidence'],
                color2['dominant_confidence']
            )
        else:
            dominant_match = 0.2
        
        # 히스토그램 유사도 (코사인 similarity)
        try:
            hist1_h = np.array(color1['histogram_h'])
            hist2_h = np.array(color2['histogram_h'])
            hist_sim_h = cosine_similarity([hist1_h], [hist2_h])[0][0]
            
            hist1_s = np.array(color1['histogram_s'])
            hist2_s = np.array(color2['histogram_s'])
            hist_sim_s = cosine_similarity([hist1_s], [hist2_s])[0][0]
            
            hist_similarity = (hist_sim_h + hist_sim_s) / 2
        except:
            hist_similarity = 0.5
        
        # 밝기 유사도
        brightness_diff = abs(color1['avg_brightness'] - color2['avg_brightness'])
        brightness_sim = 1 - brightness_diff
        
        # 종합 색상 점수
        color_score = (dominant_match * 0.5 + hist_similarity * 0.3 + brightness_sim * 0.2)
        return max(0, min(1, color_score))
    
    def _compare_size_features(self, size1, size2):
        """크기 특징 비교"""
        # 종횡비 유사도
        ratio_diff = abs(size1['aspect_ratio'] - size2['aspect_ratio'])
        ratio_sim = max(0, 1 - ratio_diff / 2)  # 비율 차이 2 이상이면 0점
        
        # 정규화된 면적 유사도
        area_diff = abs(size1['norm_area'] - size2['norm_area'])
        area_sim = max(0, 1 - area_diff * 10)  # 면적 차이에 따른 감점
        
        # 크기 카테고리 일치도
        if size1['size_category'] == size2['size_category']:
            category_bonus = 0.3
        else:
            category_bonus = 0
        
        size_score = ratio_sim * 0.4 + area_sim * 0.4 + category_bonus
        return max(0, min(1, size_score))
    
    def _compare_shape_features(self, shape1, shape2):
        """형태 특징 비교"""
        # 엣지 밀도 유사도
        edge_diff = abs(shape1['edge_density'] - shape2['edge_density'])
        edge_sim = max(0, 1 - edge_diff * 5)
        
        # 직사각형성 유사도
        rect_diff = abs(shape1['rectangularity'] - shape2['rectangularity'])
        rect_sim = max(0, 1 - rect_diff * 2)
        
        # 텍스처 유사도
        texture_diff = abs(shape1['texture_score'] - shape2['texture_score'])
        texture_sim = max(0, 1 - texture_diff * 2)
        
        shape_score = (edge_sim + rect_sim + texture_sim) / 3
        return max(0, min(1, shape_score))


class ReIDSystem:
    """차량 재식별 시스템"""
    
    def __init__(self, similarity_threshold=0.6):
        self.feature_extractor = VehicleFeatureExtractor()
        self.similarity_threshold = similarity_threshold
        
        # 분실 차량 데이터베이스
        self.lost_vehicles = {}  # track_id -> 특징 정보
        self.search_history = deque(maxlen=1000)
        
        # 성능 통계
        self.stats = {
            'total_searches': 0,
            'successful_matches': 0,
            'false_positives': 0
        }
    
    def register_lost_vehicle(self, track_id, frame, bbox, class_name, additional_info=None):
        """분실된 차량을 데이터베이스에 등록"""
        features = self.feature_extractor.extract_features(
            frame, bbox, class_name, additional_info
        )
        
        if features:
            self.lost_vehicles[track_id] = {
                'features': features,
                'registered_time': time.time(),
                'search_count': 0,
                'last_search_time': 0
            }
            
            print(f"🔍 분실 차량 등록: ID{track_id}")
            print(f"   색상: {features['color']['dominant_color']}")
            print(f"   크기: {features['size']['size_category']}")
            print(f"   클래스: {features['class_name']}")
            
            return True
        return False
    
    def search_in_new_camera(self, new_detections, new_frame, camera_name="", max_candidates=5):
        """새 카메라에서 분실 차량 탐색"""
        if not self.lost_vehicles or not new_detections:
            return []
        
        self.stats['total_searches'] += 1
        matches = []
        
        print(f"🔎 새 카메라에서 탐색 시작: {camera_name}")
        print(f"   분실 차량: {len(self.lost_vehicles)}개")
        print(f"   탐지된 차량: {len(new_detections)}개")
        
        # 각 새 탐지에 대해
        for det_idx, detection in enumerate(new_detections):
            if len(detection) < 5:  # 최소 (x1, y1, x2, y2, conf) 필요
                continue
            
            x1, y1, x2, y2, conf = detection[:5]
            class_name = detection[5] if len(detection) > 5 else "unknown"
            bbox = (x1, y1, x2, y2)
            
            # 새 차량의 특징 추출
            new_features = self.feature_extractor.extract_features(
                new_frame, bbox, class_name, {'camera': camera_name}
            )
            
            if not new_features:
                continue
            
            # 모든 분실 차량과 비교
            candidate_matches = []
            
            for lost_id, lost_data in self.lost_vehicles.items():
                lost_features = lost_data['features']
                
                # 특징 비교
                similarity = self.feature_extractor.compare_features(
                    lost_features, new_features
                )
                
                if similarity > self.similarity_threshold:
                    candidate_matches.append({
                        'lost_id': lost_id,
                        'similarity': similarity,
                        'new_bbox': bbox,
                        'new_features': new_features,
                        'lost_features': lost_features,
                        'time_since_lost': time.time() - lost_data['registered_time']
                    })
            
            # 유사도 순으로 정렬
            candidate_matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 최고 매치만 선택 (1:1 매칭)
            if candidate_matches:
                best_match = candidate_matches[0]
                matches.append(best_match)
                
                print(f"✅ 매칭 후보 발견:")
                print(f"   분실 ID: {best_match['lost_id']}")
                print(f"   유사도: {best_match['similarity']:.3f}")
                print(f"   분실 시간: {best_match['time_since_lost']:.1f}초")
        
        # 탐색 이력 기록
        search_record = {
            'timestamp': time.time(),
            'camera_name': camera_name,
            'detections_count': len(new_detections),
            'matches_found': len(matches),
            'matches': matches
        }
        self.search_history.append(search_record)
        
        if matches:
            print(f"🎯 총 {len(matches)}개 매칭 발견!")
        else:
            print("❌ 매칭된 차량 없음")
        
        return matches
    
    def confirm_match(self, match_info, confidence_boost=0.0):
        """매칭 확정 - 분실 차량 목록에서 제거"""
        lost_id = match_info['lost_id']
        
        if lost_id in self.lost_vehicles:
            final_similarity = match_info['similarity'] + confidence_boost
            
            print(f"🎉 차량 재식별 확정: ID{lost_id}")
            print(f"   최종 유사도: {final_similarity:.3f}")
            
            # 통계 업데이트
            self.stats['successful_matches'] += 1
            
            # 분실 목록에서 제거
            del self.lost_vehicles[lost_id]
            
            return True
        return False
    
    def reject_match(self, match_info, reason=""):
        """매칭 거부 - 잘못된 매칭 처리"""
        print(f"❌ 매칭 거부: ID{match_info['lost_id']} ({reason})")
        self.stats['false_positives'] += 1
        return False
    
    def cleanup_old_records(self, max_age=300):
        """오래된 분실 차량 기록 정리 (5분)"""
        current_time = time.time() 
        expired_ids = []
        
        for lost_id, lost_data in self.lost_vehicles.items():
            age = current_time - lost_data['registered_time']
            if age > max_age:
                expired_ids.append(lost_id)
        
        for lost_id in expired_ids:
            del self.lost_vehicles[lost_id]
            print(f"🗑️ 만료된 분실 차량 제거: ID{lost_id}")
        
        return len(expired_ids)
    
    def get_statistics(self):
        """Re-ID 시스템 통계 반환"""
        success_rate = 0
        if self.stats['total_searches'] > 0:
            success_rate = self.stats['successful_matches'] / self.stats['total_searches']
        
        return {
            'lost_vehicles_count': len(self.lost_vehicles),
            'total_searches': self.stats['total_searches'],
            'successful_matches': self.stats['successful_matches'],
            'success_rate': success_rate,
            'false_positives': self.stats['false_positives'],
            'recent_searches': len([s for s in self.search_history 
                                  if time.time() - s['timestamp'] < 60])
        }


# 사용 예시
if __name__ == "__main__":
    # ReID 시스템 초기화
    reid_system = ReIDSystem(similarity_threshold=0.6)
    
    # 가상의 테스트 시나리오
    print("🧪 ReID 시스템 테스트")
    
    # 1. 분실 차량 등록 (가상)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_bbox = (100, 100, 200, 150)
    
    reid_system.register_lost_vehicle(
        track_id=123,
        frame=fake_frame, 
        bbox=fake_bbox,
        class_name="car"
    )
    
    # 2. 새 카메라에서 탐색 (가상)
    fake_detections = [
        (150, 120, 250, 170, 0.8, "car"),  # 유사한 크기
        (300, 200, 350, 230, 0.9, "truck")  # 다른 클래스
    ]
    
    matches = reid_system.search_in_new_camera(
        fake_detections, fake_frame, "Test Camera"
    )
    
    # 3. 통계 출력
    stats = reid_system.get_statistics()
    print(f"\n📊 시스템 통계: {stats}")