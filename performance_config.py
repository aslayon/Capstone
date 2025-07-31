# performance_config.py
"""성능 최적화 설정 및 유틸리티"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional

class PerformanceConfig:
    """성능 최적화 설정 클래스"""
    
    def __init__(self, performance_level: str = "balanced"):
        """
        성능 레벨별 설정
        - ultra_fast: 최고 속도 우선 (5-10 FPS)
        - fast: 빠른 처리 (10-15 FPS)  
        - balanced: 균형 (15-20 FPS)
        - quality: 품질 우선 (20-25 FPS)
        """
        self.performance_level = performance_level
        self._setup_config()
    
    def _setup_config(self):
        """성능 레벨별 구체적 설정"""
        configs = {
            "ultra_fast": {
                "target_fps": 10,
                "frame_skip": 4,
                "detection_interval": 6,
                "reid_interval": 20,
                "yolo_conf_threshold": 0.5,
                "yolo_img_size": 320,
                "window_width": 800,
                "window_height": 450,
                "max_tracks": 20,
                "iou_threshold": 0.5,
                "vehicle_classes": ['car'],
                "interpolation": cv2.INTER_NEAREST,
                "tracker_max_age": 50
            },
            "fast": {
                "target_fps": 15,
                "frame_skip": 3,
                "detection_interval": 4,
                "reid_interval": 15,
                "yolo_conf_threshold": 0.4,
                "yolo_img_size": 416,
                "window_width": 960,
                "window_height": 540,
                "max_tracks": 30,
                "iou_threshold": 0.4,
                "vehicle_classes": ['car'],
                "interpolation": cv2.INTER_LINEAR,
                "tracker_max_age": 75
            },
            "balanced": {
                "target_fps": 20,
                "frame_skip": 2,
                "detection_interval": 3,
                "reid_interval": 10,
                "yolo_conf_threshold": 0.3,
                "yolo_img_size": 640,
                "window_width": 1024,
                "window_height": 576,
                "max_tracks": 50,
                "iou_threshold": 0.3,
                "vehicle_classes": ['car', 'truck'],
                "interpolation": cv2.INTER_LINEAR,
                "tracker_max_age": 100
            },
            "quality": {
                "target_fps": 25,
                "frame_skip": 1,
                "detection_interval": 2,
                "reid_interval": 8,
                "yolo_conf_threshold": 0.25,
                "yolo_img_size": 832,
                "window_width": 1280,
                "window_height": 720,
                "max_tracks": 100,
                "iou_threshold": 0.25,
                "vehicle_classes": ['car', 'truck', 'bus'],
                "interpolation": cv2.INTER_CUBIC,
                "tracker_max_age": 150
            }
        }
        
        self.config = configs.get(self.performance_level, configs["balanced"])
        print(f"🎯 성능 모드: {self.performance_level}")
        print(f"   목표 FPS: {self.config['target_fps']}")
        print(f"   YOLO 크기: {self.config['yolo_img_size']}")
        print(f"   화면 해상도: {self.config['window_width']}x{self.config['window_height']}")
    
    def get(self, key: str, default=None):
        """설정값 가져오기"""
        return self.config.get(key, default)
    
    def update_dynamic_settings(self, current_fps: float):
        """현재 FPS에 따른 동적 설정 조정"""
        target_fps = self.config['target_fps']
        
        if current_fps < target_fps * 0.7:  # 목표의 70% 미만
            # 성능 향상 필요
            self.config['frame_skip'] = min(self.config['frame_skip'] + 1, 5)
            self.config['detection_interval'] = min(self.config['detection_interval'] + 1, 8)
            print(f"⚡ 성능 개선: skip={self.config['frame_skip']}, detect={self.config['detection_interval']}")
            
        elif current_fps > target_fps * 1.2:  # 목표의 120% 초과
            # 품질 향상 가능
            self.config['frame_skip'] = max(self.config['frame_skip'] - 1, 1)
            self.config['detection_interval'] = max(self.config['detection_interval'] - 1, 1)
            print(f"🔧 품질 개선: skip={self.config['frame_skip']}, detect={self.config['detection_interval']}")


class FrameProcessor:
    """최적화된 프레임 처리 클래스"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.frame_buffer = []
        self.max_buffer_size = 3
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리 최적화"""
        if frame is None:
            return None
        
        # 메모리 효율적인 리사이즈
        height, width = frame.shape[:2]
        target_width = self.config.get('window_width')
        target_height = self.config.get('window_height')
        
        if width != target_width or height != target_height:
            interpolation = self.config.get('interpolation', cv2.INTER_LINEAR)
            frame = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
        
        return frame
    
    def smart_crop_for_detection(self, frame: np.ndarray, roi_percentage: float = 0.8) -> np.ndarray:
        """관심 영역만 크롭하여 탐지 속도 향상"""
        if frame is None:
            return None
            
        height, width = frame.shape[:2]
        
        # 중앙 80% 영역만 사용 (차량이 주로 나타나는 영역)
        margin_h = int(height * (1 - roi_percentage) / 2)
        margin_w = int(width * (1 - roi_percentage) / 2)
        
        cropped = frame[margin_h:height-margin_h, margin_w:width-margin_w]
        return cropped
    
    def add_to_buffer(self, frame: np.ndarray):
        """프레임 버퍼링"""
        if len(self.frame_buffer) >= self.max_buffer_size:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(frame)
    
    def get_buffered_frame(self, index: int = -1) -> Optional[np.ndarray]:
        """버퍼된 프레임 가져오기"""
        if not self.frame_buffer:
            return None
        return self.frame_buffer[index] if abs(index) <= len(self.frame_buffer) else None


class AdaptiveDetector:
    """적응형 탐지 시스템"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.detection_history = []
        self.performance_history = []
        self.adaptive_threshold = config.get('yolo_conf_threshold')
        
    def should_run_detection(self, frame_count: int, current_fps: float) -> bool:
        """탐지 실행 여부 결정"""
        detection_interval = self.config.get('detection_interval')
        
        # FPS가 낮으면 탐지 간격 증가
        if current_fps < self.config.get('target_fps') * 0.8:
            detection_interval *= 2
        
        return frame_count % detection_interval == 0
    
    def adaptive_confidence_threshold(self, detection_count: int) -> float:
        """탐지 개수에 따른 적응형 임계값"""
        base_threshold = self.config.get('yolo_conf_threshold')
        
        if detection_count > 50:  # 너무 많은 탐지
            return min(base_threshold + 0.1, 0.8)
        elif detection_count < 5:  # 너무 적은 탐지
            return max(base_threshold - 0.1, 0.2)
        
        return base_threshold
    
    def record_performance(self, detection_time: float, detection_count: int):
        """성능 기록"""
        self.performance_history.append({
            'time': detection_time,
            'count': detection_count,
            'threshold': self.adaptive_threshold
        })
        
        # 최근 10개 기록만 유지
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)


class MemoryManager:
    """메모리 관리 클래스"""
    
    def __init__(self):
        self.cache_limit = 100
        self.cleanup_interval = 50
        self.frame_counter = 0
        
    def should_cleanup(self) -> bool:
        """정리가 필요한지 확인"""
        self.frame_counter += 1
        return self.frame_counter % self.cleanup_interval == 0
    
    def cleanup_cache(self, cache_dict: dict, max_age: float = 2.0):
        """캐시 정리"""
        current_time = time.time()
        expired_keys = []
        
        for key, value in cache_dict.items():
            if isinstance(value, dict) and 'timestamp' in value:
                if current_time - value['timestamp'] > max_age:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del cache_dict[key]
        
        if expired_keys:
            print(f"🧹 캐시 정리: {len(expired_keys)}개 항목 삭제")
    
    def get_memory_usage(self) -> dict:
        """메모리 사용량 확인"""
        import psutil
        process = psutil.Process(os.getpid())
        return {
            'memory_percent': process.memory_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent()
        }


# 성능 모니터링 데코레이터
def performance_monitor(func):
    """성능 측정 데코레이터"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        if execution_time > 0.1:  # 100ms 이상이면 경고
            print(f"⚠️ {func.__name__}: {execution_time:.3f}s")
        
        return result
    return wrapper


# 사용 예시
if __name__ == "__main__":
    # 성능 레벨별 테스트
    for level in ["ultra_fast", "fast", "balanced", "quality"]:
        print(f"\n=== {level.upper()} 모드 ===")
        config = PerformanceConfig(level)
        
        # 주요 설정 출력
        print(f"프레임 스킵: {config.get('frame_skip')}")
        print(f"탐지 간격: {config.get('detection_interval')}")
        print(f"YOLO 크기: {config.get('yolo_img_size')}")
        print(f"신뢰도 임계값: {config.get('yolo_conf_threshold')}")