# detector/yolo_detector_optimized.py
"""
YOLO 최적화 버전 - 71ms → 10ms 목표
기존 yolo_detector.py를 대체하거나 함께 사용
"""
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import os

# 전역 모델 캐시 (한 번만 로딩)
_yolo_model_cache = {}

def get_optimized_yolo_model(model_size='n'):
    """최적화된 YOLO 모델 로딩 (캐시 사용)"""
    global _yolo_model_cache
    
    if model_size not in _yolo_model_cache:
        print(f"🚀 YOLO{model_size} 모델 로딩 중...")
        
        model_path = f'yolov8{model_size}.pt'
        model = YOLO(model_path)
        
        # GPU 사용 설정
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        # 모델 워밍업 (첫 추론을 빠르게 하기 위해)
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy_image, verbose=False)
        
        _yolo_model_cache[model_size] = model
        print(f"✅ YOLO{model_size} 로딩 완료 ({device})")
    
    return _yolo_model_cache[model_size]


def get_vehicle_detections_ultra_fast(frame, conf_threshold=0.5):
    """초고속 탐지 (목표: 10ms 이하)"""
    
    if frame is None:
        return []
    
    # 1. 극도로 작은 크기로 리사이즈 (320x320)
    original_height, original_width = frame.shape[:2]
    
    # 320x320으로 고정 (가장 빠름)
    small_frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_NEAREST)
    
    # 2. nano 모델 사용
    model = get_optimized_yolo_model('n')  # nano 모델
    
    # 3. 최적화된 추론 설정
    results = model(
        small_frame,
        conf=conf_threshold,
        classes=[2, 5, 7],  # car, bus, truck만 (더 빠름)
        verbose=False,
        imgsz=320,  # 명시적으로 320 설정
        half=True if torch.cuda.is_available() else False  # FP16 사용 (GPU에서)
    )
    
    # 4. 결과 처리 (최소한만)
    detections = []
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            # 좌표를 원본 크기로 스케일
            scale_x = original_width / 320
            scale_y = original_height / 320
            
            for box in boxes:
                # 좌표 추출
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # 원본 크기로 변환
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y) 
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # 클래스 이름
                class_names = {2: 'car', 5: 'bus', 7: 'truck'}
                class_name = class_names.get(cls, 'vehicle')
                
                detections.append((x1, y1, x2, y2, float(conf), class_name))
    
    return detections


def get_vehicle_detections_fast(frame, conf_threshold=0.4):
    """빠른 탐지 (목표: 20ms 이하)"""
    
    if frame is None:
        return []
    
    # 1. 640x480으로 리사이즈
    original_height, original_width = frame.shape[:2]
    
    if original_width > 640:
        scale = 640 / original_width
        new_height = int(original_height * scale)
        small_frame = cv2.resize(frame, (640, new_height), interpolation=cv2.INTER_NEAREST)
    else:
        small_frame = frame
        scale = 1.0
    
    # 2. nano 모델 사용
    model = get_optimized_yolo_model('n')
    
    # 3. 최적화 설정
    results = model(
        small_frame,
        conf=conf_threshold,
        classes=[2, 5, 7],  # 차량류만
        verbose=False,
        imgsz=640
    )
    
    # 4. 결과 변환
    detections = []
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # 좌표 복원
                if scale != 1.0:
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale) 
                    y2 = int(y2 / scale)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                class_names = {2: 'car', 5: 'bus', 7: 'truck'}
                class_name = class_names.get(cls, 'vehicle')
                
                detections.append((x1, y1, x2, y2, float(conf), class_name))
    
    return detections


def get_vehicle_detections_balanced(frame, conf_threshold=0.3):
    """균형 잡힌 탐지 (목표: 40ms 이하)"""
    
    if frame is None:
        return []
    
    # small 모델 사용 (nano보다는 정확하지만 여전히 빠름)
    model = get_optimized_yolo_model('s')
    
    # 원본 크기가 너무 크면 축소
    original_height, original_width = frame.shape[:2]
    
    if original_width > 1280:
        scale = 1280 / original_width
        new_height = int(original_height * scale)
        frame = cv2.resize(frame, (1280, new_height), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
    
    # 탐지 실행
    results = model(
        frame,
        conf=conf_threshold,
        classes=[2, 5, 7],  # 차량류만
        verbose=False
    )
    
    # 결과 처리
    detections = []
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # 좌표 복원
                if scale != 1.0:
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                class_names = {2: 'car', 5: 'bus', 7: 'truck'}
                class_name = class_names.get(cls, 'vehicle')
                
                detections.append((x1, y1, x2, y2, float(conf), class_name))
    
    return detections


# 성능 레벨에 따른 자동 선택
def get_vehicle_detections_auto(frame, conf_threshold=0.5, performance_level="fast"):
    """성능 레벨에 따른 자동 탐지"""
    
    if performance_level == "ultra_fast":
        return get_vehicle_detections_ultra_fast(frame, conf_threshold)
    elif performance_level == "fast": 
        return get_vehicle_detections_fast(frame, conf_threshold)
    elif performance_level == "balanced":
        return get_vehicle_detections_balanced(frame, conf_threshold)
    else:
        return get_vehicle_detections_fast(frame, conf_threshold)


# 기존 함수와의 호환성을 위한 래퍼
def get_vehicle_detections(frame, conf_threshold=0.3, vehicle_classes=None, img_size=None):
    """기존 코드와 호환되도록 하는 래퍼 함수"""
    
    # 환경변수에서 성능 레벨 읽기
    performance_level = os.getenv('PERFORMANCE_LEVEL', 'fast')
    
    # 신뢰도 임계값 조정 (성능 레벨에 따라)
    if performance_level == "ultra_fast":
        conf_threshold = max(conf_threshold, 0.6)  # 더 높은 임계값으로 빠르게
    elif performance_level == "fast":
        conf_threshold = max(conf_threshold, 0.5)
    
    return get_vehicle_detections_auto(frame, conf_threshold, performance_level)


# 성능 테스트 함수
def benchmark_yolo_performance():
    """YOLO 성능 벤치마크"""
    print("🔥 YOLO 최적화 성능 테스트")
    print("-" * 50)
    
    # 테스트 프레임들
    test_frames = [
        (np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8), "320x240"),
        (np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), "640x480"),
        (np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8), "1280x720"),
    ]
    
    # 테스트 함수들
    test_funcs = [
        (get_vehicle_detections_ultra_fast, "Ultra Fast"),
        (get_vehicle_detections_fast, "Fast"),
        (get_vehicle_detections_balanced, "Balanced")
    ]
    
    for frame, frame_name in test_frames:
        print(f"\n📏 프레임 크기: {frame_name}")
        
        for func, func_name in test_funcs:
            times = []
            
            # 워밍업
            func(frame, 0.5)
            
            # 실제 측정
            for _ in range(10):
                start_time = time.time()
                detections = func(frame, 0.5)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            print(f"   {func_name:<12}: {avg_time*1000:6.1f}ms, {fps:6.1f} FPS")


if __name__ == "__main__":
    # 성능 테스트 실행
    benchmark_yolo_performance()
    
    print(f"\n💡 사용 방법:")
    print(f"   환경변수 PERFORMANCE_LEVEL 설정:")
    print(f"   - ultra_fast: 극속 (부정확할 수 있음)")
    print(f"   - fast: 빠름 (권장)")
    print(f"   - balanced: 균형")