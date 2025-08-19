"""
개선된 차량 탐지기 - YOLO + Motion Detection
고속도로 CCTV용 최적화 (크기 제한 제거)
"""

from ultralytics import YOLO
import cv2
import numpy as np

class HighwayVehicleDetector:
    """고속도로 CCTV용 하이브리드 탐지기"""
    
    def __init__(self):
        # YOLO 모델
        self.yolo = YOLO("yolov8n.pt")
        
        # Motion Detection (KNN이 더 빠름)
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            detectShadows=False  # 그림자 무시 (속도 향상)
        )
        
        # 설정값 - 크기 제한 제거!
        self.min_area = 50    # 200 → 50 (아주 작은 것도)
        self.max_area = 50000  # 5000 → 50000 (아주 큰 것도)
        self.conf_threshold = 0.2  # YOLO 신뢰도 (매우 낮게)
        
        # 캐싱
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        self.frame_count = 0
        self.yolo_interval = 3  # 3프레임마다 YOLO
        self.last_yolo_detections = []
        
    def detect_motion(self, frame):
        """빠른 Motion Detection - 크기 제한 완화"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 배경 제거
        fg_mask = self.bg_subtractor.apply(gray)
        
        # 노이즈 제거
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        motion_detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 크기 제한 대폭 완화
            if area > self.min_area:  # 최소 크기만 체크
                x, y, w, h = cv2.boundingRect(cnt)
                
                # 종횡비 체크도 완화 (거의 모든 형태 허용)
                aspect_ratio = w / float(h) if h > 0 else 1
                if 0.1 < aspect_ratio < 10.0:  # 0.3~4.0 → 0.1~10.0
                    motion_detections.append([x, y, x+w, y+h, 0.5, 'motion'])
        
        return motion_detections
    
    def detect_yolo(self, frame):
        """YOLO 탐지 (낮은 신뢰도)"""
        results = self.yolo.predict(
            frame,
            conf=self.conf_threshold,
            classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
            verbose=False
        )
        
        yolo_detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # 클래스 이름 매핑
                    class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                    class_name = class_names.get(cls, 'vehicle')
                    
                    yolo_detections.append([x1, y1, x2, y2, conf, class_name])
        
        return yolo_detections
    
    def merge_detections(self, yolo_dets, motion_dets):
        """YOLO와 Motion 결과 병합"""
        merged = []
        used_motion = []
        
        # YOLO 결과 모두 추가 (높은 우선순위)
        for yolo_det in yolo_dets:
            merged.append(yolo_det[:5])  # x1,y1,x2,y2,conf
            
            # 겹치는 Motion 찾기
            for i, motion_det in enumerate(motion_dets):
                if self.calculate_iou(yolo_det[:4], motion_det[:4]) > 0.3:
                    used_motion.append(i)
        
        # 겹치지 않는 Motion 결과 추가
        for i, motion_det in enumerate(motion_dets):
            if i not in used_motion:
                # Motion 결과는 신뢰도 낮게
                merged.append([motion_det[0], motion_det[1], 
                             motion_det[2], motion_det[3], 0.4])
        
        return merged
    
    def calculate_iou(self, box1, box2):
        """IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - inter
        
        return inter / (union + 1e-6)
    
    def detect(self, frame):
        """통합 탐지 메서드"""
        self.frame_count += 1
        detections = []
        
        # 1. Motion Detection (매 프레임)
        motion_dets = self.detect_motion(frame)
        
        # 2. YOLO (주기적으로)
        if self.frame_count % self.yolo_interval == 0:
            self.last_yolo_detections = self.detect_yolo(frame)
        
        # 3. 결과 병합
        detections = self.merge_detections(
            self.last_yolo_detections, 
            motion_dets
        )
        
        return detections


# 전역 인스턴스 생성
detector = HighwayVehicleDetector()

# 기존 함수와 호환성 유지
def get_vehicle_detections(frame, conf_threshold=0.05, vehicle_classes=['car']):
    """
    기존 인터페이스 유지 (호환성)
    이제 YOLO + Motion 하이브리드로 작동
    """
    # conf_threshold 업데이트
    if conf_threshold != detector.conf_threshold:
        detector.conf_threshold = conf_threshold
    
    # 통합 탐지 실행
    return detector.detect(frame)


# 전역 인스턴스 생성
detector = HighwayVehicleDetector()

# 기존 함수와 호환성 유지
def get_vehicle_detections(frame, conf_threshold=0.05, vehicle_classes=['car']):
    """
    기존 인터페이스 유지 (호환성)
    이제 YOLO + Motion 하이브리드로 작동
    """
    # conf_threshold 업데이트
    if conf_threshold != detector.conf_threshold:
        detector.conf_threshold = conf_threshold
    
    # 통합 탐지 실행
    return detector.detect(frame)


# 테스트 코드
if __name__ == "__main__":
    import time
    
    # 테스트용 비디오 또는 스트림
    test_stream = "http://..."  # 여기에 실제 URL
    cap = cv2.VideoCapture(test_stream)
    
    frame_count = 0
    start_time = time.time()
    
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 탐지 실행
        detections = get_vehicle_detections(frame)
        
        # 결과 표시
        for det in detections:
            x1, y1, x2, y2, conf = det[:5]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Highway Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
        
        frame_count += 1
    
    # 성능 출력
    elapsed = time.time() - start_time
    print(f"FPS: {frame_count/elapsed:.1f}")
    
    cap.release()
    cv2.destroyAllWindows()