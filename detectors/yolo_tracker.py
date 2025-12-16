# detectors/yolo_tracker.py
"""
YOLO 내장 track() 메서드를 사용한 간단한 탐지+추적 통합
기존 MultiTracker를 완전히 대체
"""
from ultralytics import YOLO
import numpy as np

class YOLOTracker:
    def __init__(self, 
                 model_path="yolo11n.pt",
                 conf_threshold=0.2,
                 iou_threshold=0.7,
                 tracker_config="bytetrack.yaml",  # 또는 "botsort.yaml"
                 persist=True):
        """
        Args:
            model_path: YOLO 모델 경로
            conf_threshold: 탐지 신뢰도 임계값
            iou_threshold: NMS IOU 임계값
            tracker_config: 트래커 설정 ("bytetrack.yaml" 또는 "botsort.yaml")
            persist: 트래킹 ID 유지 여부
        """
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.tracker = tracker_config
        self.persist = persist
        self.selected_id = None
        self.selection_anchor_id = None  # 최초 클릭한 ID 유지용
        self.last_track_confidences = {}  # 최근 track() 호출에서의 confidence

        # 차량 클래스 (COCO dataset)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
    def update(self, frame, roi=None):
        """
        track() 메서드로 탐지+추적 동시 수행
        
        Args:
            frame: 입력 프레임 (numpy array)
            roi: ROI (x1, y1, x2, y2) 또는 None
            
        Returns:
            tracks: [(track_id, x1, y1, x2, y2), ...]
        """
        # YOLO track 실행
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker,
            persist=self.persist,
            classes=self.vehicle_classes,
            verbose=False
        )
        
        tracks = []
        track_confidences = {}
        if results and len(results) > 0:
            result = results[0]
            
            # track ID가 있는 경우만 처리
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confidences = None
                if getattr(result.boxes, "conf", None) is not None:
                    confidences = result.boxes.conf.cpu().numpy()
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    track_id = track_ids[i]
                    conf = float(confidences[i]) if confidences is not None else None
                    
                    # ROI 필터링
                    if roi is not None:
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        rx1, ry1, rx2, ry2 = roi
                        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                            continue
                    
                    tracks.append((track_id, x1, y1, x2, y2))
                    if conf is not None:
                        track_confidences[track_id] = conf

        self.last_track_confidences = track_confidences
        
        return tracks
    
    def detect_only(self, frame, roi=None, classes=None):
        """
        추적 없이 탐지만 수행 (디버깅용)
        
        Args:
            frame: 입력 프레임
            roi: ROI (x1, y1, x2, y2) 또는 None
            classes: 탐지할 클래스 (None이면 vehicle_classes 사용)
            
        Returns:
            detections: [(x1, y1, x2, y2, conf, cls), ...]
        """
        if classes is None:
            classes = self.vehicle_classes
            
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=classes,
            verbose=False
        )
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = confidences[i]
                cls = class_ids[i]
                
                # ROI 필터링
                if roi is not None:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    rx1, ry1, rx2, ry2 = roi
                    if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                        continue
                
                detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))
        
        return detections
    
    def reset(self):
        """트래커 리셋 (카메라 전환 시)"""
        self.persist = False
        self.model.predictor.trackers = []
        self.persist = True
        self.selected_id = None
        self.selection_anchor_id = None
        self.last_track_confidences = {}
        print("[INFO] 트래커 리셋")


# ===== 기존 코드 호환성을 위한 래퍼 함수 =====
def get_vehicle_detections(frame, conf_threshold=0.25, roi=None, ignore_roi=False):
    """
    기존 코드 호환용 - 탐지만 수행
    """
    global _global_yolo
    if '_global_yolo' not in globals():
        _global_yolo = YOLO("yolo11n.pt")
    
    results = _global_yolo.predict(
        frame,
        conf=conf_threshold,
        classes=[2, 3, 5, 7],
        verbose=False
    )
    
    dets = []
    if results and len(results) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = confs[i]
            cls = classes[i]
            dets.append((float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)))
    
    return dets
