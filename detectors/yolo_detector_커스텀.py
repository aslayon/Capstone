# yolo_detector_커스텀.py
from ultralytics import YOLO
import cv2
import numpy as np
import os

# YOLO 모델 초기화
model = YOLO("yolov11l")  # 필요 시 yolov8n.pt, yolov8m.pt 등 교체

# COCO 클래스 매핑
COCO_CLASSES = {
    'car': 2,
    'truck': 7,
    'bus': 5,
    'motorcycle': 3
}

# ====== [NEW] ROI 설정 ======
ROI = os.getenv("ROI_RECT", None)
if ROI:
    rx1, ry1, rx2, ry2 = map(int, ROI.split(","))
    ROI = (rx1, ry1, rx2, ry2)
else:
    ROI = (200, 120, 700, 430)  # 기본값
# ============================

def get_vehicle_detections(frame, conf_threshold=0.5, vehicle_classes=['car']):
    """
    ROI가 설정돼 있으면 해당 영역만 크롭해서 탐지,
    결과 bbox는 원본 프레임 좌표계로 복원
    """
    # 클래스 ID 리스트 생성
    class_ids = []
    for vehicle_class in vehicle_classes:
        if vehicle_class in COCO_CLASSES:
            class_ids.append(COCO_CLASSES[vehicle_class])
        else:
            print(f"⚠️ 알 수 없는 클래스: {vehicle_class}")
    
    if not class_ids:
        print("❌ 유효한 차량 클래스가 없습니다.")
        return []

    # ROI 적용
    use_roi = ROI is not None
    if use_roi:
        rx1, ry1, rx2, ry2 = ROI
        roi_img = frame[ry1:ry2, rx1:rx2]
        infer_img = roi_img
    else:
        infer_img = frame

    results = model.predict(infer_img, conf=conf_threshold, classes=class_ids, verbose=False)
    detections = []

    # 역매핑: id → 이름
    id_to_name = {v: k for k, v in COCO_CLASSES.items()}

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = id_to_name.get(class_id, 'unknown')

            # ROI offset 복원
            if use_roi:
                x1 += rx1; x2 += rx1
                y1 += ry1; y2 += ry1

            detections.append((x1, y1, x2, y2, conf, class_name))

    return detections


# 테스트 코드
if __name__ == "__main__":
    cap = cv2.VideoCapture("your_stream.m3u8")
    ok, frame = cap.read()
    cap.release()

    if ok:
        dets = get_vehicle_detections(frame, vehicle_classes=['car', 'truck'])
        for (x1, y1, x2, y2, conf, cls_name) in dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Detections", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
