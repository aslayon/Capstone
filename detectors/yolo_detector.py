# yolo_detector_커스텀.py
from ultralytics import YOLO
import cv2
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

# 기존
ROI = os.getenv("ROI_RECT", None)
if ROI:
    rx1, ry1, rx2, ry2 = map(int, ROI.split(","))
    ROI = (rx1, ry1, rx2, ry2)

# tri 모드용 ROI 추가
TRI_ROIS = {}
for name in ["LEFT", "CENTER", "RIGHT"]:
    val = os.getenv(f"TRI_ROI_{name}")
    if val:
        x1, y1, x2, y2 = map(int, val.split(","))
        TRI_ROIS[name.lower()] = (x1, y1, x2, y2)
# YOLO 모델 초기화
model = YOLO("yolo11l")  # 필요 시 yolov8n.pt, yolov8m.pt 등 교체

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

def get_vehicle_detections(frame, conf_threshold=0.5, vehicle_classes=('car','truck','bus'), roi=None, ignore_roi=False):
    vehicle_classes = list(vehicle_classes) if vehicle_classes is not None else ['car','truck','bus']
    """
    frame : 입력 프레임 (numpy array)
    roi   : 
        - None     → ROI_RECT (env) 사용
        - tuple    → 단일 ROI (x1,y1,x2,y2)
        - dict     → {"left":(x1,y1,x2,y2), "center":(...), "right":(...)} 형태 (tri 모드)
    ignore_roi : True면 ROI 무시 (전체 탐지)
    """
    # 1️⃣ 클래스 매핑
    class_ids = []
    for vehicle_class in vehicle_classes:
        if vehicle_class in COCO_CLASSES:
            class_ids.append(COCO_CLASSES[vehicle_class])
        else:
            print(f"⚠️ 알 수 없는 클래스: {vehicle_class}")
    if not class_ids:
        print("❌ 유효한 차량 클래스가 없습니다.")
        return []

    # 2️⃣ 여러 ROI(dict)일 경우
    if isinstance(roi, dict):
        detections = []
        for key, box in roi.items():
            if not box:
                continue
            x1, y1, x2, y2 = box
            roi_img = frame[y1:y2, x1:x2]
            results = model.predict(roi_img, conf=conf_threshold, classes=class_ids, verbose=False)
            id_to_name = {v: k for k, v in COCO_CLASSES.items()}

            for r in results:
                for b in r.boxes:
                    bx1, by1, bx2, by2 = map(int, b.xyxy[0])
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0])
                    cls_name = id_to_name.get(cls_id, 'unknown')

                    # ROI offset 복원
                    bx1 += x1; bx2 += x1
                    by1 += y1; by2 += y1
                    detections.append((bx1, by1, bx2, by2, conf, cls_name))
        return detections

    # 3️⃣ 단일 ROI(tuple) or 전체
    use_roi = ROI is not None and not ignore_roi and roi is None
    if use_roi:
        rx1, ry1, rx2, ry2 = ROI
        roi_img = frame[ry1:ry2, rx1:rx2]
        infer_img = roi_img
        offset = (rx1, ry1)
    elif isinstance(roi, tuple):
        rx1, ry1, rx2, ry2 = roi
        roi_img = frame[ry1:ry2, rx1:rx2]
        infer_img = roi_img
        offset = (rx1, ry1)
    else:
        infer_img = frame
        offset = (0, 0)

    # 4️⃣ YOLO 실행
    results = model.predict(infer_img, conf=conf_threshold, classes=class_ids, verbose=False)
    detections = []
    id_to_name = {v: k for k, v in COCO_CLASSES.items()}

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = id_to_name.get(class_id, 'unknown')

            # offset 적용
            x1 += offset[0]; x2 += offset[0]
            y1 += offset[1]; y2 += offset[1]

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
