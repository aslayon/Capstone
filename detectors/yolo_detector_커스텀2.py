# yolo_detector.py
from ultralytics import YOLO
import cv2
import numpy as np
import os

# =========================================
# 설정: 학습된 가중치 경로 (7 클래스 모델)
# =========================================
MODEL_PATH = os.getenv("VEH_WEIGHTS", r"best_1cls.pt")

# ====== [NEW] ROI 설정 ======
ROI = os.getenv("ROI_RECT", None)
if ROI:
    rx1, ry1, rx2, ry2 = map(int, ROI.split(","))
    ROI = (rx1, ry1, rx2, ry2)
else:
    ROI = (200, 120, 700, 430)   # 기본값 (원하는 값으로 수정)
# ============================

_model = None
def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def get_vehicle_detections(frame, conf_threshold=0.2):
    """
    ROI가 설정돼 있으면 해당 영역만 크롭해서 탐지,
    bbox 좌표는 원본 프레임 기준으로 복원.
    Returns: [(x1,y1,x2,y2,conf,class_name), ...]
    """
    model = get_model()
    use_roi = ROI is not None

    if use_roi:
        rx1, ry1, rx2, ry2 = ROI
        roi_img = frame[ry1:ry2, rx1:rx2]
        infer_img = roi_img
    else:
        infer_img = frame

    results = model.predict(infer_img, conf=conf_threshold, iou=0.55, verbose=False)
    detections = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])        # 클래스 인덱스
            cls_name = model.names[cls_id]  # 예: "car-01" ~ "car-07"

            # ROI offset 복원
            if use_roi:
                x1 += rx1; x2 += rx1
                y1 += ry1; y2 += ry1

            detections.append((x1, y1, x2, y2, conf, cls_name))

    return detections


# ===== 테스트 코드 =====
if __name__ == "__main__":
    cap = cv2.VideoCapture("your_stream.m3u8")
    ok, frame = cap.read()
    cap.release()

    if ok:
        dets = get_vehicle_detections(frame, conf_threshold=0.4)
        for (x1, y1, x2, y2, conf, cls_name) in dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("Detections", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ 프레임을 불러올 수 없습니다.")
