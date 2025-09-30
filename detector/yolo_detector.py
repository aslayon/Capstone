# yolo_detector.py  (학습 가중치 + 단일 클래스 'vehicle' 버전)
from ultralytics import YOLO
import cv2
import numpy as np
import os

# =========================================
# 설정: 학습된 가중치 경로 (필요 시 경로 수정)
# 예) E:\vehset_work\veh_yolo\weights\best.pt
# =========================================
MODEL_PATH = os.getenv("VEH_WEIGHTS", r"C:\GIT\best.pt")

# 모델은 모듈 임포트 시 1회만 로드 (성능)
_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def get_vehicle_detections(frame, conf_threshold=0.2):
    """
    단일 클래스('vehicle')로 학습된 모델 기준 탐지 결과 반환
    Args:
        frame (np.ndarray): BGR 이미지
        conf_threshold (float): confidence threshold
    Returns:
        List[Tuple[int,int,int,int,float,str]]: [(x1,y1,x2,y2,conf,'vehicle'), ...]
    """
    model = get_model()

    # 단일 클래스 모델이므로 classes=None (필터링 불필요)
    results = model.predict(frame, conf=0.30, iou=0.55, verbose=False)

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            # 학습 데이터가 1클래스이므로 항상 'vehicle'
            detections.append((x1, y1, x2, y2, conf, "vehicle"))
    return detections


# 테스트 실행(선택)
if __name__ == "__main__":
    stream_url = "https://cctvsec.ktict.co.kr/138/pQahsqagIvXoxtKYMYuTVxSWQPyEx4a/DycV69i2ghScblbPnSTRLT9ttd6K1vxfMPPuDRFosHtS9hrOw9UBx2pvvHIkU2kUsC9LaRMvXaQ="  # 필요 시 교체
    cap = cv2.VideoCapture(stream_url)
    ok, frame = cap.read()
    cap.release()

    if ok:
        dets = get_vehicle_detections(frame, conf_threshold=0.4)
        for x1, y1, x2, y2, conf, cls_name in dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(
                frame,
                f"{cls_name} {conf:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
        import matplotlib.pyplot as plt

        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Vehicle detections (single-class)")
        plt.axis("off")
        plt.show()
    else:
        print("❌ 프레임을 불러올 수 없습니다.")
