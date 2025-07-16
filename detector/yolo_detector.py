from ultralytics import YOLO
import torch
import cv2

# YOLO 모델 초기화 (자동으로 GPU 사용됨)
model = YOLO("yolov8s.pt")  # 필요에 따라 yolov8n.pt, yolov8m.pt 등 사용 가능


def get_vehicle_detections(frame, conf_threshold=0.5):
    """
    주어진 프레임에서 차량(bbox) 탐지 결과를 반환

    Args:
        frame (np.ndarray): 입력 이미지 (BGR)
        conf_threshold (float): confidence threshold

    Returns:
        List[Tuple[int, int, int, int, float]]: [(x1, y1, x2, y2, confidence), ...]
    """
    results = model.predict(frame, conf=conf_threshold, classes=[2], verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append((x1, y1, x2, y2, conf))

    return detections


# 테스트 코드 (모듈 단독 실행 시)
if __name__ == "__main__":
    stream_url = "http://cctvsec.ktict.co.kr/138/pQahsqagIvXoxtKYMYuTVxSWQPyEx4a/DycV69i2ghScblbPnSTRLT9ttd6K1vxfdzVH2B2WDjzDDFu8a5pSZocJ9jNGE5Bx51hdStrzVl0="  # 
    cap = cv2.VideoCapture(stream_url)
    ret, frame = cap.read()
    cap.release()

    if ret:
        dets = get_vehicle_detections(frame)
        for (x1, y1, x2, y2, conf) in dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"car {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        import matplotlib.pyplot as plt
        plt.imshow(frame_rgb)
        plt.title("YOLO 탐지 결과")
        plt.axis("off")
        plt.show()
    else:
        print("❌ 프레임을 불러올 수 없습니다.")
