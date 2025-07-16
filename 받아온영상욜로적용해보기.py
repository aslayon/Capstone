from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 모델 로드 (자동으로 CUDA로 로드됨)
model = YOLO("yolov8s.pt")  # 또는 yolov8n.pt for speed

# 스트리밍 프레임 불러오기
stream_url = "##"  # 실제 CCTV URL
cap = cv2.VideoCapture(stream_url)
ret, frame = cap.read()
cap.release()

# YOLO 탐지
results = model.predict(frame, conf=0.5, classes=[2])  # class 2 = car

# bbox 시각화
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"car {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame_rgb)
plt.title("YOLO 차량 탐지 결과")
plt.axis("off")
plt.show()
