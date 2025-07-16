from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. 모델 로드 (자동으로 다운로드됨)
model = YOLO("yolov8s.pt")  # 또는 yolov8n.pt (더 가볍고 빠름)

# 2. CCTV 스트림에서 프레임 1장만 읽기
stream_url = "http://cctvsec.ktict.co.kr/138/pQahsqagIvXoxtKYMYuTVxSWQPyEx4a/DycV69i2ghScblbPnSTRLT9ttd6K1vxfdzVH2B2WDjzDDFu8a5pSZocJ9jNGE5Bx51hdStrzVl0="  
cap = cv2.VideoCapture(stream_url)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ 프레임 읽기 실패")
    exit()

# 3. 차량 클래스만 탐지 (YOLO 클래스 ID: 2 = car)
results = model.predict(frame, conf=0.5, classes=[2])

# 4. 결과 시각화 (bbox 그리기)
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"car {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# BGR → RGB 변환 후 시각화
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame_rgb)
plt.title("YOLOv8 차량 탐지 결과")
plt.axis("off")
plt.show()
