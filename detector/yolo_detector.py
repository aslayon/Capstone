from ultralytics import YOLO
import torch
import cv2
import numpy as np

# YOLO 모델 초기화 (자동으로 GPU 사용됨)
model = YOLO("yolov8n")  # 필요에 따라 yolov8n.pt, yolov8m.pt 등 사용 가능

# COCO 클래스 매핑
COCO_CLASSES = {
    'car': 2,
    'truck': 7,
    'bus': 5,
    'motorcycle': 3
}

def get_vehicle_detections(frame, conf_threshold=0.5, vehicle_classes=['car']):
    """
    주어진 프레임에서 차량(bbox) 탐지 결과를 반환

    Args:
        frame (np.ndarray): 입력 이미지 (BGR)
        conf_threshold (float): confidence threshold
        vehicle_classes (list): 탐지할 차량 클래스 ['car', 'truck', 'bus', 'motorcycle']

    Returns:
        List[Tuple[int, int, int, int, float, str]]: [(x1, y1, x2, y2, confidence, class_name), ...]
    """
    # 클래스 이름을 COCO 클래스 ID로 변환
    class_ids = []
    for vehicle_class in vehicle_classes:
        if vehicle_class in COCO_CLASSES:
            class_ids.append(COCO_CLASSES[vehicle_class])
        else:
            print(f"⚠️ 알 수 없는 클래스: {vehicle_class}")
    
    if not class_ids:
        print("❌ 유효한 차량 클래스가 없습니다.")
        return []
    
    # YOLO 실행
    results = model.predict(frame, conf=conf_threshold, classes=class_ids, verbose=False)
    detections = []

    # 클래스 ID를 이름으로 매핑 (역방향)
    id_to_name = {v: k for k, v in COCO_CLASSES.items()}

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = id_to_name.get(class_id, 'unknown')
            
            detections.append((x1, y1, x2, y2, conf, class_name))

    return detections


# 테스트 코드 (모듈 단독 실행 시)
if __name__ == "__main__":
    stream_url = "http://cctvsec.ktict.co.kr/138/pQahsqagIvXoxtKYMYuTVxSWQPyEx4a/DycV69i2ghScblbPnSTRLT9ttd6K1vxfdzVH2B2WDjzDDFu8a5pSZocJ9jNGE5Bx51hdStrzVl0="
    cap = cv2.VideoCapture(stream_url)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # 차량과 트럭 모두 탐지
        dets = get_vehicle_detections(frame, vehicle_classes=['car', 'truck'])
        
        for (x1, y1, x2, y2, conf, class_name) in dets:
            # 클래스별로 다른 색상
            if class_name == 'car':
                color = (0, 255, 0)  # 초록색
            elif class_name == 'truck':
                color = (0, 0, 255)  # 빨간색
            else:
                color = (255, 0, 0)  # 파란색
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        import matplotlib.pyplot as plt
        plt.imshow(frame_rgb)
        plt.title("YOLO 차량+트럭 탐지 결과")
        plt.axis("off")
        plt.show()
    else:
        print("❌ 프레임을 불러올 수 없습니다.")