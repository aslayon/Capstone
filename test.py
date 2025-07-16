import cv2
from detector.yolo_detector import get_vehicle_detections
from tracker.tracker_test import MultiTracker
import matplotlib.pyplot as plt


def main():
    stream_url = "##"  # 실제 CCTV 주소로 교체
    cap = cv2.VideoCapture(stream_url)
    tracker = MultiTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 스트림 종료 또는 읽기 실패")
            break

        # 차량 bbox 탐지
        detections = get_vehicle_detections(frame, conf_threshold=0.5)

        # 트래킹 ID 업데이트
        tracks = tracker.update(detections)

        # 시각화
        for track_id, x1, y1, x2, y2 in tracks:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # matplotlib로 출력 (imshow는 PowerShell에서 불안정할 수 있음)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.title("Tracking Result")
        plt.axis("off")
        plt.pause(0.01)
        plt.clf()

    cap.release()
    plt.close()


if __name__ == "__main__":
    main()
