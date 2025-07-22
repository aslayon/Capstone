
import cv2
import os
import matplotlib.pyplot as plt
from detector.yolo_detector import get_vehicle_detections
from tracker.tracker_test import MultiTracker, check_boundary_event
import tracker.tracker_test as tracker_test
from handover.handover_logic import load_cctv_list
from ui.ui_utils import find_nearest_cctv_by_bbox  # ✅ 추가

tracker = MultiTracker()

def on_click(event):
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        tracker.select_track_by_point(x, y)

def main():
    stream_url = os.getenv("ITS_BASE_URL")
    current_cctv_name = os.getenv("CURRENT_CCTV_NAME")
    print(f"스트리밍 URL: {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    cctv_list = load_cctv_list()
    current_cctv = next((c for c in cctv_list if current_cctv_name in c["cctvname"]), None)
    if not current_cctv:
        print("❌ 현재 CCTV 이름을 cctv_list.json에서 찾을 수 없습니다.")
        return

    current_x = current_cctv["coordx"]
    current_y = current_cctv["coordy"]

    fig = plt.figure()
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 스트림 종료 또는 읽기 실패")
            break

        detections = get_vehicle_detections(frame, conf_threshold=0.2)
        tracks = tracker.update(detections)

        # 관심 차량 경계 접근 체크 및 기준 CCTV 전환
        bbox = tracker.get_selected_bbox()
        if bbox:
            h, w = frame.shape[:2]
            if check_boundary_event(bbox, w, h):
                new_cctv = find_nearest_cctv_by_bbox(bbox, cctv_list)
                if new_cctv and new_cctv["cctvname"] != current_cctv["cctvname"]:
                    print(f"[HANDOVER] 기준 CCTV 전환: {current_cctv['cctvname']} → {new_cctv['cctvname']}")
                    current_cctv = new_cctv
                    current_x = new_cctv["coordx"]
                    current_y = new_cctv["coordy"]
                    # 🔁 여기에 stream 전환 등 UI 재구성 연결 가능

        # 시각화
        for track_id, x1, y1, x2, y2 in tracks:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.title("Tracking Result - Click to Select Vehicle")
        plt.axis("off")
        plt.pause(0.01)
        plt.clf()

    cap.release()
    plt.close()

if __name__ == "__main__":
    main()
