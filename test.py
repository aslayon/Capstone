import cv2
import os
import matplotlib.pyplot as plt
from detector.yolo_detector import get_vehicle_detections
from tracker.tracker_test import MultiTracker, check_boundary_event
import tracker.tracker_test as tracker_test
from handover.handover_logic import load_cctv_list, find_adjacent_cctvs

# 전역 변수
tracker = MultiTracker()  # 전역으로 선언해야 on_click에서도 접근 가능

def on_click(event):
    if event.inaxes:  # 플롯 내부에서 클릭했을 때만
        x, y = int(event.xdata), int(event.ydata)
        tracker.select_track_by_point(x, y)  # 관심 차량 선택

def main():
    stream_url = os.getenv("ITS_BASE_URL")
    current_cctv_name = os.getenv("CURRENT_CCTV_NAME")  # 예: "지본교"
    print(f"스트리밍 URL: {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    # 현재 CCTV 위치 로드
    cctv_list = load_cctv_list()
    current_cctv = next((c for c in cctv_list if current_cctv_name in c["cctvname"]), None)
    if not current_cctv:
        print("❌ 현재 CCTV 이름을 cctv_list.json에서 찾을 수 없습니다.")
        return

    current_x = current_cctv["coordx"]
    current_y = current_cctv["coordy"]

    # 클릭 이벤트 등록
    fig = plt.figure()
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 스트림 종료 또는 읽기 실패")
            break

        detections = get_vehicle_detections(frame, conf_threshold=0.2) #임계 신뢰도
        tracks = tracker.update(detections)

        # 관심 차량 경계 접근 체크
        for track in tracker.tracks:
            if track.id == tracker_test.selected_id:
                bbox = track.get_bbox()
                h, w = frame.shape[:2]
                if check_boundary_event(bbox, w, h):
                    print(f"[EVENT] 관심 차량 ID {track.id} 화면 경계 접근!")
                    next_cams = find_adjacent_cctvs(current_x, current_y, cctv_list, direction="right")
                    if next_cams:
                        print("[HANDOVER] 다음 CCTV 후보:")
                        for cam in next_cams:
                            print(f"- {cam['cctvname']} ({cam['coordx']:.5f}, {cam['coordy']:.5f})")

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
