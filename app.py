# app.py
# 메인 실행: 단일 CCTV + 클릭 선택 강조 + ‘p’ 준비 신호
import os
import cv2
import time

from config import load_config, parse_roi, save_current_cctv_url
from cctv_graph import load_graph, load_cctv_list, get_neighbors, find_url_by_name
from tri_concat import concat_three
from mouse_handler import MouseSelector

from tracker_test import MultiTracker  # 재사용
from yolo_detector import get_vehicle_detections  # 재사용

WIN = "Capstone - Single CCTV (click: select, 'p': tri-prepare, 'q/ESC': quit)"

def open_cap(src):
    try:
        src_int = int(src)
        cap = cv2.VideoCapture(src_int)
    except:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")
    return cap

def filter_by_roi(dets, roi):
    if not roi:
        return dets
    x1r, y1r, x2r, y2r = roi
    out = []
    for x1,y1,x2,y2,conf,cls in dets:
        cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
        if x1r <= cx <= x2r and y1r <= cy <= y2r:
            out.append([x1,y1,x2,y2,conf,cls])
    return out

def main():
    cfg = load_config()
    current_url = cfg["CURRENT_CCTV_URL"]
    current_name = cfg["CURRENT_CCTV_NAME"]

    # 실행 시점에 CURRENT_CCTV_URL 자동 갱신 (가능하면 cctv_list_4.json으로)
    cctv_list = load_cctv_list("cctv_list_4.json")
    if current_name and cctv_list:
        found = find_url_by_name(cctv_list, current_name)
        if found and found != current_url:
            print(f"[INFO] CURRENT_CCTV_URL 갱신: {current_url} -> {found}")
            save_current_cctv_url(found)
            current_url = found

    cap = open_cap(current_url)
    tracker = MultiTracker(max_age=cfg["TRACKER_MAX_AGE"], iou_threshold=cfg["TRACKER_IOU_TH"])
    roi = parse_roi(cfg["ROI_RECT"])

    # 화면 표시 스케일 관리
    display_w, display_h = cfg["DISPLAY_W"], cfg["DISPLAY_H"]
    scale = 1.0
    frame_shape = [0,0]

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # 마우스 콜백 분리
    def get_shape(): return (frame_shape[0], frame_shape[1])
    def get_scale(): return scale
    MouseSelector(WIN, tracker, get_shape, get_scale)

    # 선택 ID의 미탐지 카운터
    import tracker_test as tt  # selected_id 공유
    lost_count = 0
    LOST_N = 15  # n프레임 연속 미탐지 기준

    graph = load_graph("cctv_graph_connections.json")  # 이후 tri-prepare 때 사용

    tri_prepare = False  # 'p' 토글
    left_cap = right_cap = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue

        H, W = frame.shape[:2]
        frame_shape[0], frame_shape[1] = H, W

        # 1) YOLO → ROI 필터
        try:
            dets = get_vehicle_detections(frame, conf_threshold=cfg["DET_CONF"])
        except Exception as e:
            print("[YOLO ERROR]", e)
            dets = []
        dets = filter_by_roi(dets, roi)

        # 2) Tracker अपडेट
        dets_xyxy = [d[:4] for d in dets]
        tracks = tracker.update(dets_xyxy)  # (id,x1,y1,x2,y2) 리스트

        # 선택 ID 미탐지 체크
        if tt.selected_id is not None and all(tid != tt.selected_id for tid, *_ in tracks):
            lost_count += 1
        else:
            lost_count = 0

        # 3) 드로잉
        # ROI 박스
        if roi:
            x1r,y1r,x2r,y2r = roi
            cv2.rectangle(frame, (x1r,y1r), (x2r,y2r), (255,255,0), 2)
            cv2.putText(frame, "ROI", (x1r, max(0,y1r-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # YOLO 박스 (연한 회색)
        for x1,y1,x2,y2,conf,cls in dets:
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (140,140,140), 1)

        # 트랙 박스
        from tracker_test import selected_id as SID
        for tid,x1,y1,x2,y2 in tracks:
            color = (0,0,255) if (SID is not None and tid == SID) else (0,255,0)
            thick = 3 if color==(0,0,255) else 2
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, thick)
            cv2.putText(frame, f"ID {tid}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 오버레이 정보
        info = f"Selected: {SID} | Lost: {lost_count}/{LOST_N} | 'p': tri-prepare={'ON' if tri_prepare else 'OFF'}"
        cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # 디스플레이 리사이즈
        scale = min(display_w / W, display_h / H)
        disp = cv2.resize(frame, (int(W*scale), int(H*scale))) if scale != 1.0 else frame

        cv2.imshow(WIN, disp)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break
        elif k == ord('p'):
            tri_prepare = not tri_prepare
            print("[DEBUG] tri-prepare:", tri_prepare)

        # ======= Tri-prepare (디버그 모드에서만, 선택ID Lost 상태일 때만) =======
        if cfg["DEBUG_MODE"] and tri_prepare and SID is not None and lost_count >= LOST_N:
            neighbors = get_neighbors(graph, current_name)  # {"left": name, "right": name}
            left_name, right_name = neighbors["left"], neighbors["right"]

            # 이웃 URL 조회 (cctv_list_4.json 기반; 없으면 스킵)
            left_url = find_url_by_name(cctv_list, left_name) if left_name else None
            right_url = find_url_by_name(cctv_list, right_name) if right_name else None

            # 이웃 캡처 준비 (필요 시 생성)
            if left_url and (left_cap is None):
                try: left_cap = open_cap(left_url)
                except: left_cap = None
            if right_url and (right_cap is None):
                try: right_cap = open_cap(right_url)
                except: right_cap = None

            # 프레임 읽기
            l_ok, l_fr = (left_cap.read() if left_cap else (False, None))
            c_ok, c_fr = True, frame
            r_ok, r_fr = (right_cap.read() if right_cap else (False, None))

            tri = concat_three(l_fr if l_ok else None, c_fr if c_ok else None, r_fr if r_ok else None, target=(2160,480))
            cv2.imshow("Tri-Prepare (L/C/R -> 2160x480)", tri)
            cv2.waitKey(1)
            # NOTE: 이후 단계에서 이 tri 프레임을 그대로 YOLO에 넣으면 됨(ROI는 각 영역 동일 좌표 적용)
            #       오늘은 여기까지 준비만: 결합/프리뷰/루틴 토글.
            #       Re-ID는 다음 단계에서.

    # 종료 정리
    if left_cap: left_cap.release()
    if right_cap: right_cap.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
