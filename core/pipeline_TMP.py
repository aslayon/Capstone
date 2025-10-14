# core/pipeline.py
#실행 루프
# 패치내역. 3캠화면 욜로 한번에 적용
import cv2
import time
import os
from core.config import load_config, parse_roi
from core.mouse_handler import MouseSelector
from core.cctv_graph import load_graph, load_cctv_list, get_neighbors, find_url_by_name
from core.tri_concat import concat_three
from core.roi_utils import shift_roi, bbox_center_in_any_roi

from core.window_utils import fit_window_to_image

from core.roi_utils import bbox_center_in_any_roi, tri_rois
from core.tri_concat import concat_three
from core.mouse_tri import TriClickHandler
from core.config import parse_roi, load_config
from core.tri_detect import split_dets_by_segment

from detectors.tracker_test import MultiTracker
from detectors.yolo_detector import get_vehicle_detections
from core.stream_manager import HLSStreamManager
from core.switch_controller import SwitchController
from core.bootstrap import refresh_initial_url
import os
import cv2
from collections import deque

# --- 전역 크롭 버퍼 ---
crop_buffer = deque(maxlen=5)

def save_selected_crop(frame, bbox, obj_id, save_dir="selected_crops"):
    """선택된 차량 bbox 이미지를 저장 (최근 5장 유지)"""
    os.makedirs(save_dir, exist_ok=True)

    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return  # 잘못된 bbox 방지

    filename = os.path.join(save_dir, f"id_{obj_id}_frame_{len(crop_buffer)}.jpg")
    cv2.imwrite(filename, crop)

    crop_buffer.append(filename)
    print(f"[CROP] saved: {filename}")



WIN = "Capstone - CCTV Tracking"

def run():
    cfg = load_config()
    tri_prepare = False
    tri_win = "Tri-Prepare (L/C/R -> 2160x480)"
    tri_mouse_ready = False
    tri_disp_scale = 1.0
    tri_total_w = 2160  # concat_three의 target W
    refreshed = refresh_initial_url()
    if refreshed:
        # 갱신된 환경으로 다시 로드
        cfg = load_config()
    
    current_url = cfg["CURRENT_CCTV_URL"]
    current_name = cfg["CURRENT_CCTV_NAME"]

    sm = HLSStreamManager(api_key=cfg["ITS_API_KEY"], update_interval=20)
    if not sm.start(current_name, current_url):
        print("❌ 스트림 시작 실패")
        return


    switcher = SwitchController(current_name, current_url, api_key=cfg["ITS_API_KEY"],
                                graph_path="data/cctv_graph_connections.json",
                                list_path="data/cctv_list_4.json", env_path=".env") 
    switcher.attach_center_manager(sm)
    
    tracker = MultiTracker(max_age=cfg["TRACKER_MAX_AGE"], iou_threshold=cfg["TRACKER_IOU_TH"])
    

    tracker_L = MultiTracker(max_age=cfg["TRACKER_MAX_AGE"], iou_threshold=cfg["TRACKER_IOU_TH"])
    tracker_R = MultiTracker(max_age=cfg["TRACKER_MAX_AGE"], iou_threshold=cfg["TRACKER_IOU_TH"])

    
    roi = parse_roi(cfg["ROI_RECT"])

    display_w, display_h = cfg["DISPLAY_W"], cfg["DISPLAY_H"]
    scale = 1.0
    frame_shape = [0,0]

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # 마우스 콜백
    def get_shape(): return (frame_shape[0], frame_shape[1])
    def get_scale(): return scale
    MouseSelector(WIN, tracker, get_shape, get_scale)






    import detectors.tracker_test as tt
    lost_count = 0
    LOST_N = 15
    tri_prepare = False
    tri_win = "Tri-Prepare (L/C/R)"
    tri_prepare = False
    tri_mouse_ready = False
    need_resize = True   # 최초 한 번만 리사이즈
    
    frame_idx = 0
    frame_count = 0
    last_time = time.time()
    while True:
        tri = None
        disp_tri = None
        switcher.tick()
        frame = switcher.center_sm.get_frame()
        disp = frame
        
        frame_idx += 1
        
        # frame 가져온 직후


        '''print("DEBUG neighbor urls:",
        "L=", switcher._find_url(switcher.left_name),           # 디버그
        "R=", switcher._find_url(switcher.right_name))'''

        
        if frame is None:
            #time.sleep(0.02)
            continue
        
        frame_count += 1
        now = time.time()

        # 1초마다 FPS 계산
        if now - last_time >= 1.0:
            fps = frame_count / (now - last_time)
            print(f"[DEBUG] FPS: {fps:.2f}")
            frame_count = 0
            last_time = now
        
        '''if frame is not None and switcher.center_sm.stats['frames_read'] % 30 == 0:

            
            # 저장 폴더 준비
            os.makedirs("debug_frames", exist_ok=True)

            # 파일 이름: cctv이름_프레임번호.jpg
            fname = f"debug_frames/{switcher.current_name}_{switcher.center_sm.stats['frames_read']}.jpg"

            # 이미지 저장
            cv2.imwrite(fname, frame)'''
        
        
        H, W = frame.shape[:2]
        frame_shape[0], frame_shape[1] = H, W
        if not tri_prepare:
            # 1) YOLO 탐지 + ROI 필터
            try:
                dets = get_vehicle_detections(frame, conf_threshold=cfg["DET_CONF"])
            except Exception as e:
                print("[YOLO ERROR]", e)
                dets = []
            dets = [d for d in dets if bbox_center_in_any_roi(d, [roi])]

            # 2) Tracker 업데이트
            dets_xyxy = [d[:4] for d in dets]
            tracks = tracker.update(dets_xyxy)

            # 선택 ID 미탐지 카운트
            if tracker.selected_id is not None and all(tid != tracker.selected_id for tid, *_ in tracks):
                lost_count += 1
            else:
                lost_count = 0

            
            # 3) 드로잉
            if roi:
                cv2.rectangle(frame, roi[:2], roi[2:], (255,255,0), 2)

            for x1,y1,x2,y2,conf,cls in dets:
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (140,140,140), 1)

            for tid,x1,y1,x2,y2 in tracks:
                color = (0,0,255) if (tracker.selected_id is not None and tid == tracker.selected_id) else (0,255,0)
                if tracker.selected_id is not None and tid == tracker.selected_id:
                    # 선택된 차량 bbox 저장
                    save_selected_crop(frame, (x1, y1, x2, y2), tid)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"ID {tid}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            info = f"Selected: {tracker.selected_id} | Lost: {lost_count}/{LOST_N} | 'p': tri-prepare={'ON' if tri_prepare else 'OFF'}"
            cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            hud = f"{switcher.center_sm.cctv_name} | updates:{switcher.center_sm.stats.get('url_updates',0)}"
            cv2.putText(frame, hud, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,255), 2)
            scale = min(display_w / W, display_h / H)
            disp = cv2.resize(frame, (int(W*scale), int(H*scale))) if scale != 1.0 else frame

        
            
            
        k = cv2.waitKey(1) & 0xFF
        
        
        if k in (27, ord('q')):
            break
        elif k == ord('p'):
            tri_prepare = not tri_prepare
            switcher.tri_mode = tri_prepare
            print("[DEBUG] tri-prepare:", tri_prepare)  # 디버그
            if tri_prepare:
                switcher.ensure_neighbor_managers()
            # 다음 프레임에서 창 크기 재조정
            need_resize = True
        else:
            switcher.on_key(k)
        # tri-prepare 확장 (추후 3캠 concat_three, ROI shift 활용)
        # === tri 모드 계산 (imshow 전에 먼저 만들기) ===
        if tri_prepare:
            lf = switcher.left_sm.get_frame()   if switcher.left_sm  else None
            cf = frame
            rf = switcher.right_sm.get_frame()  if switcher.right_sm else None
            tri = concat_three(lf, cf, rf, target=(2160, 480))

            # ROI 3분할
            cfg_now = load_config()
            def roi_of(name, fallback):
                if not name: return fallback
                s = cfg_now.get(f"ROI_{name}", "")
                return parse_roi(s) or fallback

            center_roi = roi
            left_roi   = roi_of(switcher.left_name,  center_roi)
            right_roi  = roi_of(switcher.right_name, center_roi)
            l_roi, c_roi, r_roi = tri_rois(center_roi, left_roi, right_roi, seg_w=720)

            # === (A) tri 프레임에 대해 YOLO 1회만 실행 ===
            try:
                dets_tri = get_vehicle_detections(tri, conf_threshold=cfg["DET_CONF"])
            except Exception as e:
                print("[YOLO ERROR on tri]", e)
                dets_tri = []

            # === (B) 좌/중/우로 분배 (세그먼트 로컬좌표로 변환) ===
            seg_w = 720
            split = split_dets_by_segment(dets_tri, seg_w=seg_w)
            dets_L, dets_C, dets_R = split["L"], split["C"], split["R"]

            # === (C) 세그먼트별 ROI 필터 적용 (각 det는 해당 세그먼트 로컬 좌표) ===
            if left_roi:   dets_L = [d for d in dets_L if bbox_center_in_any_roi(d, [left_roi])]
            if center_roi: dets_C = [d for d in dets_C if bbox_center_in_any_roi(d, [center_roi])]
            if right_roi:  dets_R = [d for d in dets_R if bbox_center_in_any_roi(d, [right_roi])]

            # === (D) 각 트래커 업데이트 ===
            tracks_L = tracker_L.update([d[:4] for d in dets_L])
            tracks_C = tracker.update   ([d[:4] for d in dets_C])  # 중앙은 기존 tracker 재사용
            tracks_R = tracker_R.update([d[:4] for d in dets_R])

            # === (E) tri 프레임 위에 드로잉 (회색: YOLO, 초록/빨강: 트랙) ===
            def draw_dets_on_tri(img, dets, xoff, color=(140,140,140)):
                for x1,y1,x2,y2,conf,cls in dets:
                    X1, X2 = int(x1 + xoff), int(x2 + xoff)
                    cv2.rectangle(img, (X1, int(y1)), (X2, int(y2)), color, 1)

            def draw_tracks_on_tri(img, tracks, xoff, sel_id=None):
                for tid, x1, y1, x2, y2 in tracks:
                    X1, X2 = int(x1 + xoff), int(x2 + xoff)
                    is_sel = (sel_id is not None and tid == sel_id)
                    color = (0,0,255) if is_sel else (0,255,0)
                    cv2.rectangle(img, (X1, int(y1)), (X2, int(y2)), color, 2)
                    cv2.putText(img, f"ID {tid}", (X1, int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            # 좌/중/우 YOLO 박스
            draw_dets_on_tri(tri, dets_L, 0)
            draw_dets_on_tri(tri, dets_C, seg_w)
            draw_dets_on_tri(tri, dets_R, seg_w*2)

            # 좌/중/우 트랙 박스 (중앙은 선택ID 반영)
            draw_tracks_on_tri(tri, tracks_L, 0,       sel_id=None)
            draw_tracks_on_tri(tri, tracks_C, seg_w,   sel_id=tracker.selected_id)
            draw_tracks_on_tri(tri, tracks_R, seg_w*2, sel_id=None)

            
            for r, color in [(l_roi,(100,255,100)), (c_roi,(255,255,0)), (r_roi,(100,255,255))]:
                if r: cv2.rectangle(tri, (r[0], r[1]), (r[2], r[3]), color, 2)

            # 표시용 스케일
            Ht, Wt = tri.shape[:2]
            max_w = 1600
            tri_disp_scale = max_w / Wt if Wt > max_w else 1.0
            disp_tri = cv2.resize(tri, (int(Wt*tri_disp_scale), int(Ht*tri_disp_scale))) if tri_disp_scale != 1.0 else tri

            # ✅ tri 전용 창 생성 금지 (아래 라인 삭제!)
            # cv2.namedWindow(tri_win, cv2.WINDOW_NORMAL)

            # tri 클릭 콜백을 메인 창(WIN)에 1회만
            if not tri_mouse_ready:
                TriClickHandler(
                    WIN,
                    switcher,
                    get_disp_scale=lambda: tri_disp_scale,
                    get_total_w_callable=lambda: Wt
                )
                tri_mouse_ready = True
            # tri 모드에서 선택 ID 로스트 체크 (중앙 트래커 기준)
            if tracker.selected_id is not None and all(tid != tracker.selected_id for tid, *_ in tracks_C):
                lost_count += 1
            else:
                lost_count = 0

        
        # === 출력 프레임 결정 ===
        out_img = disp_tri if (tri_prepare and disp_tri is not None) else disp
        cv2.imshow(WIN, out_img)

        # === 필요할 때만 창 크기 자동 맞춤 ===
        if need_resize and out_img is not None:
            fit_window_to_image(WIN, out_img)
            need_resize = False

    sm.stop()
    cv2.destroyAllWindows()
