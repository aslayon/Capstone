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
from core.reid_bank import ReIDBank

import numpy as np
from core.roi_utils import bbox_center_in_any_roi, tri_rois
from core.tri_concat import concat_three
from core.mouse_tri import TriClickHandler
from core.config import parse_roi, load_config
from core.tri_detect import split_dets_by_segment

from core.frame_bus import BUS

from detectors.tracker_test import MultiTracker
from detectors.yolo_detector import get_vehicle_detections
from core.stream_manager import HLSStreamManager
from core.switch_controller import SwitchController
from core.bootstrap import refresh_initial_url
WIN = "Capstone - CCTV Tracking"

tri_selected = {"seg": "C", "id": None}  # seg in {"L","C","R"}

tri_selected = {"seg": "C", "id": None}  # seg in {"L","C","R"}

def install_tri_selector(win_name, get_scale, get_seg_w, get_disp_img,
                         get_tracks_L, get_tracks_C, get_tracks_R):
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        img = get_disp_img()
        if img is None:
            return
        scale = get_scale() or 1.0
        seg_w_disp = int(get_seg_w() * scale)  # 화면에 표시된 세그먼트 폭
        # 클릭 x로 세그먼트 판별
        if x < seg_w_disp:
            seg = "L"; tracks = get_tracks_L()
            xoff = 0
        elif x < seg_w_disp*2:
            seg = "C"; tracks = get_tracks_C()
            xoff = seg_w_disp
        else:
            seg = "R"; tracks = get_tracks_R()
            xoff = seg_w_disp*2

        # 클릭 좌표를 해당 세그먼트 로컬좌표로 환산
        lx = int((x - xoff) / scale)
        ly = int(y / scale)

        # 가장 가까운 트랙 찾기
        best = (None, 1e18)
        for tid, x1, y1, x2, y2 in (tracks or []):
            cx = (x1 + x2)//2; cy = (y1 + y2)//2
            d2 = (cx - lx)**2 + (cy - ly)**2
            if d2 < best[1]:
                best = (tid, d2)

        if best[0] is not None:
            tri_selected["seg"] = seg
            tri_selected["id"] = best[0]
            print(f"[SELECT] tri {seg} -> ID {best[0]}")
    cv2.setMouseCallback(win_name, on_mouse)
   
def run_detect():
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


    selected_bank = ReIDBank(maxlen=10, h_bins=25, s_bins=30)
    collect_every = 5  # 선택ID 갤러리 수집 주기(프레임)
    REID_THRESH = 0.45 # 유사도 임계값(환경에 따라 0.40~0.50 조정)


    remap_left, remap_right = {}, {}

    import detectors.tracker_test as tt
    lost_count = 0
    LOST_N = 15
    tri_prepare = False
    tri_win = "Tri-Prepare (L/C/R)"
    tri_prepare = False
    tri_mouse_ready = False
    need_resize = True   # 최초 한 번만 리사이즈
    
    frame_idx = 0
    left_roi = None
    center_roi = None
    right_roi = None

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
                if tid == tracker.selected_id:
                    if frame_idx % collect_every == 0:
                        # tri 프레임이 아닌 '센터 원본 frame' 기준으로 수집 권장
                        selected_bank.add_from_frame(frame, (x1,y1,x2,y2))
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
        # === TRI MODE ===
        if tri_prepare:
            # 0) 좌/중/우 프레임 확보 (없으면 스킵)
            lf = switcher.left_sm.get_frame()  if switcher.left_sm  else None
            cf = frame  # 센터는 방금 읽은 프레임
            rf = switcher.right_sm.get_frame() if switcher.right_sm else None
            if lf is None or cf is None or rf is None:
                disp_tri = None
            else:
                # 1) 크기 맞추고 가로로 concat (2160x480 같은 고정 사이즈 권장)
                H, W = cf.shape[:2]
                target_h = 480
                target_w = 720
                def fit(x):
                    return cv2.resize(x, (target_w, target_h))
                lf_s, cf_s, rf_s = fit(lf), fit(cf), fit(rf)
                tri = np.hstack([lf_s, cf_s, rf_s])  # (H=480, W=2160)
                seg_w = target_w

                
                # 2) tri 프레임에 YOLO 1회
                try:
                    dets_tri = get_vehicle_detections(
                        tri,
                        conf_threshold=cfg["DET_CONF"],
                        roi=tri_rois,  #  tri ROI 세트 전달
                        ignore_roi=False
                    )
                except Exception as e:
                    print("[YOLO ERROR on tri]", e)
                    dets_tri = []


                # 3) 좌/중/우로 분배 (센터/우는 x 오프셋 제거하여 로컬 좌표로)
                def split_dets_by_segment(dets, seg_w):
                    out = {"L": [], "C": [], "R": []}
                    for x1, y1, x2, y2, conf, cls in dets:
                        cx = 0.5 * (x1 + x2)
                        if cx < seg_w:
                            out["L"].append((x1, y1, x2, y2, conf, cls))
                        elif cx < 2*seg_w:
                            out["C"].append((x1 - seg_w, y1, x2 - seg_w, y2, conf, cls))
                        else:
                            out["R"].append((x1 - 2*seg_w, y1, x2 - 2*seg_w, y2, conf, cls))
                    return out

                split = split_dets_by_segment(dets_tri, seg_w)
                dets_L, dets_C, dets_R = split["L"], split["C"], split["R"]

                # 4) 세그먼트별 ROI 필터 (각각 로컬 좌표임!)
                if left_roi:   dets_L = [d for d in dets_L if bbox_center_in_any_roi(d, [left_roi])]
                if center_roi: dets_C = [d for d in dets_C if bbox_center_in_any_roi(d, [center_roi])]
                if right_roi:  dets_R = [d for d in dets_R if bbox_center_in_any_roi(d, [right_roi])]

                # 5) 각 트래커 업데이트 (트래커는 로컬 좌표 전달!)
                tracks_L = tracker_L.update([d[:4] for d in dets_L])
                tracks_C = tracker.update   ([d[:4] for d in dets_C])  # 센터는 기존 tracker 재사용
                tracks_R = tracker_R.update([d[:4] for d in dets_R])

                remap_left, remap_center, remap_right = {}, {}, {}

                # === TRI 전용: 현재 3화면 내부 후보만 색상 유사도 비교 ===
                if selected_bank.items: 
                    def best_match_in_segment(seg_frame, tracks):
                        best = (None, 1e9, None)  # (tid, dist, (x1,y1,x2,y2))
                        for tid, x1, y1, x2, y2 in tracks:
                            d = selected_bank.score_to_gallery(seg_frame, (x1,y1,x2,y2))
                            if d is not None and d < best[1]:
                                best = (tid, d, (x1,y1,x2,y2))
                        return best

                    tidL, dL, boxL = best_match_in_segment(lf_s, tracks_L)
                    tidC, dC, boxC = best_match_in_segment(cf_s, tracks_C)
                    tidR, dR, boxR = best_match_in_segment(rf_s, tracks_R)

                    candidates = []
                    if tidL is not None: candidates.append(("L", tidL, dL, boxL))
                    if tidC is not None: candidates.append(("C", tidC, dC, boxC))
                    if tidR is not None: candidates.append(("R", tidR, dR, boxR))

                    if candidates:
                        seg_best, tid_best, dist_best, _ = min(candidates, key=lambda x: x[2])
                        if dist_best < REID_THRESH:
                            if seg_best == "L":
                                remap_left[tid_best]   = tracker.selected_id or tid_best
                            elif seg_best == "C":
                                remap_center[tid_best] = tracker.selected_id or tid_best
                            else:
                                remap_right[tid_best]  = tracker.selected_id or tid_best
                            # 디버그
                            # print(f"[ReID-tri] best={seg_best} tid={tid_best} dist={dist_best:.3f}")

                
                
                # 6) tri 좌표로 드로잉 (로컬→글로벌로 x 오프셋 복원)
                def draw_dets_on_tri(img, dets, xoff, color=(140,140,140)):
                    for x1,y1,x2,y2,conf,cls in dets:
                        X1, X2 = int(x1 + xoff), int(x2 + xoff)
                        cv2.rectangle(img, (X1, int(y1)), (X2, int(y2)), color, 1)

                def draw_tracks_on_tri(img, tracks, xoff, selected_id=None, remap=None):
                    for tid, x1, y1, x2, y2 in tracks:
                        shown_id = remap.get(tid, tid) if remap else tid
                        is_sel = (selected_id is not None and shown_id == selected_id)
                        color = (0,0,255) if is_sel else (0,255,0)
                        X1, X2 = int(x1 + xoff), int(x2 + xoff)
                        cv2.rectangle(img, (X1, int(y1)), (X2, int(y2)), color, 2)
                        cv2.putText(img, f"ID {shown_id}", (X1, int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


                # (a) YOLO 회색 박스
                draw_dets_on_tri(tri, dets_L, 0)
                draw_dets_on_tri(tri, dets_C, seg_w)
                draw_dets_on_tri(tri, dets_R, seg_w*2)

                # (b) 추적 박스 (선택ID는 빨강)
                draw_tracks_on_tri(tri, tracks_L, 0,       selected_id=tracker.selected_id, remap=remap_left)
                draw_tracks_on_tri(tri, tracks_C, seg_w,   selected_id=tracker.selected_id, remap=remap_center)
                draw_tracks_on_tri(tri, tracks_R, seg_w*2, selected_id=tracker.selected_id, remap=remap_right)



                # 7) HUD/ROI도 tri 좌표로 (원하면 표시)
                if left_roi:
                    cv2.rectangle(tri, (left_roi[0], left_roi[1]),
                                    (left_roi[2], left_roi[3]), (100,255,100), 2)
                if center_roi:
                    cv2.rectangle(tri, (center_roi[0] + seg_w, center_roi[1]),
                                    (center_roi[2] + seg_w, center_roi[3]), (255,255,0), 2)
                if right_roi:
                    cv2.rectangle(tri, (right_roi[0] + 2*seg_w, right_roi[1]),
                                    (right_roi[2] + 2*seg_w, right_roi[3]), (100,255,255), 2)

                # 8) 출력 이미지 지정
                disp_tri = tri
                '''print(f"[DEBUG] det counts L={len(dets_L)} C={len(dets_C)} R={len(dets_R)}")
                print(f"[DEBUG] track counts L={len(tracks_L)} C={len(tracks_C)} R={len(tracks_R)}")'''
            # === tri 모드일 땐 단일 경로 스킵되도록 아래에서 out_img 선택만 해주면 됨 ===
        cv2.setMouseCallback(WIN, mouse_callback)
        '''if tri_prepare and not tri_mouse_ready:
            install_tri_selector(
                WIN,
                get_scale=lambda: tri_disp_scale,
                get_seg_w=lambda: 720,
                get_disp_img=lambda: disp_tri,
                get_tracks_L=lambda: tracks_L,
                get_tracks_C=lambda: tracks_C,
                get_tracks_R=lambda: tracks_R
            )
            tri_mouse_ready = True'''
        
        

        
        # === 출력 프레임 결정 ===
        out_img = disp_tri if (tri_prepare and disp_tri is not None) else disp
        cv2.imshow(WIN, out_img)
        if out_img is not None:
            BUS.publish(out_img)
        # === 필요할 때만 창 크기 자동 맞춤 ===
        if need_resize and out_img is not None:
            fit_window_to_image(WIN, out_img)
            need_resize = False

    sm.stop()
    cv2.destroyAllWindows()


# ============================================================
# ✅ 차량 선택용 마우스 콜백
# ============================================================
def mouse_callback(event, x, y, flags, param):
    global tri_prepare, tracker, tracks, tracks_L, tracks_C, tracks_R

    if event == cv2.EVENT_LBUTTONDOWN:
        if tri_prepare:
            # --- tri 모드 클릭 처리 ---
            seg_w = 720  # 세그먼트 폭 (좌/중/우)
            clicked_seg = None
            x_local = x

            if x < seg_w:
                clicked_seg = "L"
            elif x < 2 * seg_w:
                clicked_seg = "C"
                x_local -= seg_w
            else:
                clicked_seg = "R"
                x_local -= 2 * seg_w

            print(f"[Mouse] tri 클릭 위치: seg={clicked_seg}, x={x_local}, y={y}")

            # 해당 세그먼트 트랙 리스트 선택
            seg_tracks = {
                "L": tracks_L if 'tracks_L' in globals() else [],
                "C": tracks_C if 'tracks_C' in globals() else [],
                "R": tracks_R if 'tracks_R' in globals() else []
            }.get(clicked_seg, [])

            clicked_id = None
            for tid, bx1, by1, bx2, by2 in seg_tracks:
                if bx1 <= x_local <= bx2 and by1 <= y <= by2:
                    clicked_id = tid
                    break

            if clicked_id is not None:
                tracker.selected_id = clicked_id
                print(f"[INFO] tri 모드 차량 선택됨: seg={clicked_seg}, ID={clicked_id}")
            else:
                tracker.selected_id = None
                print("[INFO] tri 모드 클릭 영역 내 차량 없음")

        else:
            # --- 일반 모드 클릭 처리 ---
            clicked_id = None
            for tid, bx1, by1, bx2, by2 in tracks:
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    clicked_id = tid
                    break

            if clicked_id is not None:
                tracker.selected_id = clicked_id
                print(f"[INFO] 차량 선택됨 (ID={clicked_id})")
            else:
                tracker.selected_id = None
                print("[INFO] 클릭 영역 내 차량 없음")
