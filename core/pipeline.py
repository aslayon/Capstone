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

from detectors.yolo_tracker import YOLOTracker
from detectors.yolo_detector import get_vehicle_detections
from core.stream_manager import HLSStreamManager
from core.switch_controller import SwitchController
from core.bootstrap import refresh_initial_url

from core.crop_saver import CropSaver  
from core.cam_stats import _CAMSTATS, build_fp, _extract_color_feat, _extract_shape_feat

from core.history import TrackHistory


import os, re
from pathlib import Path



def _sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', '_', str(name)).strip()






# 동일 카메라(= 같은 화면/세그먼트)일 때 / 다른 카메라일 때
REID_THRESH_SAME  = 0.40   # 동일 카메라: 엄격
REID_THRESH_OTHER = 0.50   # 다른 카메라: 관대 (야간은 0.55)


WIN = "Capstone - CCTV Tracking"

tri_selected = {"seg": "C", "id": None}  # seg in {"L","C","R"}

tri_selected = {"seg": "C", "id": None}  # seg in {"L","C","R"}

tri_prepare = False  # 단일 모드로 시작
tracker = None
tracks = []
tracks_L, tracks_C, tracks_R = [], [], []


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
    global tri_prepare, tracker, tracks, tracks_L, tracks_C, tracks_R, tri_ui_state, selected_bank, track_history_C, track_history_L, track_history_R, switcher, scale
    tri_prepare = False
    selected_bank = ReIDBank(maxlen=5, h_bins=25, s_bins=30)
    
    # ✅ 히스토리 버퍼 초기화 (카메라별로 분리)
    track_history_C = TrackHistory(maxlen=30)  # 중앙
    track_history_L = TrackHistory(maxlen=30)  # 좌측
    track_history_R = TrackHistory(maxlen=30)  # 우측
    
    HISTORY_CLEANUP_EVERY = 100  # 100프레임마다 정리
    
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

    sm = HLSStreamManager(api_key=cfg["ITS_API_KEY"], update_interval=300)
    if not sm.start(current_name, current_url):
        print("❌ 스트림 시작 실패")
        return


    switcher = SwitchController(current_name, current_url, api_key=cfg["ITS_API_KEY"],
                                graph_path="data/cctv_graph_connections.json",
                                list_path="data/cctv_list_4.json", env_path=".env") 
    switcher.attach_center_manager(sm)

    switcher.tri_mode = tri_prepare
    if tri_prepare:
        print("[INFO] Neighbor 카메라 활성화 중...")
        switcher.ensure_neighbor_managers()
        
        # 잠시 대기 (스트림 시작)
        
        time.sleep(2)
        
        
        
        

            
    
    
    # 명시적으로 설정 (더 안전함)
    # ✅ YOLO 내장 트래커 사용
    tracker = YOLOTracker(
        model_path="yolo11n.pt",
        conf_threshold=cfg["DET_CONF"],
        iou_threshold=cfg["TRACKER_IOU_TH"],
        tracker_config="bytetrack.yaml"
    )

    tracker_L = YOLOTracker(
        model_path="yolo11n.pt",
        conf_threshold=cfg["DET_CONF"],
        iou_threshold=cfg["TRACKER_IOU_TH"],
        tracker_config="bytetrack.yaml"
    )

    tracker_R = YOLOTracker(
        model_path="yolo11n.pt",
        conf_threshold=cfg["DET_CONF"],
        iou_threshold=cfg["TRACKER_IOU_TH"],
        tracker_config="bytetrack.yaml"
    )

    crop_saver = CropSaver(save_root="reid_crops", save_every=3, pad=2, print_interval_sec=1.0)
    crop_saver.new_camera(switcher.current_name)
    roi = parse_roi(cfg["ROI_RECT"])

    display_w, display_h = cfg["DISPLAY_W"], cfg["DISPLAY_H"]
    scale = 1.0
    frame_shape = [0,0]

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, mouse_callback)
    # 마우스 콜백
    def get_shape(): return (frame_shape[0], frame_shape[1])
    def get_scale(): return scale
    


    
    collect_every = 1  # 선택ID 갤러리 수집 주기(프레임)
    


    remap_left, remap_right = {}, {}

   
    lost_count = 0
    LOST_N = 15
    
    tri_win = "Tri-Prepare (L/C/R)"
    tri_prepare = False  # 단일 모드로 시작
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
        disp = None  # 매 프레임 초기화
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
            # ✅ YOLO track()으로 탐지+추적 동시 수행
            tracks = tracker.update(frame, roi=roi)
            
            # 디버깅용: detection도 가져오기
            dets = tracker.detect_only(frame, roi=roi)
            
            if frame_idx % 30 == 0:  # 30프레임마다만 출력
                '''print(f"[DETECT] {len(dets)} detections, {len(tracks)} tracks")'''

            # 트랙 히스토리 추가
            for tid, x1, y1, x2, y2 in tracks:
                track_history_C.add(tid, frame, (x1, y1, x2, y2), frame_idx)
            
            # 선택 ID 미탐지 카운트
            if tracker.selected_id is not None and all(tid != tracker.selected_id for tid, *_ in tracks):
                lost_count += 1
            else:
                lost_count = 0

            # 드로잉
            if roi:
                cv2.rectangle(frame, roi[:2], roi[2:], (255,255,0), 2)

            # Detection 박스 (회색) - 디버깅용
            for x1, y1, x2, y2, conf, cls in dets:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (140, 140, 140), 1)
                # confidence 표시
                cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

            # Tracking 박스 (초록/빨강)
            for tid, x1, y1, x2, y2 in tracks:
                color = (0,0,255) if (tracker.selected_id is not None and tid == tracker.selected_id) else (0,255,0)
                if tid == tracker.selected_id:
                    if frame_idx % collect_every == 0:
                        # 센터 원본 frame 기준으로 수집
                        if selected_bank.size() < 5:
                            selected_bank.add_from_frame_banded5_improved(frame, (x1,y1,x2,y2), origin_seg="C", origin_cam=switcher.current_name, cam_id=switcher.current_name)
                            if selected_bank.size() == 5:
                                print("[ReID] 갤러리 5장 채움. tri 매칭 기준 고정.")
                        # ✅ 자동 tri 모드 전환 비활성화 (수동으로 'p' 키로만 전환)
                        # if not tri_prepare:
                        #     tri_prepare = True
                        #     switcher.tri_mode = True
                        #     switcher.ensure_neighbor_managers()
                        #     need_resize = True
                        #     print("[DEBUG] tri-prepare: ON")
                            
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
            crop_saver.new_camera(switcher.current_name, reset_counts=True)
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

                
                # 2) ✅ 각 세그먼트에 YOLO track() 직접 수행
                tracks_L = tracker_L.update(lf_s, roi=left_roi)
                tracks_C = tracker.update(cf_s, roi=center_roi)
                tracks_R = tracker_R.update(rf_s, roi=right_roi)
                
                # 디버깅용: detection도 가져오기
                '''dets_L = tracker_L.detect_only(lf_s, roi=left_roi)
                dets_C = tracker.detect_only(cf_s, roi=center_roi)
                dets_R = tracker_R.detect_only(rf_s, roi=right_roi)'''

                remap_left, remap_center, remap_right = {}, {}, {}

                for tid, x1, y1, x2, y2 in tracks_C:
                    track_history_C.add(tid, cf_s, (x1, y1, x2, y2), frame_idx)

                for tid, x1, y1, x2, y2 in tracks_L:
                    track_history_L.add(tid, lf_s, (x1, y1, x2, y2), frame_idx)

                for tid, x1, y1, x2, y2 in tracks_R:
                    track_history_R.add(tid, rf_s, (x1, y1, x2, y2), frame_idx)
                
                    
                
                # === TRI 전용: 현재 3화면 내부 후보만 색상 유사도 비교 ===
                if tri_prepare:
                    
                    #
                    
                    #
                    
                    
                    
                    
                    
                    
                    best = None  # (d_mean, seg, tid, (x1,y1,x2,y2))
                    for seg_name, seg_frame, tracks_seg in [("L", lf_s, tracks_L),
                                                            ("C", cf_s, tracks_C),
                                                            ("R", rf_s, tracks_R)]:
                        for tid, x1,y1,x2,y2 in tracks_seg:
                            d_mean, _ = selected_bank.avg_bhatta_to_box_banded5_improved(
                                        seg_frame, (x1, y1, x2, y2),
                                        cam_id=switcher.left_name,  # ← 추가
                                        use_whitening=True          # ← 추가
                                    )
                            if d_mean is None: continue
                            if (best is None) or (d_mean < best[0]):
                                best = (d_mean, seg_name, tid, (x1,y1,x2,y2))

                    
                    
                    
                    # === (보강) 초경량 특징 + 카메라 표준화 기반 재평가 =========================
                    # 조건: 센터에서 선택ID의 현재 박스를 찾을 수 있을 때
                    try:
                        src_box = None
                        for tid0, x1,y1,x2,y2 in tracks_C:
                            if tid0 == tracker.selected_id:
                                src_box = (x1,y1,x2,y2); break
                        if src_box is not None:
                            # 카메라 식별자(세그+카메라명) — 카메라별 통계를 분리하기 위함
                            cam_src = f"C::{switcher.current_name}"
                            fp_src, _ = build_fp(cf_s, src_box, cam_id=cam_src, bins=32, with_shape=True, do_whiten=True)

                            # 후보들 전부 스캔해서 최고 점수로 재선택
                            best2 = None   # (score, seg, tid, box, cos, bh)
                            for seg_name, seg_frame, tracks_seg, cam_name in [
                                ("L", lf_s, tracks_L, getattr(switcher, "left_name",  None)),
                                ("C", cf_s, tracks_C, getattr(switcher, "current_name", None)),
                                ("R", rf_s, tracks_R, getattr(switcher, "right_name", None)),
                            ]:
                                cam_tgt = f"{seg_name}::{cam_name}"
                                for tid1, x1,y1,x2,y2 in tracks_seg:
                                    box = (x1,y1,x2,y2)
                                    fp_tgt, _ = build_fp(seg_frame, box, cam_id=cam_tgt, bins=32, with_shape=True, do_whiten=True)
                                    if fp_tgt is None or fp_src is None: 
                                        continue
                                    cos = _cos(fp_src, fp_tgt)
                                    bh  = _bhattacharyya(fp_src, fp_tgt)
                                    # 종합 점수 (코사인↑, 바차차야↓)
                                    score = 0.6*cos + 0.4*(1.0 - bh)
                                    if (best2 is None) or (score > best2[0]):
                                        best2 = (score, seg_name, tid1, box, cos, bh)

                            if best2 is not None:
                                score, seg2, tid2, box2, cos, bh = best2
                                # 이중 게이트(보수적으로 시작): cos>=0.85 or bh<=0.25
                                if (cos >= 0.85) or (bh <= 0.25):
                                    # 최종 remap 반영 (보강판정이 원판정보보다 우선)
                                    if seg2 == "L":      remap_left[tid2]   = tracker.selected_id
                                    elif seg2 == "C":    remap_center[tid2] = tracker.selected_id
                                    else:                remap_right[tid2]  = tracker.selected_id

                                    cv2.putText(tri, f"[REFINE OK] {seg2} cos={cos:.2f} bh={bh:.2f} s={score:.2f}",
                                                (10, 80), 0, 0.7, (0,255,0), 2)
                                else:
                                    cv2.putText(tri, f"[REFINE NO] cos={cos:.2f} bh={bh:.2f}",
                                                (10, 80), 0, 0.7, (0,0,255), 2)

                            # 통계 저장
                            _CAMSTATS.save()
                    except Exception as e:
                        # 보강 모듈 실패해도 전체 파이프라인 영향 없게
                        print("[REFINE ERROR]", e)
                    # =======================================================================

                    
                    if best is not None:
                        dist, seg, tid, _ = best

                        # --- 임계값 분기: 같은 화면(seg) vs 다른 화면 ---
                        origin_seg = getattr(selected_bank, "origin_seg", None)
                        # (옵션) 카메라 이름까지 같을 때만 SAME로 하려면 origin_cam 비교도 추가
                        # same_cam = (getattr(selected_bank, "origin_cam", None) == getattr(switcher, "current_name", None)) if seg=="C" else False
                        same_screen = (origin_seg is not None and seg == origin_seg)
                        thresh = get_adaptive_threshold(selected_bank, switcher.current_name)

                        if dist <= thresh:
                            # 동일 차량 취급(remap)
                            if seg == "L":  
                                remap_left[tid]   = tracker.selected_id
                                tracker.selected_id = tid
                                print(f"[SELECT UPDATE] tri L -> ID {tid}")
                            elif seg == "C":  
                                remap_center[tid] = tracker.selected_id
                                tracker.selected_id = tid
                                print(f"[SELECT UPDATE] tri L -> ID {tid}")
                                
                            else:           
                                remap_right[tid]  = tracker.selected_id
                                tracker.selected_id = tid
                                print(f"[SELECT UPDATE] tri L -> ID {tid}")
                                

                            cv2.putText(tri, f"match {seg} dm={dist:.3f} th={thresh:.2f} ({'same' if same_screen else 'other'})",
                                        (10, 50), 0, 0.7, (0,255,0), 2)
                        else:
                            cv2.putText(tri, f"no match {seg} dm={dist:.3f} th={thresh:.2f}",
                                        (10, 50), 0, 0.7, (0,0,255), 2)

                
                
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


                # === TRI 전용: 선택 ID 갤러리(5장) 채우기 ===
                                
                if tri_prepare and tracker.selected_id is not None:

                    # ── lazy 전역 상태 (최초 1회 초기화) ────────────────────────────────
                    global COLLECT_MIN_INTERVAL_FRAMES, _last_collect_fidx, _bank_filled_announced, last_saved_box
                    if 'COLLECT_MIN_INTERVAL_FRAMES' not in globals():
                        COLLECT_MIN_INTERVAL_FRAMES = 1   # 수집 최소 프레임 간격(필요시 5~10으로 올려도 됨)
                    if '_last_collect_fidx' not in globals():
                        _last_collect_fidx = -9999
                    if '_bank_filled_announced' not in globals():
                        _bank_filled_announced = False
                    if 'last_saved_box' not in globals():
                        last_saved_box = None

                    # ── 0) 이미 5장 채워졌으면 수집/로그 스킵 ─────────────────────────
                    if hasattr(selected_bank, "size5") and selected_bank.size5() >= 5:
                        if not _bank_filled_announced:
                            _bank_filled_announced = True
                            print("[ReID] 갤러리 5장 채움. tri 매칭 기준 고정.")
                    else:
                        # ── 1) 프레임 간격 체크 + collect_every 간격 동시 만족 ──────────
                        if (frame_idx - _last_collect_fidx) >= COLLECT_MIN_INTERVAL_FRAMES and (frame_idx % collect_every == 0):

                            # IoU 유틸(중복 샷 방지)
                            def _iou_xyxy(a, b):
                                ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
                                inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
                                inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
                                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                                    return 0.0
                                inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                                area_a = max(1.0, (ax2-ax1)*(ay2-ay1))
                                area_b = max(1.0, (bx2-bx1)*(by2-by1))
                                return inter / (area_a + area_b - inter + 1e-6)

                            # 표시 ID(shown_id) == selected_id 인 박스를 우선 수집
                            def try_collect_from(seg_name, seg_frame, tracks_seg, remap_seg):
                                global _last_collect_fidx, last_saved_box
                                for tid, x1, y1, x2, y2 in tracks_seg:
                                    shown_id = remap_seg.get(tid, tid) if remap_seg else tid
                                    if shown_id == tracker.selected_id:
                                        box = (x1, y1, x2, y2)
                                        # 직전 저장 샷과 너무 겹치면 스킵
                                        #if last_saved_box is not None and _iou_xyxy(box, last_saved_box) > 0.85:
                                            #continue
                                        ok = selected_bank.add_from_frame_banded5_improved(seg_frame, (x1, y1, x2, y2),
                                                                                origin_seg=seg_name,                           # ← 추가
                                                                                origin_cam=getattr(switcher, "current_name", None) if seg_name=="C"
                                                                                        else (getattr(switcher, "left_name", None)  if seg_name=="L"
                                                                                                else getattr(switcher, "right_name", None)))
                                        if ok:
                                            last_saved_box = box
                                            _last_collect_fidx = frame_idx
                                            # size5()가 없으면 items_band5 길이로 대체
                                            cur = selected_bank.size5() if hasattr(selected_bank, "size5") else len(getattr(selected_bank, "items_band5", []))
                                            print(f"[ReID] tri 수집: seg={seg_name}, id={shown_id}, size5={cur}")
                                        return ok
                                return False

                            # 우선순위: C → L → R (원하면 바꿔도 됨)
                            collected = (
                                try_collect_from("L", lf_s, tracks_L, remap_left)   or
                                try_collect_from("C", cf_s, tracks_C, remap_center) or
                                
                                try_collect_from("R", rf_s, tracks_R, remap_right)
                            )

                            # remap 전에라도 센터 tid==selected_id 로 한 번 더 시도
                            if not collected:
                                for tid, x1, y1, x2, y2 in tracks_C:
                                    
                                    if tid == tracker.selected_id:
                                        # ✅ origin 정보 함께 저장
                                        if frame_idx % collect_every == 0:
                                            added = selected_bank.add_from_frame_banded5_improved(
                                            cf_s, (x1, y1, x2, y2),
                                            origin_seg="C",
                                            origin_cam=switcher.current_name,
                                            cam_id=switcher.current_name,  # ← 추가
                                            use_whitening=True             # ← 추가
                                        )
                                            if added:
                                                print(f"[BANK] 5band added: ID={tid}, size={selected_bank.size5()}/5")

                        # ── 2) 이제 막 5장 채워졌다면 딱 한 번만 알림 ─────────────────
                        if hasattr(selected_bank, "size5") and selected_bank.size5() == 5 and not _bank_filled_announced:
                            _bank_filled_announced = True
                            print("[ReID] 갤러리 5장 채움. tri 매칭 기준 고정.")

                #
                # === TRI 전용: 5밴드 d 로깅 ===
                    if tri_prepare and tracker.selected_id is not None and selected_bank.size5() >= 5:
                        if frame_idx % 2 == 0:
                            now_txt = time.strftime("%Y-%m-%d %H:%M:%S") + f".{int((time.time()%1)*1000):03d}"
                            lines = []
                            for seg_name, seg_frame, tracks_seg in [
                                ("L", lf_s, tracks_L),
                                ("C", cf_s, tracks_C),
                                ("R", rf_s, tracks_R),
                            ]:
                                for tid, x1, y1, x2, y2 in tracks_seg:
                                    d_mean, per_band = selected_bank.avg_bhatta_to_box_banded5_improved(
                                                    lf_s, (x1, y1, x2, y2),
                                                    cam_id=switcher.left_name,  # ← 추가
                                                    use_whitening=True          # ← 추가
                                                )
                                    if d_mean is None: 
                                        continue
                                    
                                    per_band = np.atleast_1d(per_band).tolist()
                                    
                                    band_str = " ".join([f"b{i}={d:.4f}" if d is not None else f"b{i}=-" 
                                                        for i,d in enumerate(per_band)])
                                    lines.append(f"{now_txt}  seg={seg_name}  tid={tid}  d_mean={d_mean:.4f}  {band_str}")
                            if lines:
                                log_bhatta5(switcher.current_name, tracker.selected_id, lines)


                #
                


                # (a) YOLO 회색 박스 - 디버깅용
                '''draw_dets_on_tri(tri, dets_L, 0)
                draw_dets_on_tri(tri, dets_C, seg_w)
                draw_dets_on_tri(tri, dets_R, seg_w*2)'''

                # (b) 추적 박스 (선택ID는 빨강)
                draw_tracks_on_tri(tri, tracks_L, 0,       selected_id=tracker.selected_id, remap=remap_left)
                draw_tracks_on_tri(tri, tracks_C, seg_w,   selected_id=tracker.selected_id, remap=remap_center)
                draw_tracks_on_tri(tri, tracks_R, seg_w*2, selected_id=tracker.selected_id, remap=remap_right)

                #print(lf_s.shape, cf_s.shape, rf_s.shape)

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
        
        

        
        if tri_prepare and tri is not None:
            tri_h, tri_w = tri.shape[:2]
            seg_w = lf_s.shape[1]
            tri_disp = tri
            disp_w, disp_h = tri_w, tri_h
            ox, oy = 0, 0

            # 마우스 콜백이 쓸 상태 저장
            if 'tri_ui_state' not in globals():
                tri_ui_state = {}
            tri_ui_state.update({
                "orig_w": tri_w, "orig_h": tri_h,
                "disp_w": disp_w, "disp_h": disp_h,
                "offset_x": ox, "offset_y": oy,
                "seg_w": seg_w,
            })

            
        

        if frame_idx % HISTORY_CLEANUP_EVERY == 0:
            current_tids_C = {tid for tid, *_ in tracks_C}
            cleaned_C = track_history_C.cleanup_old_tracks(current_tids_C)
            
            if tri_prepare:
                current_tids_L = {tid for tid, *_ in tracks_L}
                current_tids_R = {tid for tid, *_ in tracks_R}
                cleaned_L = track_history_L.cleanup_old_tracks(current_tids_L)
                cleaned_R = track_history_R.cleanup_old_tracks(current_tids_R)
                
                if cleaned_C + cleaned_L + cleaned_R > 0:
                    print(f"[HISTORY] 정리: C={cleaned_C}, L={cleaned_L}, R={cleaned_R}")
            else:
                if cleaned_C > 0:
                    print(f"[HISTORY] 정리: {cleaned_C}개 트랙")
        
        
        
        
        out_img = disp_tri if (tri_prepare and disp_tri is not None) else disp
        if out_img is None:
            print("[WARN] out_img가 None입니다!")
            continue
        cv2.imshow(WIN, out_img)
        if out_img is not None:
            BUS.publish(out_img)
        # === 필요할 때만 창 크기 자동 맞춤 ===
        if need_resize and out_img is not None:
            fit_window_to_image(WIN, out_img)
            need_resize = False


    _CAMSTATS.save()
    print("[INFO] cam_stats.json 저장 완료")
    sm.stop()
    cv2.destroyAllWindows()


# ============================================================
# ✅ 차량 선택용 마우스 콜백
# ============================================================
def find_closest_track_in_history(track_history, click_x, click_y, max_frames_back=15, max_distance=150):
    """
    최근 N 프레임의 히스토리에서 클릭 위치에 가장 가까운 트랙 찾기
    
    Args:
        track_history: TrackHistory 객체
        click_x, click_y: 클릭 좌표 (로컬 세그먼트 좌표)
        max_frames_back: 최대 몇 프레임 전까지 검색할지 (기본 15)
        max_distance: 최대 허용 거리 픽셀 (기본 150)
    
    Returns:
        (track_id, distance, frame_count) or (None, None, 0)
    """
    if not hasattr(track_history, 'history') or not track_history.history:
        return None, None, 0
    
    best_tid = None
    best_distance = max_distance
    best_frame_idx = -1
    total_checked = 0
    
    # 모든 트랙의 최근 프레임 검색
    for tid, frames in track_history.history.items():
        if not frames or len(frames) == 0:
            continue
        
        # 최근 max_frames_back 프레임만 검색 (deque이므로 끝에서부터)
        recent_count = min(len(frames), max_frames_back)
        recent_frames = list(frames)[-recent_count:]
        
        for crop, bbox, fidx in recent_frames:
            total_checked += 1
            x1, y1, x2, y2 = bbox
            
            # bbox 중심점 계산
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # 클릭 위치와의 유클리드 거리 계산
            distance = np.sqrt((cx - click_x)**2 + (cy - click_y)**2)
            
            # 더 가까운 트랙 발견
            if distance < best_distance:
                best_distance = distance
                best_tid = tid
                best_frame_idx = fidx
    
    if best_tid is not None:
        print(f"[HISTORY_MATCH] ✅ {total_checked}개 프레임 검색 → ID {best_tid} 발견 (거리: {best_distance:.1f}px, 프레임: {best_frame_idx})")
        return best_tid, best_distance, total_checked
    else:
        print(f"[HISTORY_MATCH] ❌ {total_checked}개 프레임 검색했으나 {max_distance}px 이내 트랙 없음")
        return None, None, total_checked


def mouse_callback(event, x, y, flags, param):
    """
    개선된 마우스 콜백 - 히스토리 기반 트랙 검색 포함
    
    동작 순서:
    1. 현재 프레임의 트랙에서 직접 히트 테스트 (bbox 내부 클릭)
    2. 실패 시 최근 15프레임 히스토리에서 가장 가까운 트랙 검색
    3. 성공 시 해당 트랙의 히스토리에서 갤러리 수집
    """
    global tri_prepare, tracker, tracks, tracks_L, tracks_C, tracks_R, tri_ui_state, selected_bank
    global track_history_C, track_history_L, track_history_R, switcher

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if tri_prepare:
        # ============================================================
        # Tri 모드 (3-화면 동시 표시)
        # ============================================================
        s = tri_ui_state if 'tri_ui_state' in globals() else None
        if not s:
            print("[Mouse] tri_ui_state 없음")
            return

        orig_w = s.get("orig_w", 0)
        orig_h = s.get("orig_h", 0)
        disp_w = s.get("disp_w", 0)
        disp_h = s.get("disp_h", 0)
        ox = s.get("offset_x", 0)
        oy = s.get("offset_y", 0)
        seg_w = s.get("seg_w", 0)

        if orig_w <= 0 or disp_w <= 0 or seg_w <= 0:
            print("[Mouse] tri state invalid")
            return

        # 1) 표시 좌표 → tri 원본 좌표 (레터박스 보정)
        x_in = x - ox
        y_in = y - oy
        if not (0 <= x_in < disp_w and 0 <= y_in < disp_h):
            print("[Mouse] 여백 클릭")
            return

        x_tri = int(x_in * orig_w / disp_w)
        y_tri = int(y_in * orig_h / disp_h)

        # 2) 세그먼트 판정 (L/C/R)
        seg_idx = min(max(x_tri // seg_w, 0), 2)
        clicked_seg = ("L", "C", "R")[seg_idx]
        x_local = x_tri - seg_idx * seg_w
        y_local = y_tri

        print(f"[Mouse] tri click: seg={clicked_seg}, local=({x_local},{y_local}), tri=({x_tri},{y_tri})")

        # 3) 현재 프레임에서 트랙 직접 히트 테스트
        seg_tracks = {
            "L": tracks_L if 'tracks_L' in globals() else [],
            "C": tracks_C if 'tracks_C' in globals() else [],
            "R": tracks_R if 'tracks_R' in globals() else []
        }.get(clicked_seg, [])

        clicked_id = None
        
        # 3-1) 현재 프레임에서 bbox 내부 클릭 체크
        for tid, bx1, by1, bx2, by2 in seg_tracks:
            if bx1 <= x_local <= bx2 and by1 <= y_local <= by2:
                clicked_id = tid
                print(f"[DIRECT_HIT] ✅ 현재 프레임에서 ID {tid} 직접 선택")
                break
        
        # 3-2) ✅ 현재 프레임에서 못 찾으면 히스토리 검색
        if clicked_id is None:
            print(f"[HISTORY_SEARCH] 현재 프레임에 트랙 없음, 히스토리 검색 시작...")
            
            # 세그먼트에 맞는 히스토리와 카메라명 선택
            if clicked_seg == "L":
                history = track_history_L
                cam_name = switcher.left_name
            elif clicked_seg == "C":
                history = track_history_C
                cam_name = switcher.current_name
            else:  # "R"
                history = track_history_R
                cam_name = switcher.right_name
            
            # 히스토리에서 가장 가까운 트랙 찾기
            clicked_id, distance, checked = find_closest_track_in_history(
                history, x_local, y_local,
                max_frames_back=15,  # 최근 15프레임 (0.5초)
                max_distance=150     # 150픽셀 이내
            )

        # 4) 선택 처리
        if clicked_id is not None:
            tracker.selected_id = clicked_id
            print(f"[INFO] 🎯 tri 모드 차량 선택됨: seg={clicked_seg}, ID={clicked_id}")
            
            # 히스토리 및 카메라명 결정
            if clicked_seg == "L":
                history = track_history_L
                cam_name = switcher.left_name
            elif clicked_seg == "C":
                history = track_history_C
                cam_name = switcher.current_name
            else:  # "R"
                history = track_history_R
                cam_name = switcher.right_name
            
            # 히스토리에서 갤러리 수집
            collected = 0
            target = selected_bank.items_band5.maxlen
            
            history_frames = history.get_history(clicked_id)
            print(f"[HISTORY] 📦 {len(history_frames)}개 프레임 발견, {target}장 수집 시도...")
            
            for crop, bbox, fidx in reversed(history_frames):  # 최근부터
                if collected >= target:
                    break
                
                h, w = crop.shape[:2]
                added = selected_bank.add_from_frame_banded5_improved(
                    crop, (0, 0, w, h),  # crop 전체
                    origin_seg=clicked_seg,
                    origin_cam=cam_name,
                    cam_id=cam_name,
                    use_whitening=True
                )
                
                if added:
                    collected += 1
            
            if collected > 0:
                print(f"[HISTORY] ✅ {collected}/{target}장 수집 완료")
            else:
                print(f"[HISTORY] ⚠️  히스토리 수집 실패, 실시간 수집 시작")
            
        else:
            # 선택 해제
            tracker.selected_id = None
            selected_bank.clear()
            print("[INFO] ❌ tri 모드 클릭 영역 내 차량 없음 (현재+히스토리)")

    else:
        # ============================================================
        # 단일 모드 (중앙 화면만 표시)
        # ============================================================
        # ✅ 스케일 보정: 표시 좌표 → 원본 좌표
        try:
            current_scale = scale if scale > 0 else 1.0
        except:
            current_scale = 1.0
        
        orig_x = int(x / current_scale)
        orig_y = int(y / current_scale)
        
        print(f"[Mouse] 클릭: 표시=({x},{y}), 원본=({orig_x},{orig_y}), scale={current_scale:.3f}")
        
        clicked_id = None
        
        # 1) 현재 프레임에서 직접 히트 테스트
        for tid, bx1, by1, bx2, by2 in tracks:
            if bx1 <= orig_x <= bx2 and by1 <= orig_y <= by2:
                clicked_id = tid
                print(f"[DIRECT_HIT] ✅ ID {tid} 선택 (bbox={bx1},{by1},{bx2},{by2})")
                break
        
        # 2) ✅ 현재 프레임에서 못 찾으면 히스토리 검색
        if clicked_id is None:
            print(f"[HISTORY_SEARCH] 현재 프레임에 트랙 없음, 히스토리 검색 시작...")
            
            clicked_id, distance, checked = find_closest_track_in_history(
                track_history_C, orig_x, orig_y,
                max_frames_back=15,
                max_distance=150
            )

        # 3) 선택 처리
        if clicked_id is not None:
            tracker.selected_id = clicked_id
            print(f"[INFO] 🎯 차량 선택됨 (ID={clicked_id}) - 다음 프레임부터 빨간 박스로 표시됩니다")
            
            # 히스토리에서 갤러리 수집
            history_frames = track_history_C.get_history(clicked_id)
            collected = 0
            target = selected_bank.items_band5.maxlen
            
            print(f"[HISTORY] 📦 {len(history_frames)}개 프레임 발견, {target}장 수집 시도...")
            
            for crop, bbox, fidx in reversed(history_frames):
                if collected >= target:
                    break
                
                h, w = crop.shape[:2]
                if selected_bank.add_from_frame_banded5_improved(
                    crop, (0, 0, w, h),
                    origin_seg="C",
                    origin_cam=switcher.current_name,
                    cam_id=switcher.current_name,
                    use_whitening=True
                ):
                    collected += 1
            
            if collected > 0:
                print(f"[HISTORY] ✅ {collected}/{target}장 수집 완료")
            else:
                print(f"[HISTORY] ⚠️  히스토리 수집 실패, 실시간 수집 시작")
            
        else:
            # 선택 해제
            selected_bank.clear()
            tracker.selected_id = None
            print("[INFO] ❌ 클릭 영역 내 차량 없음 (현재+히스토리)")

#

def bhatta_dist_for_box(bank, frame, box):
    """5밴드 우선 사용"""
    # ✅ 5밴드 메서드 우선
    if hasattr(bank, "avg_bhatta_to_box_banded5_improved"):
        result = bank.avg_bhatta_to_box_banded5_improved(frame, box)
        if result is not None and result[0] is not None:
            return result[0]  # d_mean만 반환
    
    # 폴백: 3밴드
    if hasattr(bank, "avg_bhatta_to_box_banded"):
        return bank.avg_bhatta_to_box_banded(frame, box)
    
    # 폴백: 전체 히스토그램
    if hasattr(bank, "avg_bhatta_to_box"):
        return bank.avg_bhatta_to_box(frame, box)
    
    # 폴백: score_to_gallery
    if hasattr(bank, "score_to_gallery"):
        return bank.score_to_gallery(frame, box)
    
    return None


def get_adaptive_threshold(selected_bank, current_cam, frame_brightness=None):
    """
    선택 차량의 origin_cam과 현재 카메라/밝기를 고려한 임계값 반환
    """
    # 기본값
    same_thresh = 0.40
    other_thresh = 0.50
    
    # 야간 보정 (선택적)
    if frame_brightness is not None and frame_brightness < 100:
        other_thresh = 0.55  # 야간에는 더 관대하게
    
    # origin_cam 확인
    if not hasattr(selected_bank, 'origin_cam') or selected_bank.origin_cam is None:
        return same_thresh  # 첫 포착: 엄격
    
    if selected_bank.origin_cam == current_cam:
        return same_thresh
    else:
        return other_thresh

#






def _cos(a,b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na<1e-6 or nb<1e-6: return 0.0
    return float(np.dot(a,b)/(na*nb))

def _bhattacharyya(p, q):
    p = np.maximum(0,p); q = np.maximum(0,q)
    p = p/(p.sum()+1e-6); q = q/(q.sum()+1e-6)
    bc = np.sum(np.sqrt(p*q))
    return float(np.clip(np.sqrt(max(0.0, 1.0-bc)), 0.0, 1.0))   # 0~1, 작을수록 유사


def log_bhatta(cam_name: str, selected_id: int, lines: list, root="reid_logs"):
    """
    lines: ["YYYY-mm-dd HH:MM:SS.sss  seg=L  tid=12  d=0.423", ...]
    파일: reid_logs/<카메라명>/id_<selected_id>.txt 로 append
    """
    cam_dir = _sanitize(cam_name)
    out_dir = Path(root) / cam_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"id_{selected_id}.txt"
    with open(path, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")



# 로그 유틸은 기존 log_bhatta 사용하고, root만 분리해서 5밴드 로그로 보관
def log_bhatta5(cam_name: str, selected_id: int, lines: list, root="reid_logs_5band"):
    cam_dir = _sanitize(cam_name)
    out_dir = Path(root) / cam_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"id_{selected_id}.txt"
    with open(path, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")