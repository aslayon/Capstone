# core/pipeline.py
# ✅ UNIFIED TRACKER VERSION - 단일 트래커로 3화면 통합 처리
# 실행 루프
# 패치내역: 3캠 화면을 concat하여 단일 YOLO 트래커로 처리 (자원 효율화)
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


# ===== 🎯 세그먼트 분할 유틸리티 =====
def split_tracks_by_segment(tracks, seg_width=720):
    """
    concat된 프레임의 tracks를 세그먼트별로 분할하고 로컬 좌표로 변환
    
    Args:
        tracks: [(tid, x1, y1, x2, y2), ...] (글로벌 좌표)
        seg_width: 각 세그먼트 폭 (기본 720)
    
    Returns:
        tracks_L, tracks_C, tracks_R (각각 로컬 좌표)
    """
    tracks_L = []
    tracks_C = []
    tracks_R = []
    
    for tid, x1, y1, x2, y2 in tracks:
        cx = (x1 + x2) / 2  # 중심점으로 판단
        
        if cx < seg_width:
            # 왼쪽 세그먼트 (좌표 그대로)
            tracks_L.append((tid, x1, y1, x2, y2))
        elif cx < seg_width * 2:
            # 중앙 세그먼트 (좌표 보정)
            tracks_C.append((tid, x1 - seg_width, y1, x2 - seg_width, y2))
        else:
            # 오른쪽 세그먼트 (좌표 보정)
            tracks_R.append((tid, x1 - seg_width*2, y1, x2 - seg_width*2, y2))
    
    return tracks_L, tracks_C, tracks_R


def convert_local_to_global(tracks, seg_offset):
    """
    로컬 좌표를 글로벌 좌표로 변환 (클릭 처리용)
    
    Args:
        tracks: [(tid, x1, y1, x2, y2), ...] (로컬 좌표)
        seg_offset: 세그먼트 x 오프셋 (0, 720, 1440)
    
    Returns:
        [(tid, x1, y1, x2, y2), ...] (글로벌 좌표)
    """
    return [(tid, x1 + seg_offset, y1, x2 + seg_offset, y2) 
            for tid, x1, y1, x2, y2 in tracks]


# 동일 카메라(= 같은 화면/세그먼트)일 때 / 다른 카메라일 때
REID_THRESH_SAME  = 0.55   # 동일 카메라: 여유 있게
REID_THRESH_OTHER = 0.65   # 다른 카메라: 로그 분석 기반 최적값


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
   

# ===== 연속 매칭 검증 클래스 =====
class ConsecutiveMatchValidator:
    def __init__(self, required_count=3):
        self.required_count = required_count
        self.match_history = {}
        self.last_frame_idx = {}
    
    def validate(self, track_id, d_mean, thresh, frame_idx):
        if d_mean > thresh:
            self.match_history[track_id] = 0
            return False, "REJECT", 0
        last_frame = self.last_frame_idx.get(track_id, -999)
        if frame_idx - last_frame > 5:
            self.match_history[track_id] = 1
        else:
            count = self.match_history.get(track_id, 0) + 1
            self.match_history[track_id] = count
        self.last_frame_idx[track_id] = frame_idx
        count = self.match_history[track_id]
        if count >= self.required_count:
            return True, "CONFIRMED", count
        else:
            return False, "PENDING", count
    
    def reset(self):
        self.match_history.clear()
        self.last_frame_idx.clear()
    
    def cleanup(self, current_frame_idx, max_age=30):
        to_remove = [tid for tid, last_frame in self.last_frame_idx.items() 
                     if current_frame_idx - last_frame > max_age]
        for tid in to_remove:
            self.match_history.pop(tid, None)
            self.last_frame_idx.pop(tid, None)


def evaluate_match_confidence(d_mean):
    """
    매칭 신뢰도 평가
    
    Args:
        d_mean: 바타차야 거리
    
    Returns:
        str: "HIGH", "MEDIUM", "LOW", "REJECT"
    """
    if d_mean < 0.60:
        return "HIGH"
    elif d_mean < 0.70:
        return "MEDIUM"
    elif d_mean < 0.80:
        return "LOW"
    else:
        return "REJECT"



def run_detect():
    cfg = load_config()
    global tri_prepare, tracker, tracks, tracks_L, tracks_C, tracks_R, tri_ui_state, selected_bank, track_history_C, track_history_L, track_history_R, switcher, scale
    tri_prepare = False
    selected_bank = ReIDBank(maxlen=5, h_bins=25, s_bins=30)
    
    # ✅ 히스토리 버퍼 초기화 (카메라별로 분리)
    track_history_C = TrackHistory(maxlen=30)  # 중앙
    track_history_L = TrackHistory(maxlen=30)  # 좌측
    track_history_R = TrackHistory(maxlen=30)
    
    # ✅ 연속 매칭 검증기 초기화
    match_validator = ConsecutiveMatchValidator(required_count=3)  # 우측
    
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
                                graph_path="config/cctv_graph_connections.json",
                                list_path="data/cctv_list_4.json", env_path=".env") 
    switcher.attach_center_manager(sm)

    switcher.tri_mode = tri_prepare
    if tri_prepare:
        print("[INFO] Neighbor 카메라 활성화 중...")
        switcher.ensure_neighbor_managers()
        
        # 잠시 대기 (스트림 시작)
        
        time.sleep(2)

    # ===== 🔥 핵심 변경: 단일 트래커만 사용 =====
    print("[INFO] ✅ 단일 통합 트래커 초기화 (UNIFIED MODE)")
    tracker = YOLOTracker(
        model_path="yolo11n.pt",
        conf_threshold=cfg["DET_CONF"],
        iou_threshold=cfg["TRACKER_IOU_TH"],
        tracker_config="detectors/bytetrack.yaml"
    )
    
    # ✅ 기존 tracker_L, tracker_R 제거됨!
    print("[INFO] 🚀 메모리 절약: 트래커 3개 → 1개")
    print("[INFO] 💡 화면 경계는 '장애물'로 취급, ReID로 ID 복구")

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
    
    # ===== 🎯 단일 트래커 성능 측정 =====
    unified_stats = {
        "total_tracks": 0,
        "boundary_transitions": 0,  # 경계 횡단 감지
        "reid_recoveries": 0,       # ReID 복구 횟수
    }
    
    while True:
        tri = None
        disp_tri = None
        disp = None  # 매 프레임 초기화
        switcher.tick()
        frame = switcher.center_sm.get_frame()
        disp = frame
        
        frame_idx += 1

        if frame is None:
            continue
        
        frame_count += 1
        now = time.time()

        # 1초마다 FPS 계산
        if now - last_time >= 1.0:
            fps = frame_count / (now - last_time)
            print(f"[DEBUG] FPS: {fps:.2f} | Unified Tracks: {unified_stats['total_tracks']}")
            frame_count = 0
            last_time = now
        
        H, W = frame.shape[:2]
        frame_shape[0], frame_shape[1] = H, W
        
        if not tri_prepare:
            # ============================================================
            # 단일 모드 (중앙 화면만)
            # ============================================================
            tracks = tracker.update(frame, roi=roi)
            dets = tracker.detect_only(frame, roi=roi)
            
            if frame_idx % 30 == 0:
                pass  # 로그 생략

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

            # Detection 박스 (회색)
            for x1, y1, x2, y2, conf, cls in dets:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (140, 140, 140), 1)
                cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

            # Tracking 박스
            for tid, x1, y1, x2, y2 in tracks:
                color = (0,0,255) if (tracker.selected_id is not None and tid == tracker.selected_id) else (0,255,0)
                if tid == tracker.selected_id:
                    if frame_idx % collect_every == 0:
                        if selected_bank.size() < 5:
                            selected_bank.add_from_frame_banded5_improved(frame, (x1,y1,x2,y2), origin_seg="C", origin_cam=switcher.current_name, cam_id=switcher.current_name)
                            if selected_bank.size() == 5:
                                print("[ReID] 갤러리 5장 채움. tri 매칭 기준 고정.")
                            
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"ID {tid}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            info = f"[UNIFIED] Selected: {tracker.selected_id} | Lost: {lost_count}/{LOST_N} | 'p': tri={'ON' if tri_prepare else 'OFF'}"
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
            print("[DEBUG] tri-prepare:", tri_prepare)
            if tri_prepare:
                switcher.ensure_neighbor_managers()
            need_resize = True
        else:
            switcher.on_key(k)
            crop_saver.new_camera(switcher.current_name, reset_counts=True)

        # ============================================================
        # === 🔥 TRI 모드: 단일 트래커로 concat 처리 ===
        # ============================================================
        if tri_prepare:
            lf = switcher.left_sm.get_frame()  if switcher.left_sm  else None
            cf = frame
            rf = switcher.right_sm.get_frame() if switcher.right_sm else None
            
            if lf is None or cf is None or rf is None:
                disp_tri = None
            else:
                # 1) 크기 맞추기
                target_h = 480
                target_w = 720
                def fit(x):
                    return cv2.resize(x, (target_w, target_h))
                lf_s, cf_s, rf_s = fit(lf), fit(cf), fit(rf)
                
                # 2) ✅ 핵심: Concat 후 단일 트래커로 한 번에 처리
                tri = np.hstack([lf_s, cf_s, rf_s])  # (480, 2160)
                seg_w = target_w

                print(f"[UNIFIED] 🔥 Concat 완료: {tri.shape}, 단일 트래커 실행...")
                
                # 3) ✅ 단일 추적 실행 (하나의 큰 프레임으로 취급)
                all_tracks = tracker.update(tri)  # 글로벌 좌표
                
                # 4) ✅ 세그먼트별 분할 (좌표 변환)
                tracks_L, tracks_C, tracks_R = split_tracks_by_segment(all_tracks, seg_w)
                
                unified_stats["total_tracks"] = len(all_tracks)
                
                print(f"[UNIFIED] 📊 분할 결과: L={len(tracks_L)}, C={len(tracks_C)}, R={len(tracks_R)}")

                # 5) 히스토리 추가 (로컬 좌표)
                for tid, x1, y1, x2, y2 in tracks_L:
                    track_history_L.add(tid, lf_s, (x1, y1, x2, y2), frame_idx)
                for tid, x1, y1, x2, y2 in tracks_C:
                    track_history_C.add(tid, cf_s, (x1, y1, x2, y2), frame_idx)
                for tid, x1, y1, x2, y2 in tracks_R:
                    track_history_R.add(tid, rf_s, (x1, y1, x2, y2), frame_idx)

                # 6) ✅ ReID 매칭 (세그먼트 간 ID 복구)
                remap_left, remap_center, remap_right = {}, {}, {}
                
                if tracker.selected_id is not None and selected_bank.size5() >= 5:
                    best = None
                    for seg_name, seg_frame, tracks_seg in [("L", lf_s, tracks_L),
                                                            ("C", cf_s, tracks_C),
                                                            ("R", rf_s, tracks_R)]:
                        for tid, x1, y1, x2, y2 in tracks_seg:
                            d_mean, _ = selected_bank.avg_bhatta_to_box_banded5_improved(
                                seg_frame, (x1, y1, x2, y2),
                                cam_id=switcher.current_name,
                                use_whitening=True
                            )
                            if d_mean is None:
                                continue
                            if (best is None) or (d_mean < best[0]):
                                best = (d_mean, seg_name, tid, (x1, y1, x2, y2))

                    if best is not None:
                        dist, seg, tid, _ = best
                        origin_seg = getattr(selected_bank, "origin_seg", None)
                        thresh = get_adaptive_threshold(selected_bank, seg, switcher.current_name)

                        confirmed, status, count = match_validator.validate(tid, dist, thresh, frame_idx)
                        
                        if confirmed:
                            confidence = evaluate_match_confidence(dist)
                            context = f"{origin_seg or '?'}->{seg}"
                            print(f"[UNIFIED MATCH ✅] x{count} d={dist:.3f} conf={confidence} [{context}]")
                            
                            # ✅ 경계 횡단 감지
                            if origin_seg != seg:
                                unified_stats["boundary_transitions"] += 1
                                print(f"[BOUNDARY] 🚦 경계 횡단 감지! {origin_seg} → {seg}")
                            
                            unified_stats["reid_recoveries"] += 1
                            
                            # ID 매핑
                            if seg == "L":
                                remap_left[tid] = tracker.selected_id
                                tracker.selected_id = tid
                            elif seg == "C":
                                remap_center[tid] = tracker.selected_id
                                tracker.selected_id = tid
                            else:
                                remap_right[tid] = tracker.selected_id
                                tracker.selected_id = tid
                            
                            # ✅ origin_seg 업데이트 (세그먼트 이동 추적)
                            selected_bank.origin_seg = seg

                # 7) 드로잉
                def draw_tracks_on_tri(img, tracks, xoff, selected_id=None, remap=None, seg_name=""):
                    for tid, x1, y1, x2, y2 in tracks:
                        shown_id = remap.get(tid, tid) if remap else tid
                        is_sel = (selected_id is not None and shown_id == selected_id)
                        color = (0, 0, 255) if is_sel else (0, 255, 0)
                        X1, X2 = int(x1 + xoff), int(x2 + xoff)
                        cv2.rectangle(img, (X1, int(y1)), (X2, int(y2)), color, 2)
                        cv2.putText(img, f"ID {shown_id}", (X1, int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    
                    # 세그먼트 라벨
                    cv2.putText(img, f"[{seg_name}]", (xoff + 10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                draw_tracks_on_tri(tri, tracks_L, 0, tracker.selected_id, remap_left, "L")
                draw_tracks_on_tri(tri, tracks_C, seg_w, tracker.selected_id, remap_center, "C")
                draw_tracks_on_tri(tri, tracks_R, seg_w*2, tracker.selected_id, remap_right, "R")

                # 8) 통계 표시
                stats_text = f"UNIFIED: Tracks={unified_stats['total_tracks']} | Boundaries={unified_stats['boundary_transitions']} | ReID={unified_stats['reid_recoveries']}"
                cv2.putText(tri, stats_text, (10, tri.shape[0]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                disp_tri = tri

        # 히스토리 정리
        if frame_idx % HISTORY_CLEANUP_EVERY == 0:
            if tri_prepare:
                current_tids_L = {tid for tid, *_ in tracks_L}
                current_tids_C = {tid for tid, *_ in tracks_C}
                current_tids_R = {tid for tid, *_ in tracks_R}
                cleaned_L = track_history_L.cleanup_old_tracks(current_tids_L)
                cleaned_C = track_history_C.cleanup_old_tracks(current_tids_C)
                cleaned_R = track_history_R.cleanup_old_tracks(current_tids_R)
                if cleaned_C + cleaned_L + cleaned_R > 0:
                    print(f"[HISTORY] 정리: L={cleaned_L}, C={cleaned_C}, R={cleaned_R}")
            else:
                current_tids_C = {tid for tid, *_ in tracks}
                cleaned_C = track_history_C.cleanup_old_tracks(current_tids_C)
                if cleaned_C > 0:
                    print(f"[HISTORY] 정리: {cleaned_C}개 트랙")
        
        out_img = disp_tri if (tri_prepare and disp_tri is not None) else disp
        if out_img is None:
            continue
        
        cv2.imshow(WIN, out_img)
        if out_img is not None:
            BUS.publish(out_img)
        
        if need_resize and out_img is not None:
            fit_window_to_image(WIN, out_img)
            need_resize = False

    # 최종 통계 출력
    print("\n" + "="*60)
    print("🎯 UNIFIED TRACKER 성능 통계")
    print("="*60)
    print(f"총 추적 횟수: {unified_stats['total_tracks']}")
    print(f"경계 횡단 감지: {unified_stats['boundary_transitions']}")
    print(f"ReID 복구 성공: {unified_stats['reid_recoveries']}")
    print("="*60 + "\n")

    _CAMSTATS.save()
    print("[INFO] cam_stats 저장 완료: config/cam_stats.json")
    sm.stop()
    cv2.destroyAllWindows()


# ============================================================
# ✅ 차량 선택용 마우스 콜백
# ============================================================
def find_closest_track_in_history(track_history, click_x, click_y, max_frames_back=15, max_distance=150):
    """
    최근 N 프레임의 히스토리에서 클릭 위치에 가장 가까운 트랙 찾기
    """
    if not hasattr(track_history, 'history') or not track_history.history:
        return None, None, 0
    
    best_tid = None
    best_distance = max_distance
    best_frame_idx = -1
    total_checked = 0
    
    for tid, frames in track_history.history.items():
        if not frames or len(frames) == 0:
            continue
        
        recent_count = min(len(frames), max_frames_back)
        recent_frames = list(frames)[-recent_count:]
        
        for crop, bbox, fidx in recent_frames:
            total_checked += 1
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            distance = np.sqrt((cx - click_x)**2 + (cy - click_y)**2)
            
            if distance < best_distance:
                best_distance = distance
                best_tid = tid
                best_frame_idx = fidx
    
    if best_tid is not None:
        print(f"[HISTORY_HIT] ✅ ID {best_tid} 발견 (거리={best_distance:.1f}px, {total_checked}개 프레임 검색)")
    
    return best_tid, best_distance, total_checked


def mouse_callback(event, x, y, flags, param):
    """
    단일 모드 + Tri 모드 통합 마우스 콜백
    """
    global tri_selected, tracker, selected_bank, tri_prepare, tracks, tracks_L, tracks_C, tracks_R, scale, track_history_C, track_history_L, track_history_R, switcher, frame_idx
    
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    
    if tri_prepare:
        # ============================================================
        # Tri 모드: 세그먼트 판별
        # ============================================================
        if 'tri_ui_state' not in globals():
            return
        
        state = tri_ui_state
        seg_w = state.get("seg_w", 720)
        
        # 세그먼트 판별
        if x < seg_w:
            seg, seg_tracks, seg_hist = "L", tracks_L, track_history_L
            xoff = 0
        elif x < seg_w * 2:
            seg, seg_tracks, seg_hist = "C", tracks_C, track_history_C
            xoff = seg_w
        else:
            seg, seg_tracks, seg_hist = "R", tracks_R, track_history_R
            xoff = seg_w * 2
        
        # 로컬 좌표 변환
        orig_x = x - xoff
        orig_y = y
        
        print(f"[Mouse] Tri 모드 클릭: seg={seg}, 로컬=({orig_x},{orig_y})")
        
        clicked_id = None
        
        # 1) 현재 프레임에서 직접 히트
        for tid, bx1, by1, bx2, by2 in seg_tracks:
            if bx1 <= orig_x <= bx2 and by1 <= orig_y <= by2:
                clicked_id = tid
                print(f"[DIRECT_HIT] ✅ seg={seg} ID {tid} 선택")
                break
        
        # 2) 히스토리 검색
        if clicked_id is None:
            clicked_id, distance, checked = find_closest_track_in_history(
                seg_hist, orig_x, orig_y,
                max_frames_back=15,
                max_distance=150
            )
        
        # 3) 선택 처리
        if clicked_id is not None:
            tracker.selected_id = clicked_id
            cam_name = switcher.current_name if seg == "C" else (
                switcher.left_name if seg == "L" else switcher.right_name
            )
            
            print(f"[INFO] 🎯 차량 선택: seg={seg}, ID={clicked_id}, 카메라={cam_name}")
            
            # 히스토리 수집
            history_frames = seg_hist.get_history(clicked_id)
            collected = 0
            target = 5
            
            for crop, bbox, fidx in reversed(history_frames):
                if collected >= target:
                    break
                h, w = crop.shape[:2] if crop is not None else (0, 0)
                if h < 30 or w < 30:
                    continue
                
                try:
                    if selected_bank.add_from_frame_banded5_improved(
                        crop, (0, 0, w, h),
                        pad=0,
                        center_ratio=1.0,
                        origin_seg=seg,
                        origin_cam=cam_name,
                        cam_id=cam_name,
                        use_whitening=True
                    ):
                        collected += 1
                except:
                    pass
            
            print(f"[HISTORY] ✅ {collected}/{target}장 수집 완료")
        else:
            tracker.selected_id = None
            selected_bank.clear()
            print("[INFO] ❌ 클릭 영역 내 차량 없음")

    else:
        # ============================================================
        # 단일 모드 (기존 로직)
        # ============================================================
        try:
            current_scale = scale if scale > 0 else 1.0
        except:
            current_scale = 1.0
        
        orig_x = int(x / current_scale)
        orig_y = int(y / current_scale)
        
        clicked_id = None
        
        for tid, bx1, by1, bx2, by2 in tracks:
            if bx1 <= orig_x <= bx2 and by1 <= orig_y <= by2:
                clicked_id = tid
                break
        
        if clicked_id is None:
            clicked_id, distance, checked = find_closest_track_in_history(
                track_history_C, orig_x, orig_y,
                max_frames_back=15,
                max_distance=150
            )
        
        if clicked_id is not None:
            tracker.selected_id = clicked_id
            print(f"[INFO] 🎯 차량 선택: ID={clicked_id}")
            
            # 히스토리 수집
            history_frames = track_history_C.get_history(clicked_id)
            collected = 0
            target = 5
            
            for crop, bbox, fidx in reversed(history_frames):
                if collected >= target:
                    break
                h, w = crop.shape[:2] if crop is not None else (0, 0)
                if h < 30 or w < 30:
                    continue
                
                try:
                    if selected_bank.add_from_frame_banded5_improved(
                        crop, (0, 0, w, h),
                        pad=0,
                        center_ratio=1.0,
                        origin_seg="C",
                        origin_cam=switcher.current_name,
                        cam_id=switcher.current_name,
                        use_whitening=True
                    ):
                        collected += 1
                except:
                    pass
            
            print(f"[HISTORY] ✅ {collected}/{target}장 수집 완료")
        else:
            tracker.selected_id = None
            selected_bank.clear()


def bhatta_dist_for_box(bank, frame, box):
    """5밴드 우선 사용"""
    if hasattr(bank, "avg_bhatta_to_box_banded5_improved"):
        result = bank.avg_bhatta_to_box_banded5_improved(frame, box)
        if result is not None and result[0] is not None:
            return result[0]
    
    if hasattr(bank, "avg_bhatta_to_box_banded"):
        return bank.avg_bhatta_to_box_banded(frame, box)
    
    if hasattr(bank, "avg_bhatta_to_box"):
        return bank.avg_bhatta_to_box(frame, box)
    
    if hasattr(bank, "score_to_gallery"):
        return bank.score_to_gallery(frame, box)
    
    return None


def get_adaptive_threshold(selected_bank, current_seg, current_cam=None, frame_brightness=None):
    """세그먼트별 임계값"""
    same_seg_thresh = 0.50
    other_seg_thresh = REID_THRESH_OTHER
    
    if frame_brightness is not None and frame_brightness < 100:
        other_seg_thresh += 0.05
    
    origin_seg = getattr(selected_bank, "origin_seg", None)
    if origin_seg is None:
        return same_seg_thresh
    
    if origin_seg == current_seg:
        return same_seg_thresh
    else:
        return other_seg_thresh


def log_bhatta5(cam_name: str, selected_id: int, lines: list, root="reid_logs_5band"):
    cam_dir = _sanitize(cam_name)
    out_dir = Path(root) / cam_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"id_{selected_id}.txt"
    with open(path, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


if __name__ == "__main__":
    run_detect()