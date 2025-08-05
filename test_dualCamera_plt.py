"""
핸드오버 시스템 통합 메인 스크립트 (디버깅 포함)
파일명: main_handover_system.py

기존 test_dualCamera_plt.py를 기반으로 새로운 핸드오버 시스템 통합
bbox 좌표 문제 디버깅 기능 포함
"""

import cv2
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dotenv import load_dotenv

# 우리가 만든 모듈들 import
from core.data_manager import DataManager
from handover.frame_concatenator import FrameConcatenator, BBoxSeparator
from handover.coordinate_transformer import CoordinateTransformer
from handover.handover_manager import HandoverManager, HandoverState
from ui.simple_handover_ui import SimpleHandoverUI, UIMode

# 기존 모듈들 import
from detector.yolo_detector import get_vehicle_detections
from tracker.tracker_test import MultiTracker, check_boundary_event
from reid.feature_extractor import ReIDSystem

class IntegratedHandoverSystem:
    """통합 핸드오버 시스템 (디버깅 포함)"""
    
    def __init__(self):
        print("🚀 통합 핸드오버 시스템 초기화 중...")
        
        # === 우리가 만든 새로운 모듈들 ===
        self.data_manager = DataManager()
        self.frame_concatenator = FrameConcatenator()
        self.bbox_separator = BBoxSeparator()
        self.coord_transformer = CoordinateTransformer()
        self.handover_manager = HandoverManager(self.data_manager)
        self.ui_system = SimpleHandoverUI()
        
        # === 기존 시스템들 ===
        self.tracker = MultiTracker()
        self.reid_system = ReIDSystem(similarity_threshold=0.7)
        
        # === 카메라 관리 ===
        self.current_cap = None
        self.secondary_cap = None
        self.current_cctv = None
        self.secondary_cctv = None
        
        # === matplotlib 설정 ===
        plt.ion()
        self.fig = None
        self.ax = None
        
        # === 성능 설정 ===
        self.detection_interval = 2  # 개선: 2프레임마다
        self.frame_counter = 0
        self.last_detections = []
        
        # === 상태 관리 ===
        self.selected_vehicle_id = None
        self.reid_registered = set()
        
        # === 디버깅 설정 ===
        self.debug_mode = False
        
        print("✅ 시스템 초기화 완료")
    
    def apply_debug_mode(self):
        """디버깅 모드 적용"""
        print("🐛 디버깅 모드 활성화")
        self.debug_mode = True
        print("  - 상세한 좌표 로그 출력")
        print("  - bbox 유효성 검사")
        print("  - 클릭 이벤트 추적")
    
    def initialize_modules(self):
        """모듈 간 연결 설정"""
        # handover_manager에 필요한 모듈들 주입
        self.handover_manager.frame_concatenator = self.frame_concatenator
        self.handover_manager.bbox_separator = self.bbox_separator
        self.handover_manager.coord_transformer = self.coord_transformer
        self.handover_manager.initialize_modules()
        
        print("🔗 모듈 간 연결 완료")
    
    def setup_matplotlib(self):
        """matplotlib 초기화"""
        if self.fig:
            plt.close(self.fig)
        
        self.fig = plt.figure(figsize=(16, 10))
        self.ax = self.fig.add_subplot(111)
        
        # 클릭 이벤트 연결
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        print("📊 matplotlib 설정 완료")
    
    def start_camera(self, cctv_name: str, stream_url: str) -> bool:
        """카메라 시작"""
        print(f"📡 카메라 연결 시도: {cctv_name}")
        
        # 카메라 정보 찾기
        camera_connections = self.data_manager.get_camera_connections()
        self.current_cctv = None
        
        for cctv_info in camera_connections:
            if cctv_name in cctv_info["cctvname"]:
                self.current_cctv = cctv_info
                break
        
        if not self.current_cctv:
            print(f"❌ 카메라 정보를 찾을 수 없음: {cctv_name}")
            return False
        
        # 카메라 연결
        self.current_cap = cv2.VideoCapture(stream_url)
        if not self.current_cap.isOpened():
            print("❌ 카메라 연결 실패")
            return False
        
        self.current_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # UI를 단일 모드로 설정
        self.ui_system.set_single_mode(self.current_cctv["cctvname"])
        self.data_manager.set_ui_mode("single", self.current_cctv["cctvname"])
        
        print(f"✅ 카메라 연결 성공: {cctv_name}")
        return True
    
    def get_frames(self) -> tuple:
        """프레임 읽기"""
        current_frame = None
        secondary_frame = None
        
        # 현재 카메라
        if self.current_cap:
            ret, current_frame = self.current_cap.read()
            if not ret:
                current_frame = None
        
        # 보조 카메라 (듀얼 모드시)
        if self.secondary_cap:
            ret, secondary_frame = self.secondary_cap.read()
            if not ret:
                secondary_frame = None
        
        return current_frame, secondary_frame
    
    def validate_bbox(self, bbox, frame_width, frame_height):
        """bbox 좌표 유효성 검사"""
        x1, y1, x2, y2 = bbox
        
        issues = []
        
        if x1 < 0: issues.append(f"x1({x1}) < 0")
        if y1 < 0: issues.append(f"y1({y1}) < 0")
        if x2 > frame_width: issues.append(f"x2({x2}) > width({frame_width})")
        if y2 > frame_height: issues.append(f"y2({y2}) > height({frame_height})")
        if x2 <= x1: issues.append(f"x2({x2}) <= x1({x1})")
        if y2 <= y1: issues.append(f"y2({y2}) <= y1({y1})")
        
        return len(issues) == 0, issues
    
    def clip_bbox(self, bbox, frame_width, frame_height):
        """bbox를 프레임 범위로 클리핑"""
        x1, y1, x2, y2 = bbox
        
        x1 = max(0, min(x1, frame_width-1))
        y1 = max(0, min(y1, frame_height-1))
        x2 = max(x1+1, min(x2, frame_width))
        y2 = max(y1+1, min(y2, frame_height))
        
        return [x1, y1, x2, y2]
    
    def process_detections(self, frame: np.ndarray) -> list:
        """개선된 탐지 처리 (디버깅 포함)"""
        self.frame_counter += 1
        
        if self.frame_counter % self.detection_interval == 0:
            original_h, original_w = frame.shape[:2]
            target_w, target_h = 800, 600
            
            if self.debug_mode:
                print(f"🔍 원본 프레임 크기: {original_w}x{original_h}")
                print(f"🔍 YOLO 입력 크기: {target_w}x{target_h}")
            
            # 리사이즈
            small_frame = cv2.resize(frame, (target_w, target_h))
            
            # YOLO 탐지
            detections = get_vehicle_detections(small_frame, conf_threshold=0.2)
            
            if self.debug_mode:
                print(f"🔍 원본 detection 수: {len(detections) if detections else 0}")
            
            if detections:
                # 스케일 계산
                scale_x = original_w / target_w
                scale_y = original_h / target_h
                
                if self.debug_mode:
                    print(f"🔍 스케일: x={scale_x:.3f}, y={scale_y:.3f}")
                
                self.last_detections = []
                for i, det in enumerate(detections):
                    if len(det) >= 4:
                        x1, y1, x2, y2 = det[:4]
                        conf = det[4] if len(det) > 4 else 0.5
                        
                        if self.debug_mode:
                            print(f"🔍 Detection {i}: 원본({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                        
                        # 좌표 복원
                        scaled_x1 = int(x1 * scale_x)
                        scaled_y1 = int(y1 * scale_y)
                        scaled_x2 = int(x2 * scale_x)
                        scaled_y2 = int(y2 * scale_y)
                        
                        if self.debug_mode:
                            print(f"🔍 Detection {i}: 복원({scaled_x1},{scaled_y1},{scaled_x2},{scaled_y2})")
                        
                        # bbox 유효성 검사
                        bbox = [scaled_x1, scaled_y1, scaled_x2, scaled_y2]
                        is_valid, issues = self.validate_bbox(bbox, original_w, original_h)
                        
                        if not is_valid and self.debug_mode:
                            print(f"⚠️ Detection {i}: 좌표 문제 - {issues}")
                        
                        # 클리핑
                        clipped_bbox = self.clip_bbox(bbox, original_w, original_h)
                        scaled_x1, scaled_y1, scaled_x2, scaled_y2 = clipped_bbox
                        
                        # 크기 필터링
                        width = scaled_x2 - scaled_x1
                        height = scaled_y2 - scaled_y1
                        
                        if self.debug_mode:
                            print(f"🔍 Detection {i}: 최종({scaled_x1},{scaled_y1},{scaled_x2},{scaled_y2}) 크기({width}x{height})")
                        
                        if width > 20 and height > 20:
                            self.last_detections.append((
                                scaled_x1, scaled_y1, scaled_x2, scaled_y2, conf
                            ))
                            if self.debug_mode:
                                print(f"✅ Detection {i}: 유효함")
                        else:
                            if self.debug_mode:
                                print(f"❌ Detection {i}: 너무 작음")
            else:
                self.last_detections = []
                if self.debug_mode:
                    print("🔍 탐지된 객체 없음")
        
        return self.last_detections
    
    def process_handover_logic(self, current_frame: np.ndarray, tracks: list):
        """핸드오버 로직 처리"""
        if not tracks or not self.selected_vehicle_id:
            return
        
        # 선택된 차량 찾기
        selected_track = None
        for track_data in tracks:
            track_id = track_data[0]
            if track_id == self.selected_vehicle_id:
                selected_track = track_data
                break
        
        if not selected_track:
            return
        
        track_id, x1, y1, x2, y2 = selected_track
        bbox = [x1, y1, x2, y2]
        
        # 핸드오버 트리거 확인
        if self.handover_manager.check_handover_trigger(
            str(track_id), bbox, self.current_cctv["cctvname"]
        ):
            # 진행 방향 추정 (간단한 로직)
            h, w = current_frame.shape[:2]
            center_x = (x1 + x2) // 2
            
            if center_x < w // 3:
                direction = "south"
            elif center_x > 2 * w // 3:
                direction = "north"
            else:
                return
            
            # 다음 카메라 찾기
            next_camera_name = self.data_manager.get_next_camera(
                self.current_cctv["cctvname"], direction
            )
            
            if next_camera_name:
                if self.debug_mode:
                    print(f"🔄 핸드오버 시작: {direction} → {next_camera_name}")
                self.start_handover(next_camera_name)
    
    def start_handover(self, next_camera_name: str):
        """핸드오버 시작"""
        # 핸드오버 매니저로 시작
        success = self.handover_manager.start_handover(
            str(self.selected_vehicle_id),
            self.current_cctv["cctvname"],
            next_camera_name
        )
        
        if success:
            # 보조 카메라 연결 (실제로는 같은 스트림, 테스트용)
            stream_url = os.getenv("CURRENT_CCTV_URL", "")
            self.secondary_cap = cv2.VideoCapture(stream_url)
            
            if self.secondary_cap.isOpened():
                self.secondary_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # UI를 듀얼 모드로 전환
                self.ui_system.set_dual_mode(
                    self.current_cctv["cctvname"], 
                    next_camera_name
                )
                
                # 핸드오버 상태 업데이트
                self.ui_system.update_handover_status(0.1, "핸드오버 시작", 0.0)
    
    def update_handover_state(self):
        """핸드오버 상태 업데이트"""
        if self.handover_manager.current_state == HandoverState.IDLE:
            return
        
        # 핸드오버 매니저 상태 업데이트
        handover_status = self.handover_manager.update_handover_state()
        
        # UI 상태 업데이트
        if handover_status:
            progress = handover_status.get("progress", 0.0)
            message = handover_status.get("message", "")
            elapsed = handover_status.get("elapsed_time", 0.0)
            
            self.ui_system.update_handover_status(progress, message, elapsed)
            
            # 핸드오버 완료시 단일 모드로 복귀
            if handover_status.get("state") in ["success", "timeout"]:
                self.end_handover()
    
    def end_handover(self):
        """핸드오버 종료"""
        if self.secondary_cap:
            self.secondary_cap.release()
            self.secondary_cap = None
        
        # UI를 단일 모드로 복귀
        self.ui_system.set_single_mode(self.current_cctv["cctvname"])
        
        if self.debug_mode:
            print("🛑 핸드오버 종료")
    
    def create_display_frame(self, current_frame: np.ndarray, secondary_frame: np.ndarray = None) -> np.ndarray:
        """표시용 프레임 생성"""
        frame_dict = {}
        
        if current_frame is not None:
            frame_dict[self.current_cctv["cctvname"]] = current_frame
        
        if secondary_frame is not None and self.ui_system.secondary_camera:
            frame_dict[self.ui_system.secondary_camera] = secondary_frame
        
        display_frame, transform_info = self.ui_system.create_display_frame(frame_dict)
        
        return display_frame
    
    def draw_tracks_on_matplotlib(self, display_frame: np.ndarray, tracks: list):
        """matplotlib에 트랙 그리기 (디버깅 포함)"""
        self.ax.clear()
        
        # 프레임 정보 출력
        frame_h, frame_w = display_frame.shape[:2]
        if self.debug_mode:
            print(f"🖼️ 표시 프레임 크기: {frame_w}x{frame_h}")
            print(f"🖼️ UI 모드: {self.ui_system.current_mode.value}")
        
        # 프레임 표시
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        self.ax.imshow(frame_rgb)
        
        # 트랙 그리기 (단일 모드에서만, 듀얼 모드는 UI에서 처리)
        if self.ui_system.current_mode == UIMode.SINGLE and tracks:
            if self.debug_mode:
                print(f"🎯 그릴 트랙 수: {len(tracks)}")
            
            for i, track_data in enumerate(tracks):
                track_id, x1, y1, x2, y2 = track_data
                width = x2 - x1
                height = y2 - y1
                
                if self.debug_mode:
                    print(f"🎯 트랙 {i} (ID{track_id}): ({x1},{y1},{x2},{y2}) 크기({width}x{height})")
                
                # 좌표 유효성 검사
                if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
                    if self.debug_mode:
                        print(f"⚠️ 트랙 {i}: 좌표가 프레임 범위를 벗어남!")
                    
                    # 클리핑
                    x1 = max(0, x1)
                    y1 = max(0, y1) 
                    x2 = min(frame_w, x2)
                    y2 = min(frame_h, y2)
                    width = x2 - x1
                    height = y2 - y1
                    
                    if self.debug_mode:
                        print(f"🔧 트랙 {i}: 클리핑 후 ({x1},{y1},{x2},{y2})")
                
                # 색상 선택
                if track_id == self.selected_vehicle_id:
                    color = 'magenta'
                    linewidth = 3
                elif track_id in self.reid_registered:
                    color = 'yellow'
                    linewidth = 2
                else:
                    color = 'red'
                    linewidth = 2
                
                # 사각형 그리기
                try:
                    rect = Rectangle((x1, y1), width, height,
                                   linewidth=linewidth, edgecolor=color, facecolor='none')
                    self.ax.add_patch(rect)
                    
                    if self.debug_mode:
                        print(f"✅ 트랙 {i}: 사각형 그리기 성공")
                    
                    # 라벨
                    label = f"ID{track_id}"
                    if track_id == self.selected_vehicle_id:
                        label += " [SELECTED]"
                    
                    # 라벨 위치도 확인
                    label_y = max(10, y1-5)  # 화면 위쪽 경계 고려
                    self.ax.text(x1, label_y, label, color=color, fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.5))
                    
                except Exception as e:
                    if self.debug_mode:
                        print(f"❌ 트랙 {i}: 그리기 실패 - {e}")
        
        # 상태 정보
        status_parts = []
        status_parts.append(f"Frame: {self.frame_counter}")
        status_parts.append(f"Mode: {self.ui_system.current_mode.value.upper()}")
        status_parts.append(f"Size: {frame_w}x{frame_h}")
        
        if self.selected_vehicle_id:
            status_parts.append(f"Selected: ID{self.selected_vehicle_id}")
        
        if self.handover_manager.current_state != HandoverState.IDLE:
            status_parts.append(f"Handover: {self.handover_manager.current_state.value}")
        
        if self.debug_mode:
            status_parts.append("DEBUG")
        
        self.fig.suptitle(" | ".join(status_parts), fontsize=12)
        self.ax.axis('off')
        
        # matplotlib axes 범위 출력
        if self.debug_mode:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            print(f"🖼️ matplotlib 범위: x({xlim[0]:.1f},{xlim[1]:.1f}) y({ylim[0]:.1f},{ylim[1]:.1f})")
        
        plt.pause(0.01)
    
    def on_click(self, event):
        """마우스 클릭 이벤트 (디버깅 포함)"""
        if not event.inaxes == self.ax:
            if self.debug_mode:
                print("🖱️ 클릭: axes 범위 밖")
            return
        
        # 클릭 좌표
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            if self.debug_mode:
                print("🖱️ 클릭: 좌표 없음")
            return
        
        click_x, click_y = int(click_x), int(click_y)
        if self.debug_mode:
            print(f"🖱️ 클릭 좌표: ({click_x}, {click_y})")
        
        # 현재 표시 중인 프레임 크기
        current_frame, _ = self.get_frames()
        if current_frame is not None and self.debug_mode:
            orig_h, orig_w = current_frame.shape[:2]
            print(f"🖱️ 원본 프레임 크기: {orig_w}x{orig_h}")
        
        # UI 시스템으로 클릭 처리
        click_result = self.ui_system.handle_click(click_x, click_y)
        if self.debug_mode:
            print(f"🖱️ UI 클릭 결과: {click_result}")
        
        if click_result["success"] and self.ui_system.current_mode == UIMode.SINGLE:
            # 트랙 선택 시도
            if self.debug_mode:
                print(f"🖱️ 트랙 선택 시도: ({click_x}, {click_y})")
            
            try:
                selected = self.tracker.select_track_by_point(click_x, click_y)
                if self.debug_mode:
                    print(f"🖱️ 트래커 선택 결과: {selected}")
                
                if selected:
                    track_id = selected['id']
                    bbox = selected['bbox']
                    if self.debug_mode:
                        print(f"🎯 선택된 트랙: ID{track_id}, bbox{bbox}")
                    
                    self.selected_vehicle_id = track_id
                    self.data_manager.set_selected_vehicle(str(track_id))
                    
                    # ReID 등록
                    if current_frame is not None:
                        self.register_vehicle_for_reid(track_id, current_frame, bbox)
                else:
                    if self.debug_mode:
                        print("🖱️ 선택된 트랙 없음")
                        
            except Exception as e:
                if self.debug_mode:
                    print(f"❌ 트랙 선택 오류: {e}")
                    import traceback
                    traceback.print_exc()
    
    def register_vehicle_for_reid(self, track_id: int, frame: np.ndarray, bbox: list):
        """ReID를 위한 차량 등록"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # 경계 확인
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        if self.debug_mode:
            print(f"🎯 ReID 등록: ID{track_id}, bbox({x1},{y1},{x2},{y2})")
        
        # 차량 이미지 추출
        vehicle_crop = frame[y1:y2, x1:x2]
        
        if vehicle_crop.size > 0:
            # ReID 시스템에 등록
            self.reid_system.register_lost_vehicle(
                track_id, vehicle_crop, bbox, 'car',
                {'cctv': self.current_cctv['cctvname']}
            )
            
            # 데이터 매니저에 차량 추가
            self.data_manager.add_vehicle(
                str(track_id),
                self.current_cctv['cctvname'],
                {"x": x1, "y": y1, "w": x2-x1, "h": y2-y1},
                "car"
            )
            
            self.reid_registered.add(track_id)
            print(f"✅ ID{track_id} ReID 등록 완료")
        else:
            if self.debug_mode:
                print(f"❌ ID{track_id} ReID 등록 실패: 빈 이미지")
    
    def run(self):
        """메인 실행 루프"""
        print("\n🚀 통합 핸드오버 시스템 시작!")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("사용법:")
        print("  🖱️  마우스 클릭: 차량 선택 및 추적 시작")
        print("  🔄  자동 핸드오버: 선택된 차량이 화면 경계 근처에 도달시")
        print("  ⌨️  키보드 'q': 종료")
        if self.debug_mode:
            print("  🐛  디버깅 모드: 상세 로그 출력")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        frame_count = 0
        fps_start = time.time()
        
        try:
            while True:
                # 프레임 읽기
                current_frame, secondary_frame = self.get_frames()
                
                if current_frame is None:
                    if self.debug_mode and frame_count % 100 == 0:  # 가끔만 출력
                        print("❌ 프레임 읽기 실패")
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                
                # 프레임 등록 (핸드오버 매니저용)
                self.handover_manager.update_frame(
                    self.current_cctv["cctvname"], current_frame
                )
                
                # 탐지 처리
                detections = self.process_detections(current_frame)
                
                # 트래커 업데이트
                tracks = self.tracker.update(detections)
                
                # 핸드오버 로직 처리
                self.process_handover_logic(current_frame, tracks)
                
                # 핸드오버 상태 업데이트
                self.update_handover_state()
                
                # 표시용 프레임 생성
                display_frame = self.create_display_frame(current_frame, secondary_frame)
                
                # matplotlib에 그리기
                self.draw_tracks_on_matplotlib(display_frame, tracks)
                
                # FPS 계산
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start
                    fps = 30 / elapsed if elapsed > 0 else 0
                    status_msg = f"📊 FPS: {fps:.1f} | 프레임: {frame_count} | 모드: {self.ui_system.current_mode.value}"
                    if self.debug_mode:
                        status_msg += f" | 탐지: {len(detections)} | 트랙: {len(tracks) if tracks else 0}"
                    print(status_msg)
                    fps_start = time.time()
        
        except KeyboardInterrupt:
            print("\n⌨️ 사용자 중단")
        except Exception as e:
            print(f"\n💥 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()
    
    def shutdown(self):
        """시스템 종료"""
        print("\n🛑 시스템 종료 중...")
        
        if self.current_cap:
            self.current_cap.release()
        
        if self.secondary_cap:
            self.secondary_cap.release()
        
        if self.fig:
            plt.close(self.fig)
        
        plt.ioff()
        
        print("✅ 시스템 종료 완료")







def main():
    """메인 함수"""
    load_dotenv()
    
    # 환경변수에서 설정 읽기
    stream_url = os.getenv("CURRENT_CCTV_URL", "")
    cctv_name = os.getenv("CURRENT_CCTV_NAME", "죽평")
    
    if not stream_url:
        print("❌ CURRENT_CCTV_URL 환경변수가 설정되지 않았습니다.")
        return
    
    print("🎬 통합 핸드오버 시스템")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📹 카메라: {cctv_name}")
    print(f"🔗 스트림: {stream_url[:50]}...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # 시스템 초기화
    system = IntegratedHandoverSystem()
    
    # 디버깅 모드 활성화 (bbox 좌표 문제 해결용)
    system.apply_debug_mode()  # 이 줄을 주석 처리하면 디버깅 끄기
    
    system.initialize_modules()
    system.setup_matplotlib()
    
    # 카메라 시작
    if system.start_camera(cctv_name, stream_url):
        system.run()
    else:
        print("❌ 시스템 시작 실패")


def main_without_debug():
    """디버깅 없이 실행"""
    load_dotenv()
    
    stream_url = os.getenv("CURRENT_CCTV_URL", "")
    cctv_name = os.getenv("CURRENT_CCTV_NAME", "죽평")
    
    if not stream_url:
        print("❌ CURRENT_CCTV_URL 환경변수가 설정되지 않았습니다.")
        return
    
    print("🎬 통합 핸드오버 시스템 (일반 모드)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📹 카메라: {cctv_name}")
    print(f"🔗 스트림: {stream_url[:50]}...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # 시스템 초기화 (디버깅 모드 없이)
    system = IntegratedHandoverSystem()
    system.initialize_modules()
    system.setup_matplotlib()
    
    # 카메라 시작
    if system.start_camera(cctv_name, stream_url):
        system.run()
    else:
        print("❌ 시스템 시작 실패")


if __name__ == "__main__":
    # 디버깅 모드로 실행
    main()
    
    # 일반 모드로 실행하려면 대신 이것 사용:
    # main_without_debug()


    # main_handover_system.py에 추가할 HLS 점프 분석 코드



class HLSJumpDetector:
    """HLS 세그먼트 점프 감지기"""
    
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.jump_history = []
        self.last_frame_time = None
        self.frame_counter = 0
        
    def detect_jump(self, frame):
        """프레임에서 점프 감지"""
        current_time = time.time()
        self.frame_counter += 1
        
        if self.last_frame_time is not None:
            # 프레임 간 시간 간격
            interval = current_time - self.last_frame_time
            self.frame_times.append(interval)
            
            # 평균 프레임 간격 계산
            if len(self.frame_times) > 10:
                avg_interval = sum(self.frame_times) / len(self.frame_times)
                
                # 비정상적으로 긴 간격 감지 (점프)
                if interval > avg_interval * 3:  # 평균의 3배 이상
                    jump_info = {
                        'frame_count': self.frame_counter,
                        'time': current_time,
                        'interval': interval,
                        'avg_interval': avg_interval,
                        'jump_size': interval - avg_interval
                    }
                    
                    self.jump_history.append(jump_info)
                    
                    print(f"🔥 HLS 점프 감지!")
                    print(f"  프레임: {self.frame_counter}")
                    print(f"  간격: {interval:.3f}초 (평균: {avg_interval:.3f}초)")
                    print(f"  점프 크기: {jump_info['jump_size']:.3f}초")
                    
                    return True
        
        self.last_frame_time = current_time
        return False
    
    def get_jump_pattern(self):
        """점프 패턴 분석"""
        if len(self.jump_history) < 2:
            return None
        
        # 점프 간 간격 계산
        intervals = []
        for i in range(1, len(self.jump_history)):
            time_diff = self.jump_history[i]['time'] - self.jump_history[i-1]['time']
            intervals.append(time_diff)
        
        if intervals:
            avg_jump_interval = sum(intervals) / len(intervals)
            print(f"📊 점프 패턴 분석:")
            print(f"  총 점프 수: {len(self.jump_history)}")
            print(f"  평균 점프 간격: {avg_jump_interval:.1f}초")
            print(f"  점프 간격들: {[f'{i:.1f}s' for i in intervals[-5:]]}")  # 최근 5개
            
            return {
                'total_jumps': len(self.jump_history),
                'avg_interval': avg_jump_interval,
                'recent_intervals': intervals[-5:]
            }
        
        return None

# IntegratedHandoverSystem 클래스에 추가할 메서드들
def initialize_jump_detection(self):
    """점프 감지기 초기화"""
    self.jump_detector = HLSJumpDetector()
    print("🔍 HLS 점프 감지기 활성화")

def monitor_stream_continuity(self, frame):
    """스트림 연속성 모니터링"""
    if hasattr(self, 'jump_detector'):
        jump_detected = self.jump_detector.detect_jump(frame)
        
        if jump_detected:
            # 점프 발생시 대응
            self.handle_stream_jump()
            
            # 패턴 분석 (10번째 점프마다)
            if len(self.jump_detector.jump_history) % 10 == 0:
                pattern = self.jump_detector.get_jump_pattern()
                if pattern and pattern['avg_interval'] > 0:
                    print(f"💡 예상 다음 점프: {pattern['avg_interval']:.1f}초 후")

def handle_stream_jump(self):
    """스트림 점프 발생시 처리"""
    print("🔧 스트림 점프 처리 중...")
    
    # 1. 트래커 상태 유지 (중요!)
    if hasattr(self, 'tracker') and self.selected_vehicle_id:
        print(f"📌 선택된 차량 ID{self.selected_vehicle_id} 상태 보존")
    
    # 2. 캐시된 detection 초기화
    self.last_detections = []
    
    # 3. 필요시 스트림 재연결 시도
    if self.should_reconnect_stream():
        self.reconnect_stream()

def should_reconnect_stream(self):
    """스트림 재연결 필요 여부 판단"""
    if hasattr(self, 'jump_detector'):
        # 최근 1분내 점프가 너무 많으면 재연결
        recent_jumps = [j for j in self.jump_detector.jump_history 
                       if time.time() - j['time'] < 60]
        
        if len(recent_jumps) > 10:  # 1분에 10번 이상 점프
            print("⚠️ 과도한 점프 발생, 재연결 권장")
            return True
    
    return False

def reconnect_stream(self):
    """스트림 재연결"""
    print("🔄 스트림 재연결 시도...")
    
    if self.current_cap:
        self.current_cap.release()
        time.sleep(0.5)  # 짧은 대기
        
        # 재연결
        stream_url = os.getenv("CURRENT_CCTV_URL", "")
        self.current_cap = self.setup_camera_optimized(stream_url)
        
        if self.current_cap:
            print("✅ 스트림 재연결 성공")
            return True
        else:
            print("❌ 스트림 재연결 실패")
            return False

def setup_camera_optimized(self, stream_url: str):
    """HLS 최적화된 카메라 설정"""
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        return None
    
    # HLS 점프 최소화 설정
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # 최소 버퍼로 지연 최소화
    cap.set(cv2.CAP_PROP_FPS, 25)          # 안정적인 FPS
    
    # 추가 안정성 옵션
    try:
        # 자동 재연결 허용
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    except:
        pass
    
    return cap

# 메인 실행 루프에서 사용
def run_with_jump_monitoring(self):
    """점프 모니터링이 포함된 실행 루프"""
    
    # 점프 감지기 초기화
    self.initialize_jump_detection()
    
    print("\n🚀 HLS 점프 모니터링 포함 실행!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    frame_count = 0
    fps_start = time.time()
    
    try:
        while True:
            current_frame, secondary_frame = self.get_frames()
            
            if current_frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # HLS 점프 모니터링 (핵심!)
            self.monitor_stream_continuity(current_frame)
            
            # 나머지 처리는 기존과 동일
            self.handover_manager.update_frame(
                self.current_cctv["cctvname"], current_frame
            )
            
            detections = self.process_detections(current_frame)
            tracks = self.tracker.update(detections)
            
            self.process_handover_logic(current_frame, tracks)
            self.update_handover_state()
            
            display_frame = self.create_display_frame(current_frame, secondary_frame)
            self.draw_tracks_on_matplotlib(display_frame, tracks)
            
            # FPS 및 점프 통계
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start
                fps = 30 / elapsed if elapsed > 0 else 0
                
                jump_count = len(self.jump_detector.jump_history) if hasattr(self, 'jump_detector') else 0
                print(f"📊 FPS: {fps:.1f} | 프레임: {frame_count} | 점프: {jump_count}회")
                
                fps_start = time.time()
    
    except KeyboardInterrupt:
        print("\n⌨️ 사용자 중단")
        
        # 최종 점프 패턴 분석
        if hasattr(self, 'jump_detector'):
            final_pattern = self.jump_detector.get_jump_pattern()
            if final_pattern:
                print(f"\n📈 최종 점프 분석:")
                print(f"  전체 실행 중 {final_pattern['total_jumps']}회 점프 발생")
                print(f"  평균 {final_pattern['avg_interval']:.1f}초 간격")
    
    finally:
        self.shutdown()

# 사용법: main 함수에서
def main_with_jump_detection():
    """점프 감지 포함 실행"""
    load_dotenv()
    
    stream_url = os.getenv("CURRENT_CCTV_URL", "")
    cctv_name = os.getenv("CURRENT_CCTV_NAME", "죽평")
    
    if not stream_url:
        print("❌ CURRENT_CCTV_URL 환경변수가 설정되지 않았습니다.")
        return
    
    system = IntegratedHandoverSystem()
    system.apply_debug_mode()
    system.initialize_modules()
    system.setup_matplotlib()
    
    if system.start_camera(cctv_name, stream_url):
        system.run_with_jump_monitoring()  # 점프 모니터링 포함 실행
    else:
        print("❌ 시스템 시작 실패")

if __name__ == "__main__":
    main_with_jump_detection()  # 점프 분석 버전으로 실행