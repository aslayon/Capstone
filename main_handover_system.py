"""
핸드오버 시스템 통합 메인 스크립트
파일명: main_handover_system.py

기존 test_dualCamera_plt.py를 기반으로 새로운 핸드오버 시스템 통합
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
    """통합 핸드오버 시스템"""
    
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
        
        print("✅ 시스템 초기화 완료")
    
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
    
    def process_detections(self, frame: np.ndarray) -> list:
        """개선된 탐지 처리"""
        self.frame_counter += 1
        
        if self.frame_counter % self.detection_interval == 0:
            # 개선: 더 큰 해상도
            small_frame = cv2.resize(frame, (800, 600))
            
            # 개선: 더 낮은 임계값
            detections = get_vehicle_detections(small_frame, conf_threshold=0.2)
            
            if detections:
                scale_x = frame.shape[1] / 800
                scale_y = frame.shape[0] / 600
                
                self.last_detections = []
                for det in detections:
                    if len(det) >= 4:
                        x1, y1, x2, y2 = det[:4]
                        
                        # 좌표 복원
                        scaled_x1 = int(x1 * scale_x)
                        scaled_y1 = int(y1 * scale_y)
                        scaled_x2 = int(x2 * scale_x)
                        scaled_y2 = int(y2 * scale_y)
                        
                        # 크기 필터링 (개선)
                        width = scaled_x2 - scaled_x1
                        height = scaled_y2 - scaled_y1
                        
                        if width > 20 and height > 20:
                            self.last_detections.append((
                                scaled_x1, scaled_y1, scaled_x2, scaled_y2
                            ) + det[4:])
            else:
                self.last_detections = []
        
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
        """matplotlib에 트랙 그리기"""
        self.ax.clear()
        
        # 프레임 표시
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        self.ax.imshow(frame_rgb)
        
        # 트랙 그리기 (단일 모드에서만, 듀얼 모드는 UI에서 처리)
        if self.ui_system.current_mode == UIMode.SINGLE and tracks:
            for track_data in tracks:
                track_id, x1, y1, x2, y2 = track_data
                width = x2 - x1
                height = y2 - y1
                
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
                rect = Rectangle((x1, y1), width, height,
                               linewidth=linewidth, edgecolor=color, facecolor='none')
                self.ax.add_patch(rect)
                
                # 라벨
                label = f"ID{track_id}"
                if track_id == self.selected_vehicle_id:
                    label += " [SELECTED]"
                
                self.ax.text(x1, y1-5, label, color=color, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.5))
        
        # 상태 정보
        status_parts = []
        status_parts.append(f"Frame: {self.frame_counter}")
        status_parts.append(f"Mode: {self.ui_system.current_mode.value.upper()}")
        
        if self.selected_vehicle_id:
            status_parts.append(f"Selected: ID{self.selected_vehicle_id}")
        
        if self.handover_manager.current_state != HandoverState.IDLE:
            status_parts.append(f"Handover: {self.handover_manager.current_state.value}")
        
        self.fig.suptitle(" | ".join(status_parts), fontsize=12)
        self.ax.axis('off')
        
        plt.pause(0.01)
    
    def on_click(self, event):
        """마우스 클릭 이벤트"""
        if not event.inaxes == self.ax:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        # 클릭 처리
        click_result = self.ui_system.handle_click(x, y)
        
        if click_result["success"] and self.ui_system.current_mode == UIMode.SINGLE:
            # 트랙 선택
            selected = self.tracker.select_track_by_point(x, y)
            
            if selected:
                track_id = selected['id']
                self.selected_vehicle_id = track_id
                
                # 데이터 매니저에 선택된 차량 기록
                self.data_manager.set_selected_vehicle(str(track_id))
                
                # ReID 시스템에 등록
                current_frame, _ = self.get_frames()
                if current_frame is not None:
                    bbox = selected['bbox']
                    self.register_vehicle_for_reid(track_id, current_frame, bbox)
                
                print(f"🎯 차량 선택: ID{track_id}")
    
    def register_vehicle_for_reid(self, track_id: int, frame: np.ndarray, bbox: list):
        """ReID를 위한 차량 등록"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # 경계 확인
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
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
    
    def run(self):
        """메인 실행 루프"""
        print("\n🚀 통합 핸드오버 시스템 시작!")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("사용법:")
        print("  🖱️  마우스 클릭: 차량 선택 및 추적 시작")
        print("  🔄  자동 핸드오버: 선택된 차량이 화면 경계 근처에 도달시")
        print("  ⌨️  키보드 'q': 종료")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        frame_count = 0
        fps_start = time.time()
        
        try:
            while True:
                # 프레임 읽기
                current_frame, secondary_frame = self.get_frames()
                
                if current_frame is None:
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
                    print(f"📊 FPS: {fps:.1f} | 프레임: {frame_count} | 모드: {self.ui_system.current_mode.value}")
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
    system.initialize_modules()
    system.setup_matplotlib()
    
    # 카메라 시작
    if system.start_camera(cctv_name, stream_url):
        system.run()
    else:
        print("❌ 시스템 시작 실패")


if __name__ == "__main__":
    main()