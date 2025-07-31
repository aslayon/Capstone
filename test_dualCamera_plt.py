"""
matplotlib 기반 부드러운 듀얼카메라
파일명: test_matplotlib_dual.py

test.py의 matplotlib 방식을 사용한 듀얼카메라
"""
import cv2
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from detector.yolo_detector import get_vehicle_detections
from tracker.tracker_test import MultiTracker, check_boundary_event
from handover.handover_logic import load_cctv_list
from reid.feature_extractor import ReIDSystem
from dotenv import load_dotenv

class MatplotlibDualCameraSystem:
    """matplotlib 기반 부드러운 듀얼카메라"""
    
    def __init__(self):
        # 한글 폰트 경고 해결
        import matplotlib
        matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        import warnings
        warnings.filterwarnings("ignore", message="Glyph .* missing from font")
        
        # 기본 시스템 (test.py와 동일)
        self.tracker = MultiTracker()
        self.reid_system = ReIDSystem(similarity_threshold=0.7)
        
        # 카메라 설정
        self.current_cap = None
        self.next_cap = None
        self.dual_mode = False
        
        # CCTV 정보
        self.cctv_list = load_cctv_list()
        self.connections = self._load_connections()
        self.current_cctv = None
        self.next_cctv = None
        
        # matplotlib 설정 (test.py와 동일)
        plt.ion()  # interactive mode
        self.fig = None
        self.ax_current = None
        self.ax_next = None
        
        # 성능 설정
        self.detection_interval = 3  # 3프레임마다
        self.frame_counter = 0
        self.last_detections = []
        
        # 선택된 객체 관리
        self.selected_track_id = None
        self.reid_registered = set()
        self.lost_tracks = {}
        
        print("📊 matplotlib 기반 듀얼카메라 시스템 초기화")
    
    def _load_connections(self):
        """연결 관계 로드"""
        try:
            with open("cctv_graph_connections.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    
    def find_cctv_by_name(self, name):
        """CCTV 찾기"""
        for cctv in self.cctv_list:
            if name in cctv["cctvname"]:
                return cctv
        return None
    
    def find_next_camera(self, direction):
        """다음 카메라 찾기"""
        if not self.current_cctv:
            return None
        
        current_name = self.current_cctv["cctvname"]
        
        for connection in self.connections:
            if current_name == connection["cctvname"]:
                for conn in connection["connections"]:
                    if conn["direction"] == direction:
                        return self.find_cctv_by_name(conn["target"])
        return None
    
    def start_with_camera(self, cctv_name, stream_url):
        """카메라 시작"""
        self.current_cctv = self.find_cctv_by_name(cctv_name)
        if not self.current_cctv:
            return False
        
        print(f"📡 카메라 연결: {cctv_name}")
        self.current_cap = cv2.VideoCapture(stream_url)
        
        if not self.current_cap.isOpened():
            return False
        
        self.current_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # matplotlib 초기화 (싱글 모드)
        self.setup_matplotlib_single()
        
        print(f"✅ 카메라 연결 성공")
        return True
    
    def setup_matplotlib_single(self):
        """matplotlib 싱글 모드 설정"""
        if self.fig:
            plt.close(self.fig)
        
        self.fig = plt.figure(figsize=(12, 8))
        self.ax_current = self.fig.add_subplot(111)
        self.ax_next = None
        
        # 마우스 클릭 이벤트 연결 (test.py와 동일)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.dual_mode = False
        print("📊 matplotlib 싱글 모드 설정")
    
    def setup_matplotlib_dual(self):
        """matplotlib 듀얼 모드 설정"""
        if self.fig:
            plt.close(self.fig)
        
        self.fig = plt.figure(figsize=(16, 8))
        
        # 좌우 분할
        self.ax_current = self.fig.add_subplot(121)
        self.ax_next = self.fig.add_subplot(122)
        
        self.ax_current.set_title("CURRENT CAMERA")
        self.ax_next.set_title("NEXT CAMERA")
        
        # 마우스 클릭 이벤트
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.dual_mode = True
        print("📊 matplotlib 듀얼 모드 설정")
    
    def activate_dual_mode(self, direction):
        """듀얼 모드 활성화"""
        if self.dual_mode:
            return False
        
        next_cctv = self.find_next_camera(direction)
        if not next_cctv:
            print(f"❌ {direction} 방향 카메라 없음")
            return False
        
        print(f"🔄 듀얼 모드 활성화: {direction}")
        
        # 다음 카메라 연결
        stream_url = os.getenv("CURRENT_CCTV_URL", "")
        self.next_cap = cv2.VideoCapture(stream_url)
        
        if not self.next_cap.isOpened():
            print("❌ 다음 카메라 연결 실패")
            return False
        
        self.next_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.next_cctv = next_cctv
        
        # matplotlib 듀얼 모드로 전환
        self.setup_matplotlib_dual()
        
        print(f"✅ 듀얼 모드 시작")
        return True
    
    def deactivate_dual_mode(self):
        """듀얼 모드 비활성화"""
        if not self.dual_mode:
            return
        
        print("🛑 듀얼 모드 종료")
        
        if self.next_cap:
            self.next_cap.release()
            self.next_cap = None
        
        self.next_cctv = None
        
        # matplotlib 싱글 모드로 전환
        self.setup_matplotlib_single()
    
    def get_frames(self):
        """프레임 읽기"""
        current_frame = None
        next_frame = None
        
        if self.current_cap:
            ret, current_frame = self.current_cap.read()
            if not ret:
                current_frame = None
        
        if self.dual_mode and self.next_cap:
            ret, next_frame = self.next_cap.read()
            if not ret:
                next_frame = None
        
        return current_frame, next_frame
    
    def process_detections(self, frame):
        """탐지 처리"""
        self.frame_counter += 1
        
        # 3프레임마다만 YOLO
        if self.frame_counter % self.detection_interval == 0:
            small_frame = cv2.resize(frame, (640, 480))
            detections = get_vehicle_detections(small_frame, conf_threshold=0.3)
            
            # 좌표 복원
            if detections:
                scale_x = frame.shape[1] / 640
                scale_y = frame.shape[0] / 480
                
                self.last_detections = []
                for det in detections:
                    if len(det) >= 4:
                        x1, y1, x2, y2 = det[:4]
                        self.last_detections.append((
                            int(x1 * scale_x), int(y1 * scale_y),
                            int(x2 * scale_x), int(y2 * scale_y)
                        ) + det[4:])
            else:
                self.last_detections = []
        
        return self.last_detections
    
    def on_track_selected(self, track_id, frame, bbox):
        """트랙 선택 - ReID 등록"""
        if track_id in self.reid_registered:
            return
        
        print(f"🎯 트랙 선택: ID{track_id}")
        
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        vehicle_crop = frame[y1:y2, x1:x2]
        
        if vehicle_crop.size > 0:
            self.reid_system.register_lost_vehicle(
                track_id, vehicle_crop, bbox, 'car',
                {'cctv': self.current_cctv['cctvname']}
            )
            
            self.reid_registered.add(track_id)
            self.selected_track_id = track_id
            print(f"✅ ID{track_id} ReID 등록 완료")
    
    def check_handover(self, frame):
        """핸드오버 체크"""
        bbox = self.tracker.get_selected_bbox()
        if not bbox:
            return
        
        h, w = frame.shape[:2]
        
        if check_boundary_event(bbox, w, h) and not self.dual_mode:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            
            if center_x < w // 3:
                direction = 'south'
            elif center_x > 2 * w // 3:
                direction = 'north'
            else:
                return
            
            self.activate_dual_mode(direction)
    
    def process_reid_if_needed(self, next_frame):
        """필요시 ReID 처리"""
        if not self.dual_mode or not next_frame is not None:
            return []
        
        if not self.lost_tracks:
            return []
        
        # 다음 카메라에서 탐지
        small_next = cv2.resize(next_frame, (640, 480))
        next_detections = get_vehicle_detections(small_next, conf_threshold=0.4)
        
        if not next_detections:
            return []
        
        # ReID 검색
        matches = self.reid_system.search_in_new_camera(
            next_detections, next_frame,
            self.next_cctv['cctvname'] if self.next_cctv else "Unknown"
        )
        
        # 높은 유사도 매칭만 처리
        good_matches = [m for m in matches if m['similarity'] > 0.8]
        
        for match in good_matches:
            if match['lost_id'] in self.lost_tracks:
                del self.lost_tracks[match['lost_id']]
                print(f"🎯 ReID 매칭: ID{match['lost_id']}, 유사도 {match['similarity']:.3f}")
        
        return good_matches
    
    def draw_matplotlib_frame(self, current_frame, next_frame=None, tracks=None, reid_matches=None):
        """matplotlib으로 프레임 그리기 (test.py 스타일)"""
        
        if self.dual_mode and next_frame is not None:
            # 듀얼 모드: 좌우 분할
            
            # 현재 카메라
            self.ax_current.clear()
            current_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            self.ax_current.imshow(current_rgb)
            self.ax_current.set_title(f"CURRENT: {self.current_cctv['cctvname'][:20]}")
            self.ax_current.axis('off')
            
            # 트랙 그리기 (현재 카메라)
            if tracks:
                for track_id, x1, y1, x2, y2 in tracks:
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 색상 선택
                    if track_id == self.selected_track_id:
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
                    self.ax_current.add_patch(rect)
                    
                    # 레이블
                    label = f"ID{track_id}"
                    if track_id == self.selected_track_id:
                        label += " [SEL]"
                    elif track_id in self.reid_registered:
                        label += " [ReID]"
                    
                    self.ax_current.text(x1, y1-5, label, color=color, fontsize=8, 
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.5))
            
            # 다음 카메라
            self.ax_next.clear()
            next_rgb = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
            self.ax_next.imshow(next_rgb)
            self.ax_next.set_title(f"NEXT: {self.next_cctv['cctvname'][:20]}")
            self.ax_next.axis('off')
            
            # ReID 매칭 표시
            if reid_matches:
                for i, match in enumerate(reid_matches):
                    self.ax_next.text(10, 30 + i*20, 
                                    f"Match: ID{match['lost_id']} ({match['similarity']:.2f})",
                                    color='cyan', fontsize=10,
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
        else:
            # 싱글 모드 (test.py와 동일)
            self.ax_current.clear()
            frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            self.ax_current.imshow(frame_rgb)
            self.ax_current.set_title(f"Vehicle Tracking - Camera")  # 한글 제거
            self.ax_current.axis('off')
            
            # 트랙 그리기 (test.py와 유사)
            if tracks:
                for track_id, x1, y1, x2, y2 in tracks:
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 색상 선택
                    if track_id == self.selected_track_id:
                        color = 'magenta'
                        linewidth = 3
                    elif track_id in self.reid_registered:
                        color = 'yellow'
                        linewidth = 2
                    else:
                        color = 'red'
                        linewidth = 2
                    
                    # 사각형 그리기 (test.py 스타일)
                    rect = Rectangle((x1, y1), width, height, 
                                   linewidth=linewidth, edgecolor=color, facecolor='none')
                    self.ax_current.add_patch(rect)
                    
                    # ID 레이블
                    label = f"ID {track_id}"
                    if track_id == self.selected_track_id:
                        label += " [SELECTED]"
                    elif track_id in self.reid_registered:
                        label += " [ReID]"
                    
                    self.ax_current.text(x1, y1-5, label, color=color, fontsize=10, 
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.5))
        
        # 상태 정보 표시
        status_text = f"Frame: {self.frame_counter}"
        if self.dual_mode:
            status_text += " | DUAL MODE"
        if self.selected_track_id:
            status_text += f" | Selected: ID{self.selected_track_id}"
        if self.lost_tracks:
            status_text += f" | Lost: {len(self.lost_tracks)}"
        
        self.fig.suptitle(status_text, fontsize=12)
        
        # test.py와 동일한 방식으로 업데이트
        plt.pause(0.01)  # 핵심! test.py와 동일
    
    def on_click(self, event):
        """마우스 클릭 이벤트 (test.py와 동일)"""
        if event.inaxes == self.ax_current:  # 현재 카메라만 클릭 가능
            x, y = int(event.xdata), int(event.ydata)
            
            # 트랙 선택 (test.py와 동일)
            selected = self.tracker.select_track_by_point(x, y)
            
            if selected:
                track_id = selected['id']
                bbox = selected['bbox']
                
                # 현재 프레임에서 ReID 등록
                current_frame, _ = self.get_frames()
                if current_frame is not None:
                    self.on_track_selected(track_id, current_frame, bbox)
    
    def update_tracker_with_events(self, detections):
        """트래커 업데이트 및 이벤트 감지"""
        # 기존 트랙 ID들 저장 (단순화)
        old_track_ids = set()
        
        try:
            # tracker_test.py의 MultiTracker 사용시
            if hasattr(self.tracker, 'tracks'):
                old_track_ids = {getattr(track, 'id', track) for track in self.tracker.tracks}
        except:
            # 안전한 대안
            old_track_ids = set()
        
        # 트래커 업데이트 (test.py와 동일)
        tracks = self.tracker.update(detections)
        
        # 현재 트랙 ID들
        current_track_ids = {track_id for track_id, *_ in tracks}
        
        # 분실된 트랙 감지 (단순화)
        lost_ids = old_track_ids - current_track_ids
        
        for lost_id in lost_ids:
            if lost_id in self.reid_registered:
                print(f"📉 등록된 트랙 분실: ID{lost_id}")
                self.lost_tracks[lost_id] = time.time()
                
                # 듀얼 모드 자동 활성화
                if not self.dual_mode:
                    print("🔄 분실로 인한 듀얼 모드 활성화")
                    self.activate_dual_mode('north')
        
        return tracks
    
    def run(self):
        """메인 실행 루프 (test.py 스타일)"""
        print("\n📊 matplotlib 기반 듀얼카메라 시작!")
        print("사용법:")
        print("  - 마우스 클릭: 차량 선택 (ReID 등록)")
        print("  - 키보드 'd': 듀얼 모드 토글")
        print("  - 키보드 'q': 종료")
        print("  - 키보드 'h': 상태 정보")
        
        frame_count = 0
        fps_start = time.time()
        
        try:
            while True:
                # 프레임 읽기
                current_frame, next_frame = self.get_frames()
                
                if current_frame is None:
                    print("❌ 프레임 읽기 실패")
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                
                # 탐지 처리 (간헐적)
                detections = self.process_detections(current_frame)
                
                # 트래커 업데이트 (이벤트 포함)
                tracks = self.update_tracker_with_events(detections)
                
                # 핸드오버 체크
                self.check_handover(current_frame)
                
                # ReID 처리 (필요시만)
                reid_matches = []
                if self.dual_mode and next_frame is not None and self.lost_tracks:
                    reid_matches = self.process_reid_if_needed(next_frame)
                
                # matplotlib으로 그리기 (test.py와 동일한 방식)
                self.draw_matplotlib_frame(current_frame, next_frame, tracks, reid_matches)
                
                # FPS 계산 (간헐적)
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start
                    fps = 30 / elapsed if elapsed > 0 else 0
                    print(f"📊 FPS: {fps:.1f} | 프레임: {frame_count}")
                    fps_start = time.time()
                
                # 키보드 입력 처리 (matplotlib 이벤트)
                if plt.waitforbuttonpress(timeout=0.001):  # 1ms 타임아웃
                    # 간단한 키 입력 처리는 제한적이므로 별도 처리 필요
                    pass
        
        except KeyboardInterrupt:
            print("\n⌨️ 사용자 중단")
        except Exception as e:
            print(f"\n💥 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.shutdown()
    
    def run_with_keyboard_input(self):
        """키보드 입력이 포함된 실행 루프"""
        print("\n📊 matplotlib 듀얼카메라 (키보드 지원)")
        print("matplotlib 창을 활성화한 후:")
        print("  - 마우스 클릭: 차량 선택")
        print("  - 터미널에서 명령어:")
        print("    'd': 듀얼 모드 토글")
        print("    'h': 상태 정보")
        print("    'q': 종료")
        
        import threading
        import sys
        
        # 키보드 입력 스레드
        def keyboard_input():
            while True:
                try:
                    cmd = input().strip().lower()
                    if cmd == 'q':
                        print("🛑 종료 명령")
                        self.shutdown()
                        break
                    elif cmd == 'd':
                        if self.dual_mode:
                            self.deactivate_dual_mode()
                        else:
                            self.activate_dual_mode('north')
                    elif cmd == 'h':
                        print(f"\n📊 현재 상태:")
                        print(f"  프레임: {self.frame_counter}")
                        print(f"  듀얼 모드: {self.dual_mode}")
                        print(f"  선택된 트랙: {self.selected_track_id}")
                        print(f"  ReID 등록: {len(self.reid_registered)}")
                        print(f"  분실 트랙: {len(self.lost_tracks)}")
                except:
                    break
        
        # 키보드 스레드 시작
        keyboard_thread = threading.Thread(target=keyboard_input, daemon=True)
        keyboard_thread.start()
        
        # 메인 루프 실행
        self.run()
    
    def shutdown(self):
        """정리"""
        print("\n🛑 시스템 종료 중...")
        
        if self.current_cap:
            self.current_cap.release()
        
        if self.next_cap:
            self.next_cap.release()
        
        if self.fig:
            plt.close(self.fig)
        
        plt.ioff()  # interactive mode off
        
        print("✅ 시스템 종료 완료")


def main():
    """메인 함수"""
    load_dotenv()
    
    stream_url = os.getenv("CURRENT_CCTV_URL", "")
    cctv_name = os.getenv("CURRENT_CCTV_NAME", "죽평")
    
    if not stream_url:
        print("❌ CURRENT_CCTV_URL 환경변수가 설정되지 않았습니다.")
        return
    
    print(f"📹 카메라: {cctv_name}")
    print(f"🔗 스트림: {stream_url[:50]}...")
    
    system = MatplotlibDualCameraSystem()
    
    if system.start_with_camera(cctv_name, stream_url):
        # 키보드 입력 지원 버전으로 실행
        system.run_with_keyboard_input()
    else:
        print("❌ 시스템 시작 실패")


if __name__ == "__main__":
    main()