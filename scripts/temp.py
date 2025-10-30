"""
핸드오버 시스템 통합 메인 스크립트 (정리된 버전)
파일명: clean_handover_system.py
"""
import cv2
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dotenv import load_dotenv
from collections import deque

# 기존 모듈들 import (실제 사용시 주석 해제)
# from core.data_manager import DataManager
# from handover.frame_concatenator import FrameConcatenator, BBoxSeparator
# from handover.coordinate_transformer import CoordinateTransformer
# from handover.handover_manager import HandoverManager, HandoverState
# from ui.simple_handover_ui import SimpleHandoverUI, UIMode
# from detector.yolo_detector import get_vehicle_detections
# from tracker.tracker_test import MultiTracker, check_boundary_event
# from reid.feature_extractor import ReIDSystem

# 임시 더미 클래스들 (테스트용)
class DataManager:
    def __init__(self): pass
    def get_camera_connections(self): return [{"cctvname": "[남해선] 죽평"}]
    def set_ui_mode(self, *args): pass
    def set_selected_vehicle(self, *args): pass
    def add_vehicle(self, *args): pass

class FrameConcatenator:
    def __init__(self): pass

class BBoxSeparator:
    def __init__(self): pass

class CoordinateTransformer:
    def __init__(self): pass

class HandoverManager:
    def __init__(self, dm): 
        self.current_state = "IDLE"
    def initialize_modules(self): pass
    def update_frame(self, *args): pass
    def check_handover_trigger(self, *args): return False
    def update_handover_state(self): return None

class SimpleHandoverUI:
    def __init__(self):
        self.current_mode = "SINGLE"
        self.secondary_camera = None
    def set_single_mode(self, *args): pass
    def set_dual_mode(self, *args): pass
    def create_display_frame(self, frames): 
        if frames:
            return list(frames.values())[0], {}
        return np.zeros((480, 640, 3), dtype=np.uint8), {}
    def handle_click(self, x, y): return {"success": True}

class MultiTracker:
    def __init__(self): pass
    def update(self, detections): 
        tracks = []
        for i, det in enumerate(detections or []):
            if len(det) >= 4:
                tracks.append((i+1, det[0], det[1], det[2], det[3]))
        return tracks
    def select_track_by_point(self, x, y): return {"id": 1, "bbox": [x-50, y-50, x+50, y+50]}

class ReIDSystem:
    def __init__(self, **kwargs): pass
    def register_lost_vehicle(self, *args): pass

def get_vehicle_detections(frame, **kwargs):
    # 더미 detection (테스트용)
    h, w = frame.shape[:2]
    return [[w//4, h//4, w//2, h//2, 0.8]]

class HLSJumpDetector:
    """HLS 세그먼트 점프 감지기 (개선된 버전)"""
    
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.jump_history = []
        self.last_frame_time = None
        self.frame_counter = 0
        
        # 개선된 설정값
        self.jump_threshold_multiplier = 5.0  # 3.0 → 5.0으로 증가 (덜 민감하게)
        self.min_jump_size = 2.0  # 최소 2초 이상만 점프로 간주
        self.min_avg_interval = 0.01  # 최소 평균 간격 (너무 작으면 무시)
        self.stable_frames_needed = 20  # 안정화를 위해 20프레임 필요
        
    def detect_jump(self, frame):
        """프레임에서 점프 감지 (개선된 로직)"""
        current_time = time.time()
        self.frame_counter += 1
        
        if self.last_frame_time is not None:
            interval = current_time - self.last_frame_time
            self.frame_times.append(interval)
            
            # 충분한 데이터가 쌓인 후에만 점프 감지
            if len(self.frame_times) > self.stable_frames_needed:
                avg_interval = sum(self.frame_times) / len(self.frame_times)
                
                # 점프 조건을 더 엄격하게
                is_significant_jump = (
                    interval > avg_interval * self.jump_threshold_multiplier and  # 5배 이상
                    interval > self.min_jump_size and  # 절대적으로 2초 이상
                    avg_interval > self.min_avg_interval  # 평균이 너무 작지 않음
                )
                
                if is_significant_jump:
                    # 연속된 점프 방지 (마지막 점프로부터 5초 이상 경과)
                    if (not self.jump_history or 
                        current_time - self.jump_history[-1]['time'] > 5.0):
                        
                        jump_info = {
                            'frame_count': self.frame_counter,
                            'time': current_time,
                            'interval': interval,
                            'avg_interval': avg_interval,
                            'jump_size': interval - avg_interval
                        }
                        
                        self.jump_history.append(jump_info)
                        
                        print(f"🔥 진짜 HLS 점프 감지!")
                        print(f"  프레임: {self.frame_counter}")
                        print(f"  간격: {interval:.3f}초 (평균: {avg_interval:.3f}초)")
                        print(f"  점프 크기: {jump_info['jump_size']:.3f}초")
                        
                        return True
        
        self.last_frame_time = current_time
        return False
    
    def get_jump_pattern(self):
        """점프 패턴 분석 (개선된 버전)"""
        if len(self.jump_history) < 2:
            return None
        
        intervals = []
        for i in range(1, len(self.jump_history)):
            time_diff = self.jump_history[i]['time'] - self.jump_history[i-1]['time']
            intervals.append(time_diff)
        
        if intervals:
            avg_jump_interval = sum(intervals) / len(intervals)
            print(f"📊 진짜 점프 패턴 분석:")
            print(f"  총 점프 수: {len(self.jump_history)}")
            print(f"  평균 점프 간격: {avg_jump_interval:.1f}초")
            print(f"  점프 간격들: {[f'{i:.1f}s' for i in intervals[-5:]]}")
            
            return {
                'total_jumps': len(self.jump_history),
                'avg_interval': avg_jump_interval,
                'recent_intervals': intervals[-5:]
            }
        
        return None

class CleanHandoverSystem:
    """정리된 핸드오버 시스템"""
    
    def __init__(self):
        print("🚀 핸드오버 시스템 초기화 중...")
        
        # 모듈들 초기화
        self.data_manager = DataManager()
        self.frame_concatenator = FrameConcatenator()
        self.bbox_separator = BBoxSeparator()
        self.coord_transformer = CoordinateTransformer()
        self.handover_manager = HandoverManager(self.data_manager)
        self.ui_system = SimpleHandoverUI()
        self.tracker = MultiTracker()
        self.reid_system = ReIDSystem(similarity_threshold=0.7)
        
        # 카메라 설정
        self.current_cap = None
        self.secondary_cap = None
        self.current_cctv = None
        
        # matplotlib 설정
        plt.ion()
        self.fig = None
        self.ax = None
        
        # 상태 관리
        self.detection_interval = 2
        self.frame_counter = 0
        self.last_detections = []
        self.selected_vehicle_id = None
        self.reid_registered = set()
        
        # 디버깅 및 점프 감지
        self.debug_mode = False
        self.jump_detector = None
        self.jump_monitoring = False
        
        print("✅ 시스템 초기화 완료")
    
    def enable_debug_mode(self):
        """디버깅 모드 활성화"""
        print("🐛 디버깅 모드 활성화")
        self.debug_mode = True
    
    def enable_jump_monitoring(self):
        """점프 모니터링 활성화"""
        print("🔍 HLS 점프 모니터링 활성화")
        self.jump_detector = HLSJumpDetector()
        self.jump_monitoring = True
    
    def setup_matplotlib(self):
        """matplotlib 설정"""
        if self.fig:
            plt.close(self.fig)
        
        self.fig = plt.figure(figsize=(14, 8))
        self.ax = self.fig.add_subplot(111)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        print("📊 matplotlib 설정 완료")
    
    def start_camera(self, stream_url: str) -> bool:
        """카메라 시작"""
        print(f"📡 카메라 연결 시도...")
        
        self.current_cap = cv2.VideoCapture(stream_url)
        if not self.current_cap.isOpened():
            print("❌ 카메라 연결 실패")
            return False
        
        # HLS 최적화 설정
        self.current_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.current_cap.set(cv2.CAP_PROP_FPS, 25)
        
        # 카메라 정보 설정
        self.current_cctv = {"cctvname": "[테스트] 카메라"}
        
        print("✅ 카메라 연결 성공")
        return True
    
    def get_frame(self):
        """프레임 읽기"""
        if not self.current_cap:
            return None
        
        for attempt in range(3):
            ret, frame = self.current_cap.read()
            if ret and frame is not None:
                return frame
            elif attempt < 2:
                time.sleep(0.001)
        
        return None
    
    def monitor_stream_continuity(self, frame):
        """스트림 연속성 모니터링"""
        if not self.jump_monitoring or not self.jump_detector:
            return
        
        jump_detected = self.jump_detector.detect_jump(frame)
        
        if jump_detected:
            if self.debug_mode:
                print("🔧 스트림 점프 처리 중...")
            
            # 캐시된 detection 초기화
            self.last_detections = []
            
            # 패턴 분석 (5번째 점프마다)
            if len(self.jump_detector.jump_history) % 5 == 0:
                pattern = self.jump_detector.get_jump_pattern()
                if pattern and pattern['avg_interval'] > 0:
                    print(f"💡 예상 다음 점프: {pattern['avg_interval']:.1f}초 후")
    
    def process_detections(self, frame):
        """탐지 처리"""
        self.frame_counter += 1
        
        if self.frame_counter % self.detection_interval == 0:
            original_h, original_w = frame.shape[:2]
            target_w, target_h = 800, 600
            
            if self.debug_mode:
                print(f"🔍 탐지 실행: 프레임 {self.frame_counter}")
            
            # 리사이즈 및 탐지
            small_frame = cv2.resize(frame, (target_w, target_h))
            detections = get_vehicle_detections(small_frame, conf_threshold=0.2)
            
            if detections:
                # 좌표 복원
                scale_x = original_w / target_w
                scale_y = original_h / target_h
                
                self.last_detections = []
                for i, det in enumerate(detections):
                    if len(det) >= 4:
                        x1, y1, x2, y2 = det[:4]
                        conf = det[4] if len(det) > 4 else 0.5
                        
                        # 좌표 복원
                        scaled_x1 = int(x1 * scale_x)
                        scaled_y1 = int(y1 * scale_y)
                        scaled_x2 = int(x2 * scale_x)
                        scaled_y2 = int(y2 * scale_y)
                        
                        # 경계 클리핑
                        scaled_x1 = max(0, min(scaled_x1, original_w-1))
                        scaled_y1 = max(0, min(scaled_y1, original_h-1))
                        scaled_x2 = max(scaled_x1+1, min(scaled_x2, original_w))
                        scaled_y2 = max(scaled_y1+1, min(scaled_y2, original_h))
                        
                        # 크기 필터링
                        if (scaled_x2 - scaled_x1) > 20 and (scaled_y2 - scaled_y1) > 20:
                            self.last_detections.append((
                                scaled_x1, scaled_y1, scaled_x2, scaled_y2, conf
                            ))
            else:
                self.last_detections = []
        
        return self.last_detections
    
    def draw_frame(self, frame, tracks):
        """프레임 그리기"""
        self.ax.clear()
        
        # 프레임 표시
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax.imshow(frame_rgb)
        
        # 트랙 그리기
        if tracks:
            for track_data in tracks:
                track_id, x1, y1, x2, y2 = track_data
                width = x2 - x1
                height = y2 - y1
                
                # 색상 선택
                if track_id == self.selected_vehicle_id:
                    color = 'magenta'
                    linewidth = 3
                    label = f"ID{track_id} [SELECTED]"
                else:
                    color = 'red'
                    linewidth = 2
                    label = f"ID{track_id}"
                
                # 사각형 그리기
                rect = Rectangle((x1, y1), width, height,
                               linewidth=linewidth, edgecolor=color, facecolor='none')
                self.ax.add_patch(rect)
                
                # 라벨
                self.ax.text(x1, max(10, y1-5), label, color=color, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.5))
        
        # 상태 정보
        status_parts = [f"Frame: {self.frame_counter}"]
        if self.selected_vehicle_id:
            status_parts.append(f"Selected: ID{self.selected_vehicle_id}")
        if self.jump_monitoring and self.jump_detector:
            jump_count = len(self.jump_detector.jump_history)
            status_parts.append(f"Jumps: {jump_count}")
        
        self.fig.suptitle(" | ".join(status_parts), fontsize=12)
        self.ax.axis('off')
        
        plt.pause(0.01)
    
    def on_click(self, event):
        """마우스 클릭 이벤트"""
        if not event.inaxes == self.ax:
            return
        
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return
        
        click_x, click_y = int(click_x), int(click_y)
        
        if self.debug_mode:
            print(f"🖱️ 클릭 좌표: ({click_x}, {click_y})")
        
        # 트랙 선택
        try:
            selected = self.tracker.select_track_by_point(click_x, click_y)
            if selected:
                track_id = selected['id']
                self.selected_vehicle_id = track_id
                print(f"🎯 차량 선택: ID{track_id}")
        except Exception as e:
            if self.debug_mode:
                print(f"❌ 트랙 선택 오류: {e}")
    
    def run(self):
        """메인 실행 루프"""
        print("\n🚀 핸드오버 시스템 시작!")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("사용법:")
        print("  🖱️  마우스 클릭: 차량 선택")
        print("  ⌨️  키보드 Ctrl+C: 종료")
        if self.jump_monitoring:
            print("  🔍  점프 모니터링 활성화")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        frame_count = 0
        fps_start = time.time()
        
        try:
            while True:
                # 프레임 읽기
                frame = self.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                
                # 점프 모니터링
                if self.jump_monitoring:
                    self.monitor_stream_continuity(frame)
                
                # 탐지 처리
                detections = self.process_detections(frame)
                
                # 트래커 업데이트
                tracks = self.tracker.update(detections)
                
                # 화면 그리기
                self.draw_frame(frame, tracks)
                
                # FPS 계산
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start
                    fps = 30 / elapsed if elapsed > 0 else 0
                    
                    status = f"📊 FPS: {fps:.1f} | 프레임: {frame_count}"
                    if self.debug_mode:
                        status += f" | 탐지: {len(detections)} | 트랙: {len(tracks)}"
                    if self.jump_monitoring and self.jump_detector:
                        status += f" | 점프: {len(self.jump_detector.jump_history)}회"
                    
                    print(status)
                    fps_start = time.time()
        
        except KeyboardInterrupt:
            print("\n⌨️ 사용자 중단")
            
            # 최종 점프 분석
            if self.jump_monitoring and self.jump_detector:
                pattern = self.jump_detector.get_jump_pattern()
                if pattern:
                    print(f"\n📈 최종 점프 분석:")
                    print(f"  총 {pattern['total_jumps']}회 점프")
                    print(f"  평균 {pattern['avg_interval']:.1f}초 간격")
        
        except Exception as e:
            print(f"\n💥 오류: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """시스템 종료"""
        print("\n🛑 시스템 종료 중...")
        
        if self.current_cap:
            self.current_cap.release()
        
        if self.fig:
            plt.close(self.fig)
        
        plt.ioff()
        print("✅ 시스템 종료 완료")

def main():
    """기본 실행"""
    load_dotenv()
    
    stream_url = os.getenv("CURRENT_CCTV_URL", "")
    if not stream_url:
        print("❌ CURRENT_CCTV_URL 환경변수가 설정되지 않았습니다.")
        return
    
    print("🎬 정리된 핸드오버 시스템")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    system = CleanHandoverSystem()
    system.enable_debug_mode()
    system.setup_matplotlib()
    
    if system.start_camera(stream_url):
        system.run()
    else:
        print("❌ 시스템 시작 실패")

def main_with_jump_detection():
    """점프 감지 포함 실행"""
    load_dotenv()
    
    stream_url = os.getenv("CURRENT_CCTV_URL", "")
    if not stream_url:
        print("❌ CURRENT_CCTV_URL 환경변수가 설정되지 않았습니다.")
        return
    
    print("🎬 점프 모니터링 핸드오버 시스템")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    system = CleanHandoverSystem()
    system.enable_debug_mode()
    system.enable_jump_monitoring()  # 점프 감지 활성화
    system.setup_matplotlib()
    
    if system.start_camera(stream_url):
        system.run()
    else:
        print("❌ 시스템 시작 실패")

if __name__ == "__main__":
    # 점프 감지 포함 실행 (권장)
    main_with_jump_detection()
    
    # 기본 실행
    # main()