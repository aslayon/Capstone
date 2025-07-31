"""
성능 최적화된 듀얼 카메라 시스템
파일명: test_optimized_camera.py

기존 파일들을 수정하지 않고 새로운 최적화 버전
"""
import cv2
import os
import time
import json
import numpy as np
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

# 기존 모듈들 import (수정 없음)
from detector.yolo_detector_optimized import get_vehicle_detections
from tracker.tracker_v2 import PersistentMultiTracker
from handover.handover_logic import load_cctv_list
from handover.handover_detector import HandoverDetector
from reid.feature_extractor import ReIDSystem
from utils.stream import DualCameraManager  # 기존 그대로 사용
from dotenv import load_dotenv

class PerformanceConfig:
    """성능 최적화 설정 - 인라인 포함"""
    
    def __init__(self, performance_level: str = None):
        # 환경변수에서 성능 레벨 읽기
        if performance_level is None:
            performance_level = os.getenv("PERFORMANCE_LEVEL", "fast")
        
        self.performance_level = performance_level
        self._setup_config()
    
    def _setup_config(self):
        """성능 레벨별 설정"""
        configs = {
            "ultra_fast": {
                "target_fps": 8,
                "frame_skip": 5,
                "detection_interval": 8,
                "reid_interval": 30,
                "yolo_conf_threshold": 0.6,
                "window_width": 640,
                "window_height": 360,
                "max_tracks": 15,
                "iou_threshold": 0.6,
                "vehicle_classes": ['car']
            },
            "fast": {
                "target_fps": 12,
                "frame_skip": 3,
                "detection_interval": 5,
                "reid_interval": 20,
                "yolo_conf_threshold": 0.5,
                "window_width": 800,
                "window_height": 450,
                "max_tracks": 25,
                "iou_threshold": 0.5,
                "vehicle_classes": ['car']
            },
            "balanced": {
                "target_fps": 18,
                "frame_skip": 2,
                "detection_interval": 3,
                "reid_interval": 15,
                "yolo_conf_threshold": 0.4,
                "window_width": 1024,
                "window_height": 576,
                "max_tracks": 40,
                "iou_threshold": 0.4,
                "vehicle_classes": ['car', 'truck']
            }
        }
        
        self.config = configs.get(self.performance_level, configs["fast"])
        print(f"🎯 성능 모드: {self.performance_level.upper()}")
        print(f"   목표 FPS: {self.config['target_fps']}")
        print(f"   프레임 스킵: {self.config['frame_skip']}")
        print(f"   화면 크기: {self.config['window_width']}x{self.config['window_height']}")
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)


class OptimizedDualCameraSystem:
    """성능 최적화된 듀얼 카메라 시스템"""
    
    def __init__(self):
        # 성능 설정 로드
        self.perf_config = PerformanceConfig()
        
        # 성능 설정 적용
        self.target_fps = self.perf_config.get('target_fps')
        self.frame_skip = self.perf_config.get('frame_skip')
        self.detection_interval = self.perf_config.get('detection_interval')
        
        # 멀티스레딩 (간소화)
        self.detection_queue = Queue(maxsize=3)
        self.detection_thread = None
        self.detection_running = False
        
        # 핵심 시스템들 (기존과 동일)
        self.tracker = PersistentMultiTracker(
            max_age=80,  # 기본값보다 작게
            iou_threshold=self.perf_config.get('iou_threshold')
        )
        self.handover_detector = HandoverDetector()
        self.reid_system = ReIDSystem(similarity_threshold=0.7)
        self.camera_manager = DualCameraManager()  # 기존 클래스 사용
        
        # CCTV 정보 (기존과 동일)
        self.cctv_list = load_cctv_list()
        self.connections = self._load_connections()
        self.current_cctv = None
        self.next_cctv = None
        
        # 화면 설정 (최적화)
        self.window_width = self.perf_config.get('window_width')
        self.window_height = self.perf_config.get('window_height')
        self.dual_layout_active = False
        
        # 성능 측정
        self.fps_current = 0.0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.frame_counter = 0
        
        # 최적화된 캐시
        self.detection_cache = {}
        self.cache_max_size = 20  # 캐시 크기 제한
        
        # 통계
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'skipped_frames': 0,
            'detection_calls': 0,
            'cache_hits': 0
        }
        
        print("🚀 최적화된 듀얼 카메라 시스템 초기화 완료")
    
    def _load_connections(self):
        """연결 관계 로드 (기존과 동일)"""
        try:
            with open("cctv_graph_connections.json", 'r', encoding='utf-8') as f:
                connections = json.load(f)
            return connections
        except Exception as e:
            print(f"❌ 연결 관계 로드 실패: {e}")
            return []
    
    def start_detection_thread(self):
        """탐지 전용 스레드 시작"""
        self.detection_running = True
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.detection_thread.start()
        print("✅ 탐지 스레드 시작")
    
    def _detection_worker(self):
        """백그라운드 탐지 처리"""
        while self.detection_running:
            try:
                frame_data = self.detection_queue.get(timeout=0.1)
                if frame_data is None:  # 종료 신호
                    break
                
                frame, frame_id = frame_data
                
                # 빠른 탐지 수행
                detections = self._fast_detection(frame)
                
                # 캐시에 저장
                self.detection_cache[frame_id] = {
                    'detections': detections,
                    'timestamp': time.time()
                }
                
                # 캐시 크기 제한
                if len(self.detection_cache) > self.cache_max_size:
                    oldest_key = min(self.detection_cache.keys())
                    del self.detection_cache[oldest_key]
                
                self.stats['detection_calls'] += 1
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ 탐지 처리 오류: {e}")
    
    def _fast_detection(self, frame):
        """최적화된 빠른 탐지"""
        if frame is None:
            return []
        
        # 프레임 축소 (탐지 속도 향상)
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_height = int(height * scale)
            small_frame = cv2.resize(frame, (640, new_height), interpolation=cv2.INTER_NEAREST)
        else:
            small_frame = frame
        
        # 빠른 탐지 설정
        detections = get_vehicle_detections(
            small_frame, 
            conf_threshold=self.perf_config.get('yolo_conf_threshold'),
            vehicle_classes=self.perf_config.get('vehicle_classes')
        )
        
        # 좌표를 원본 크기로 복원
        if width > 640:
            scale_back = width / 640
            for i, det in enumerate(detections):
                if len(det) >= 4:
                    x1, y1, x2, y2 = det[:4]
                    detections[i] = (
                        int(x1 * scale_back), int(y1 * scale_back),
                        int(x2 * scale_back), int(y2 * scale_back)
                    ) + det[4:]
        
        return detections
    
    def get_detections(self, frame_id):
        """탐지 결과 가져오기 (캐시 활용)"""
        # 캐시에서 찾기
        if frame_id in self.detection_cache:
            self.stats['cache_hits'] += 1
            return self.detection_cache[frame_id]['detections']
        
        # 최근 캐시 중에서 찾기
        for i in range(1, min(6, len(self.detection_cache))):
            cache_key = frame_id - i
            if cache_key in self.detection_cache:
                self.stats['cache_hits'] += 1
                return self.detection_cache[cache_key]['detections']
        
        # 캐시 없으면 빈 리스트
        return []
    
    def find_cctv_by_name(self, name):
        """CCTV 이름으로 정보 찾기"""
        for cctv in self.cctv_list:
            if name in cctv["cctvname"]:
                return cctv
        return None
    
    def find_next_camera(self, direction):
        """방향에 따른 다음 카메라 찾기"""
        if not self.current_cctv:
            return None
        
        current_name = self.current_cctv["cctvname"]
        
        for connection in self.connections:
            if current_name == connection["cctvname"]:
                for conn in connection["connections"]:
                    if conn["direction"] == direction:
                        target_name = conn["target"]
                        return self.find_cctv_by_name(target_name)
        return None
    
    def start_with_camera(self, cctv_name):
        """카메라로 시스템 시작"""
        for cctv in self.cctv_list:
            if cctv_name in cctv["cctvname"] or cctv["cctvname"] in cctv_name:
                self.current_cctv = cctv
                break
        
        if not self.current_cctv:
            print(f"❌ CCTV를 찾을 수 없음: {cctv_name}")
            return False
        
        success = self.camera_manager.set_current_camera(
            self.current_cctv["cctvname"], 
            self.current_cctv["cctvurl"]
        )
        
        if success:
            self.start_detection_thread()
            print(f"✅ 시스템 시작: {self.current_cctv['cctvname']}")
            return True
        else:
            print(f"❌ 카메라 연결 실패: {cctv_name}")
            return False
    
    def activate_handover_mode(self, direction):
        """핸드오버 모드 활성화"""
        next_cctv = self.find_next_camera(direction)
        if not next_cctv:
            return False
        
        print(f"🔄 핸드오버 모드: {direction}")
        
        success = self.camera_manager.activate_dual_mode(
            next_cctv["cctvname"],
            next_cctv["cctvurl"]
        )
        
        if success:
            self.next_cctv = next_cctv
            self.dual_layout_active = True
            return True
        return False
    
    def create_simple_layout(self, current_frame, next_frame=None):
        """간단한 레이아웃 생성"""
        if self.dual_layout_active and next_frame is not None:
            # 듀얼 모드
            half_width = self.window_width // 2
            
            # 프레임 리사이즈
            current_resized = cv2.resize(current_frame, (half_width, self.window_height), 
                                       interpolation=cv2.INTER_NEAREST)
            next_resized = cv2.resize(next_frame, (half_width, self.window_height), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # 가로 결합
            combined = np.hstack([current_resized, next_resized])
            
            # 구분선
            cv2.line(combined, (half_width, 0), (half_width, self.window_height), (255, 255, 255), 2)
            
            return combined, half_width
        else:
            # 싱글 모드
            return cv2.resize(current_frame, (self.window_width, self.window_height), 
                            interpolation=cv2.INTER_NEAREST), 0
    
    def draw_simple_tracks(self, frame, tracks, offset_x=0):
        """간단한 트랙 그리기"""
        count = 0
        max_tracks = self.perf_config.get('max_tracks')
        
        for track_id, x1, y1, x2, y2, state, confidence in tracks:
            if count >= max_tracks:
                break
            
            # 좌표 변환
            scale_x = (self.window_width // 2 if offset_x > 0 else self.window_width) / frame.shape[1] if frame.shape[1] > 0 else 1
            scale_y = self.window_height / frame.shape[0] if frame.shape[0] > 0 else 1
            
            x1_scaled = int(x1 * scale_x) + offset_x
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x) + offset_x
            y2_scaled = int(y2 * scale_y)
            
            # 상태별 색상
            color = (0, 255, 0) if state == "DETECTED" else (0, 255, 255)
            
            # 박스 그리기
            cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 1)
            cv2.putText(frame, f"ID{track_id}", (x1_scaled, y1_scaled-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            count += 1
    
    def draw_simple_info(self, frame):
        """간단한 정보 표시"""
        # FPS만 표시
        cv2.putText(frame, f"FPS: {self.fps_current:.1f}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 모드 표시
        mode = f"{self.perf_config.performance_level.upper()}"
        if self.dual_layout_active:
            mode += " DUAL"
        cv2.putText(frame, mode, (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    def update_fps(self):
        """FPS 업데이트"""
        self.fps_frame_count += 1
        
        if self.fps_frame_count >= 15:
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            if elapsed > 0:
                self.fps_current = self.fps_frame_count / elapsed
            self.fps_start_time = current_time
            self.fps_frame_count = 0
    
    def handle_user_input(self, key):
        """사용자 입력 처리"""
        if key == ord('q'):
            return 'quit'
        elif key == ord('1'):  # Ultra Fast
            self.perf_config = PerformanceConfig("ultra_fast")
            self._apply_new_settings()
            print("⚡ Ultra Fast 모드")
        elif key == ord('2'):  # Fast
            self.perf_config = PerformanceConfig("fast")
            self._apply_new_settings()
            print("🚀 Fast 모드")
        elif key == ord('3'):  # Balanced
            self.perf_config = PerformanceConfig("balanced")
            self._apply_new_settings()
            print("⚖️ Balanced 모드")
        elif key == ord('h'):
            self._print_stats()
        elif key == ord('c'):
            self.detection_cache.clear()
            print("🧹 캐시 정리")
        elif key == ord('t'):
            if not self.dual_layout_active:
                self.activate_handover_mode('north')
        
        return 'continue'
    
    def _apply_new_settings(self):
        """새 설정 적용"""
        self.frame_skip = self.perf_config.get('frame_skip')
        self.detection_interval = self.perf_config.get('detection_interval')
        self.window_width = self.perf_config.get('window_width')
        self.window_height = self.perf_config.get('window_height')
        
        # 윈도우 크기 변경
        cv2.resizeWindow("Optimized Camera System", self.window_width, self.window_height)
    
    def _print_stats(self):
        """통계 출력"""
        print(f"\n📊 성능 통계 ({self.perf_config.performance_level} 모드):")
        print(f"  현재 FPS: {self.fps_current:.1f}")
        print(f"  총 프레임: {self.stats['total_frames']}")
        print(f"  처리 프레임: {self.stats['processed_frames']}")
        print(f"  스킵 프레임: {self.stats['skipped_frames']}")
        processing_rate = (self.stats['processed_frames'] / max(self.stats['total_frames'], 1)) * 100
        print(f"  처리율: {processing_rate:.1f}%")
        print(f"  탐지 호출: {self.stats['detection_calls']}")
        print(f"  캐시 히트: {self.stats['cache_hits']}")
        print(f"  캐시 크기: {len(self.detection_cache)}")
    
    def run(self):
        """최적화된 메인 루프"""
        print(f"\n⚡ 최적화 시스템 시작! ({self.perf_config.performance_level} 모드)")
        print("단축키:")
        print("  1: Ultra Fast | 2: Fast | 3: Balanced")
        print("  h: 통계 | c: 캐시정리 | t: 듀얼모드 | q: 종료")
        
        cv2.namedWindow("Optimized Camera System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Optimized Camera System", self.window_width, self.window_height)
        
        try:
            while True:
                loop_start = time.time()
                self.frame_counter += 1
                self.stats['total_frames'] += 1
                
                # 프레임 읽기
                current_frame, next_frame = self.camera_manager.get_frames()
                
                if current_frame is None:
                    time.sleep(0.01)
                    continue
                
                # 프레임 스킵 적용
                if self.frame_counter % self.frame_skip != 0:
                    self.stats['skipped_frames'] += 1
                    continue
                
                self.stats['processed_frames'] += 1
                
                # 탐지 요청 (비동기)
                if self.frame_counter % self.detection_interval == 0:
                    if not self.detection_queue.full():
                        self.detection_queue.put((current_frame.copy(), self.frame_counter))
                
                # 캐시된 탐지 결과 가져오기
                detections = self.get_detections(self.frame_counter)
                
                # 트래커 업데이트
                tracks = self.tracker.update(detections)
                
                # 화면 구성
                display_frame, offset = self.create_simple_layout(current_frame, next_frame)
                
                # 트랙 그리기
                if self.dual_layout_active and offset > 0:
                    # 듀얼 모드
                    self.draw_simple_tracks(display_frame, tracks, 0)  # 왼쪽
                    # 오른쪽은 새로운 탐지만 간단히
                else:
                    # 싱글 모드
                    self.draw_simple_tracks(display_frame, tracks)
                
                # 정보 표시
                self.draw_simple_info(display_frame)
                
                # FPS 업데이트
                self.update_fps()
                
                # 화면 출력
                cv2.imshow("Optimized Camera System", display_frame)
                
                # 키보드 입력
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    action = self.handle_user_input(key)
                    if action == 'quit':
                        break
                
                # 성능 제한
                loop_time = time.time() - loop_start
                target_time = 1.0 / self.target_fps
                if loop_time < target_time:
                    time.sleep(target_time - loop_time)
        
        except KeyboardInterrupt:
            print("\n⌨️ 사용자 중단")
        except Exception as e:
            print(f"\n💥 예외 발생: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """시스템 종료"""
        print("\n🛑 최적화 시스템 종료 중...")
        
        # 탐지 스레드 종료
        self.detection_running = False
        if self.detection_queue:
            self.detection_queue.put(None)
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)
        
        # 카메라 매니저 종료
        self.camera_manager.shutdown()
        
        cv2.destroyAllWindows()
        
        # 최종 통계
        self._print_stats()
        print("⚡ 최적화 시스템 종료 완료")


def main():
    """메인 함수"""
    load_dotenv()
    
    # 환경 변수 설정 확인
    performance_level = os.getenv("PERFORMANCE_LEVEL", "fast")
    start_camera = os.getenv("CURRENT_CCTV_NAME", "죽평")
    
    print(f"🎯 성능 레벨: {performance_level}")
    print(f"📹 시작 카메라: {start_camera}")
    
    system = OptimizedDualCameraSystem()
    
    if system.start_with_camera(start_camera):
        print(f"✅ 최적화 시스템 시작: {start_camera}")
        system.run()
    else:
        print(f"❌ 시스템 시작 실패: {start_camera}")
        print("💡 .env 파일의 CURRENT_CCTV_NAME과 CURRENT_CCTV_URL을 확인하세요")


if __name__ == "__main__":
    main()