import cv2
import os
import time
import json
import numpy as np
import gc
from detector.yolo_detector import get_vehicle_detections
from tracker.tracker_v2 import PersistentMultiTracker
from handover.handover_detector import HandoverDetector
from reid.feature_extractor import ReIDSystem
from dotenv import load_dotenv

class FixedDualCamera:
    """문제점 해결된 듀얼 카메라 시스템"""
    
    def __init__(self):
        # 핵심 시스템들
        self.tracker = PersistentMultiTracker(max_age=150, iou_threshold=0.3)
        self.handover_detector = HandoverDetector()
        self.reid_system = ReIDSystem(similarity_threshold=0.65)
        
        # CCTV 정보 로드
        self.cctv_list = self._load_cctv_list()
        self.connections = self._load_connections()
        self.current_cctv = None
        self.next_cctv = None
        
        # 카메라 스트림
        self.current_cap = None
        self.next_cap = None
        self.dual_mode = False
        self.dual_mode_start_time = 0
        self.dual_mode_timeout = 30.0  # 30초로 늘림
        
        # 화면 설정 (해상도 문제 해결)
        self.single_width = 800
        self.single_height = 600
        self.dual_width = 1600   # 2배로 늘림
        self.dual_height = 600   # 높이는 유지
        
        # 선택된 차량 추적
        self.selected_track_id = None
        self.handover_requested = False  # 사용자가 핸드오버를 요청했는지
        
        # 성능 최적화 설정
        self.target_fps = 25
        self.frame_skip = 0
        self.yolo_skip = 2
        
        # 성능 측정
        self.fps_current = 0.0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        
        # 메모리 관리
        self.memory_cleanup_interval = 300
        self.last_cleanup_time = time.time()
        
        # 통계
        self.stats = {
            'total_frames': 0,
            'yolo_calls': 0,
            'handover_attempts': 0,
            'successful_handovers': 0,
            'reid_matches': 0,
            'memory_cleanups': 0
        }
        
        print("🚀 문제점 해결된 듀얼 카메라 시스템 초기화")
        print(f"📺 싱글 모드: {self.single_width}x{self.single_height}")
        print(f"📺 듀얼 모드: {self.dual_width}x{self.dual_height}")
    
    def _load_cctv_list(self):
        """CCTV 목록 로드"""
        try:
            with open("cctv_list_4.json", 'r', encoding='utf-8') as f:
                cctv_list = json.load(f)
            print(f"✅ CCTV 목록 로드: {len(cctv_list)}개")
            return cctv_list
        except Exception as e:
            print(f"❌ CCTV 목록 로드 실패: {e}")
            return []
    
    def _load_connections(self):
        """연결 관계 로드"""
        try:
            with open("cctv_graph_connections.json", 'r', encoding='utf-8') as f:
                connections = json.load(f)
            print(f"✅ 연결 관계 로드: {len(connections)}개")
            return connections
        except Exception as e:
            print(f"❌ 연결 관계 로드 실패: {e}")
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
                        next_cctv = self.find_cctv_by_name(target_name)
                        if next_cctv:
                            print(f"🎯 다음 카메라 찾음: {current_name} → {target_name}")
                            return next_cctv
        
        print(f"❌ 연결된 카메라 없음: {current_name} → {direction}")
        return None
    
    def connect_camera(self, cctv_info):
        """최적화된 카메라 연결"""
        print(f"📡 카메라 연결: {cctv_info['cctvname']}")
        
        cap = cv2.VideoCapture(cctv_info['cctvurl'])
        
        # OpenCV 최적화 설정
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ 연결 성공: {frame.shape}")
                return cap
            else:
                print("⚠️ 연결되었지만 프레임 없음")
                cap.release()
                return None
        else:
            print("❌ 연결 실패")
            return None
    
    def start_with_camera(self, cctv_name):
        """특정 카메라로 시스템 시작"""
        for cctv in self.cctv_list:
            if cctv_name in cctv["cctvname"] or cctv["cctvname"] in cctv_name:
                self.current_cctv = cctv
                break
        
        if not self.current_cctv:
            print(f"❌ CCTV를 찾을 수 없음: {cctv_name}")
            return False
        
        self.current_cap = self.connect_camera(self.current_cctv)
        if self.current_cap:
            print(f"✅ 시스템 시작: {self.current_cctv['cctvname']}")
            return True
        else:
            print(f"❌ 카메라 연결 실패")
            return False
    
    def activate_dual_mode_for_selected(self, direction):
        """선택된 차량을 위한 듀얼 모드 활성화"""
        if self.dual_mode:
            print("⚠️ 이미 듀얼 모드 활성화됨")
            return True
        
        if not self.selected_track_id:
            print("❌ 선택된 차량이 없습니다")
            return False
        
        next_cctv = self.find_next_camera(direction)
        if not next_cctv:
            return False
        
        print(f"🔄 선택된 차량 ID{self.selected_track_id}을 위한 듀얼 모드 활성화")
        
        self.next_cap = self.connect_camera(next_cctv)
        if self.next_cap:
            self.next_cctv = next_cctv
            self.dual_mode = True
            self.dual_mode_start_time = time.time()
            self.handover_requested = True
            self.stats['handover_attempts'] += 1
            
            # 창 크기 변경
            cv2.resizeWindow("Fixed Dual Camera", self.dual_width, self.dual_height)
            
            print(f"✅ 듀얼 모드 시작 (ID{self.selected_track_id} 추적)")
            return True
        else:
            print("❌ 다음 카메라 연결 실패")
            return False
    
    def deactivate_dual_mode(self):
        """듀얼 모드 해제"""
        if not self.dual_mode:
            return
        
        if self.next_cap:
            self.next_cap.release()
            self.next_cap = None
        
        self.next_cctv = None
        self.dual_mode = False
        self.dual_mode_start_time = 0
        self.handover_requested = False
        
        # 창 크기 복원
        cv2.resizeWindow("Fixed Dual Camera", self.single_width, self.single_height)
        
        print("🛑 듀얼 모드 해제")
        gc.collect()
    
    def complete_handover(self):
        """핸드오버 완료"""
        if not self.dual_mode or not self.next_cap:
            print("❌ 듀얼 모드가 아니거나 다음 카메라가 없습니다")
            return False
        
        print(f"🎯 핸드오버 완료: {self.current_cctv['cctvname']} → {self.next_cctv['cctvname']}")
        
        # 현재 카메라 해제
        if self.current_cap:
            self.current_cap.release()
        
        # 다음 카메라를 현재로 변경
        self.current_cap = self.next_cap
        self.current_cctv = self.next_cctv
        
        # 듀얼 모드 해제
        self.next_cap = None
        self.next_cctv = None
        self.dual_mode = False
        self.handover_requested = False
        
        # 창 크기 복원
        cv2.resizeWindow("Fixed Dual Camera", self.single_width, self.single_height)
        
        self.stats['successful_handovers'] += 1
        print("✅ 핸드오버 완료 - 싱글 모드로 복귀")
        
        gc.collect()
        return True
    
    def get_frames(self):
        """프레임 읽기"""
        current_frame = None
        next_frame = None
        
        if self.current_cap:
            ret, frame = self.current_cap.read()
            if ret:
                current_frame = frame
        
        if self.dual_mode and self.next_cap:
            ret, frame = self.next_cap.read()
            if ret:
                next_frame = frame
        
        return current_frame, next_frame
    
    def should_run_yolo(self):
        """YOLO 실행 여부 결정"""
        self.frame_skip += 1
        if self.frame_skip > self.yolo_skip:
            self.frame_skip = 0
            return True
        return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백 - 듀얼 모드에서 좌측/우측 모두 클릭 가능"""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"🖱️ 클릭 위치: ({x}, {y})")
            
            if self.dual_mode:
                half_width = self.dual_width // 2  # 800
                
                if x < half_width:
                    # 좌측 화면 (현재 카메라) 클릭
                    original_x = x * 640 // half_width
                    original_y = y * 480 // self.dual_height
                    print(f"🎯 좌측(현재) 변환: 화면({x},{y}) → 원본({original_x},{original_y})")
                    self.select_track_by_click(original_x, original_y, camera_side='current')
                else:
                    # 우측 화면 (다음 카메라) 클릭
                    right_x = x - half_width  # 우측 화면 내 상대 좌표
                    original_x = right_x * 640 // half_width
                    original_y = y * 480 // self.dual_height
                    print(f"🎯 우측(다음) 변환: 화면({x},{y}) → 상대({right_x},{y}) → 원본({original_x},{original_y})")
                    self.select_next_camera_detection(original_x, original_y)
            else:
                # 싱글 모드: 전체 화면을 원본 좌표로 변환
                original_x = x * 640 // self.single_width
                original_y = y * 480 // self.single_height
                print(f"🎯 싱글 모드 변환: 화면({x},{y}) → 원본({original_x},{original_y})")
                self.select_track_by_click(original_x, original_y, camera_side='current')
    
    def select_track_by_click(self, x, y, camera_side='current'):
        """현재 카메라에서 트랙 선택"""
        print(f"🔍 {camera_side} 카메라 트랙 검색: 클릭 위치 ({x}, {y})")
        
        found_track = None
        for track in self.tracker.tracks:
            x1, y1, x2, y2 = track.get_bbox()
            print(f"   트랙 ID{track.id}: bbox({x1}, {y1}, {x2}, {y2})")
            
            # bbox 내부 클릭 확인 (약간의 여유 공간 추가)
            margin = 10  # 10픽셀 여유
            if (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin):
                found_track = track
                print(f"✅ 매칭됨: ID{track.id}")
                break
        
        if found_track:
            if self.selected_track_id == found_track.id:
                # 같은 트랙 클릭시 선택 해제
                self.selected_track_id = None
                print(f"❌ 트랙 선택 해제: ID{found_track.id}")
            else:
                # 새 트랙 선택
                self.selected_track_id = found_track.id
                print(f"✅ 트랙 선택: ID{found_track.id} (상태: {found_track.state})")
        else:
            # 빈 공간 클릭시 선택 해제
            if self.selected_track_id:
                print("❌ 빈 공간 클릭으로 트랙 선택 해제")
                self.selected_track_id = None
    
    def select_next_camera_detection(self, x, y):
        """다음 카메라에서 탐지된 차량과 현재 선택된 차량 매칭 제안"""
        if not self.dual_mode:
            return
        
        print(f"🔍 다음 카메라 탐지 검색: 클릭 위치 ({x}, {y})")
        
        # 현재 저장된 다음 카메라 탐지 결과에서 찾기
        if hasattr(self, 'current_next_detections') and self.current_next_detections:
            found_detection = None
            
            for i, detection in enumerate(self.current_next_detections):
                if len(detection) >= 5:
                    x1, y1, x2, y2, conf = detection[:5]
                    print(f"   탐지 N{i}: bbox({x1}, {y1}, {x2}, {y2}) conf={conf:.2f}")
                    
                    # bbox 내부 클릭 확인
                    margin = 15  # 다음 카메라는 여유를 더 크게
                    if (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin):
                        found_detection = (i, detection)
                        print(f"✅ 다음 카메라 탐지 매칭: N{i}")
                        break
            
            if found_detection and self.selected_track_id:
                # 현재 선택된 차량과 다음 카메라 탐지 매칭 제안
                detection_idx, detection = found_detection
                self.propose_handover_match(self.selected_track_id, detection_idx, detection)
            elif found_detection:
                print("💡 차량을 먼저 선택한 후 다음 카메라에서 매칭하세요")
                # 시각적 강조를 위해 해당 탐지를 임시 하이라이트
                detection_idx, detection = found_detection
                self.highlight_next_detection = detection_idx
            else:
                print("❌ 다음 카메라에서 해당 위치에 탐지된 차량 없음")
        else:
            print("❌ 다음 카메라 탐지 결과 없음")
    
    def propose_handover_match(self, track_id, detection_idx, detection):
        """핸드오버 매칭 제안"""
        x1, y1, x2, y2, conf = detection[:5]
        class_name = detection[5] if len(detection) > 5 else "unknown"
        
        print(f"\n🎯 핸드오버 매칭 제안:")
        print(f"   현재 차량: ID{track_id}")
        print(f"   다음 탐지: N{detection_idx} ({class_name}, conf={conf:.2f})")
        print(f"   다음 위치: ({x1}, {y1}, {x2}, {y2})")
        print(f"💡 'C' 키를 눌러 핸드오버를 완료하세요!")
        
        # 매칭 정보 저장
        self.proposed_match = {
            'track_id': track_id,
            'detection_idx': detection_idx,
            'detection': detection,
            'proposed_time': time.time()
        }
    
    def draw_next_camera_detections(self, display, next_detections):
        """다음 카메라 탐지 결과 그리기 (클릭 가능하게 개선)"""
        if not self.dual_mode or not next_detections:
            return
        
        # 현재 탐지 결과 저장 (클릭 시 사용)
        self.current_next_detections = next_detections
        
        half_w = self.dual_width // 2
        scale_x = half_w / 640.0
        scale_y = self.dual_height / 480.0
        
        for i, detection in enumerate(next_detections[:10]):  # 최대 10개
            if len(detection) >= 5:
                x1, y1, x2, y2, conf = detection[:5]
                class_name = detection[5] if len(detection) > 5 else "car"
                
                # 우측 화면 좌표로 변환
                x1_scaled = half_w + int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = half_w + int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                
                # 색상 결정
                if hasattr(self, 'proposed_match') and self.proposed_match and self.proposed_match['detection_idx'] == i:
                    # 제안된 매칭은 노란색으로 강조
                    color = (0, 255, 255)
                    thickness = 3
                elif hasattr(self, 'highlight_next_detection') and self.highlight_next_detection == i:
                    # 클릭된 탐지는 초록색으로 강조
                    color = (0, 255, 0)
                    thickness = 3
                else:
                    # 기본은 흰색
                    color = (255, 255, 255)
                    thickness = 2
                
                # 사각형 그리기
                cv2.rectangle(display, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, thickness)
                
                # 라벨
                label = f"N{i}({class_name[:3]})"
                if conf > 0:
                    label += f" {conf:.2f}"
                
                # 라벨 배경
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(display, 
                             (x1_scaled, y1_scaled - label_size[1] - 5),
                             (x1_scaled + label_size[0], y1_scaled), 
                             color, -1)
                
                cv2.putText(display, label, (x1_scaled, y1_scaled - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def create_display(self, current_frame, next_frame):
        """화면 구성 - 잔상 문제 해결"""
        if self.dual_mode and next_frame is not None and current_frame is not None:
            # 듀얼 모드: 해상도 2배
            half_w = self.dual_width // 2
            full_h = self.dual_height
            
            # 좌측: 현재 카메라
            left = cv2.resize(current_frame, (half_w, full_h))
            
            # 우측: 다음 카메라
            right = cv2.resize(next_frame, (half_w, full_h))
            
            # 수평 결합
            display = np.hstack([left, right])
            
            # 구분선
            cv2.line(display, (half_w, 0), (half_w, full_h), (255, 255, 255), 3)
            
            # 라벨
            cv2.putText(display, f"CURRENT: {self.current_cctv['cctvname'][-10:]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"NEXT: {self.next_cctv['cctvname'][-10:]}", 
                       (half_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 선택된 차량 정보 표시
            if self.selected_track_id:
                cv2.putText(display, f"TRACKING: ID{self.selected_track_id}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        else:
            # 싱글 모드
            if current_frame is not None:
                display = cv2.resize(current_frame, (self.single_width, self.single_height))
                cv2.putText(display, f"SINGLE: {self.current_cctv['cctvname'][-15:]}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 선택된 차량 정보 표시
                if self.selected_track_id:
                    cv2.putText(display, f"SELECTED: ID{self.selected_track_id}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            else:
                display = np.zeros((self.single_height, self.single_width, 3), dtype=np.uint8)
                cv2.putText(display, "NO FRAME", (300, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        return display
    
    def draw_tracks(self, display, tracks):
        """트랙 그리기 - bbox 위치 정확성 개선 (완전히 개선됨)"""
        if not tracks:
            return
        
        # 화면 모드에 따른 정확한 스케일링
        if self.dual_mode:
            # 듀얼 모드: 좌측 화면 (원본 640x480 → 800x600)
            half_width = self.dual_width // 2  # 800
            scale_x = half_width / 640.0       # 800/640 = 1.25
            scale_y = self.dual_height / 480.0 # 600/480 = 1.25
            offset_x = 0
        else:
            # 싱글 모드: 전체 화면 (원본 640x480 → 800x600)
            scale_x = self.single_width / 640.0   # 800/640 = 1.25
            scale_y = self.single_height / 480.0  # 600/480 = 1.25
            offset_x = 0
        
        print(f"📐 스케일링: x={scale_x:.3f}, y={scale_y:.3f}")
        
        for track_id, x1, y1, x2, y2, state, confidence in tracks:
            # 원본 좌표를 화면 좌표로 정확히 변환
            x1_scaled = int(x1 * scale_x) + offset_x
            y1_scaled = int(y1 * scale_y)  
            x2_scaled = int(x2 * scale_x) + offset_x
            y2_scaled = int(y2 * scale_y)
            
            # 화면 경계 확인
            x1_scaled = max(0, min(x1_scaled, display.shape[1] - 1))
            y1_scaled = max(0, min(y1_scaled, display.shape[0] - 1))
            x2_scaled = max(0, min(x2_scaled, display.shape[1] - 1))
            y2_scaled = max(0, min(y2_scaled, display.shape[0] - 1))
            
            # 색상 및 두께 결정
            if track_id == self.selected_track_id:
                # 선택된 차량은 보라색으로 강조
                color = (255, 0, 255)
                thickness = 4
            elif state == "DETECTED":
                color = (0, 255, 0)
                thickness = 2
            elif state == "PREDICTING":
                color = (0, 255, 255)
                thickness = 2
            else:
                color = (0, 0, 255)
                thickness = 2
            
            # 사각형 그리기
            cv2.rectangle(display, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, thickness)
            
            # 라벨 (선택된 차량과 DETECTED 상태만 표시)
            if track_id == self.selected_track_id or state == "DETECTED":
                label = f"ID{track_id}"
                if track_id == self.selected_track_id:
                    label += f" [{state}] {confidence:.2f}"
                
                # 라벨 배경
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display, 
                             (x1_scaled, y1_scaled - label_size[1] - 5),
                             (x1_scaled + label_size[0], y1_scaled), 
                             color, -1)
                
                # 라벨 텍스트
                cv2.putText(display, label, (x1_scaled, y1_scaled - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 디버그: 원본 좌표 표시 (선택된 차량만)
            if track_id == self.selected_track_id:
                debug_text = f"orig:({x1},{y1})"
                cv2.putText(display, debug_text, (x1_scaled, y2_scaled + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_next_camera_detections(self, display, next_detections):
        """다음 카메라 탐지 결과 그리기 (클릭 가능하게 개선)"""
        if not self.dual_mode or not next_detections:
            return
        
        # 현재 탐지 결과 저장 (클릭 시 사용)
        self.current_next_detections = next_detections
        
        half_w = self.dual_width // 2
        scale_x = half_w / 640.0
        scale_y = self.dual_height / 480.0
        
        for i, detection in enumerate(next_detections[:10]):  # 최대 10개
            if len(detection) >= 5:
                x1, y1, x2, y2, conf = detection[:5]
                class_name = detection[5] if len(detection) > 5 else "car"
                
                # 우측 화면 좌표로 변환
                x1_scaled = half_w + int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = half_w + int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                
                # 색상 결정
                if hasattr(self, 'proposed_match') and self.proposed_match and self.proposed_match['detection_idx'] == i:
                    # 제안된 매칭은 노란색으로 강조
                    color = (0, 255, 255)
                    thickness = 3
                elif hasattr(self, 'highlight_next_detection') and self.highlight_next_detection == i:
                    # 클릭된 탐지는 초록색으로 강조
                    color = (0, 255, 0)
                    thickness = 3
                else:
                    # 기본은 흰색
                    color = (255, 255, 255)
                    thickness = 2
                
                # 사각형 그리기
                cv2.rectangle(display, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, thickness)
                
                # 라벨
                label = f"N{i}({class_name[:3]})"
                if conf > 0:
                    label += f" {conf:.2f}"
                
                # 라벨 배경
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(display, 
                             (x1_scaled, y1_scaled - label_size[1] - 5),
                             (x1_scaled + label_size[0], y1_scaled), 
                             color, -1)
                
                cv2.putText(display, label, (x1_scaled, y1_scaled - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def complete_handover(self):
        """핸드오버 완료 (매칭 제안 기반)"""
        # 제안된 매칭이 있으면 우선 처리
        if hasattr(self, 'proposed_match') and self.proposed_match:
            track_id = self.proposed_match['track_id']
            detection = self.proposed_match['detection']
            
            print(f"🎯 제안된 매칭으로 핸드오버 완료: ID{track_id}")
            
            # Re-ID 시스템에 매칭 정보 전달 (선택사항)
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                # 현재 선택된 차량의 정보로 Re-ID 등록
                for track in self.tracker.tracks:
                    if track.id == track_id:
                        bbox = track.get_bbox()
                        self.reid_system.register_lost_vehicle(
                            track_id, self.current_frame, bbox, "car",
                            {'handover_match': detection}
                        )
                        break
            
            # 매칭 제안 초기화
            self.proposed_match = None
            if hasattr(self, 'highlight_next_detection'):
                delattr(self, 'highlight_next_detection')
        
        # 기본 핸드오버 완료 프로세스
        if not self.dual_mode or not self.next_cap:
            print("❌ 듀얼 모드가 아니거나 다음 카메라가 없습니다")
            return False
        
        print(f"🎯 핸드오버 완료: {self.current_cctv['cctvname']} → {self.next_cctv['cctvname']}")
        
        # 현재 카메라 해제
        if self.current_cap:
            self.current_cap.release()
        
        # 다음 카메라를 현재로 변경
        self.current_cap = self.next_cap
        self.current_cctv = self.next_cctv
        
        # 듀얼 모드 해제
        self.next_cap = None
        self.next_cctv = None
        self.dual_mode = False
        self.handover_requested = False
        
        # 창 크기 복원
        cv2.resizeWindow("Fixed Dual Camera", self.single_width, self.single_height)
        
        self.stats['successful_handovers'] += 1
        print("✅ 핸드오버 완료 - 싱글 모드로 복귀")
        
        # 상태 초기화
        if hasattr(self, 'current_next_detections'):
            delattr(self, 'current_next_detections')
        if hasattr(self, 'proposed_match'):
            delattr(self, 'proposed_match')
        if hasattr(self, 'highlight_next_detection'):
            delattr(self, 'highlight_next_detection')
        
        gc.collect()
        return True
    
    def update_fps(self):
        """FPS 계산"""
        current_time = time.time()
        self.fps_frame_count += 1
        
        if self.fps_frame_count % 30 == 0:
            elapsed = current_time - self.fps_start_time
            self.fps_current = self.fps_frame_count / elapsed
    
    def memory_cleanup(self):
        """주기적 메모리 정리"""
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.memory_cleanup_interval:
            print("🧹 메모리 정리 실행")
            
            # 트래커 히스토리 정리
            if hasattr(self.tracker, 'tracks'):
                for track in self.tracker.tracks:
                    if hasattr(track, 'detection_history') and len(track.detection_history) > 20:
                        track.detection_history = track.detection_history[-10:]
                    if hasattr(track, 'prediction_history') and len(track.prediction_history) > 50:
                        track.prediction_history = track.prediction_history[-25:]
            
            self.reid_system.cleanup_old_records()
            gc.collect()
            
            self.last_cleanup_time = current_time
            self.stats['memory_cleanups'] += 1
    
    def draw_performance_info(self, display):
        """성능 정보 표시"""
        info_y = display.shape[0] - 80
        
        # FPS
        fps_color = (0, 255, 0) if self.fps_current >= self.target_fps * 0.8 else (0, 165, 255)
        cv2.putText(display, f"FPS: {self.fps_current:.1f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # 모드 및 상태
        mode_text = f"{'DUAL' if self.dual_mode else 'SINGLE'}"
        if self.selected_track_id:
            mode_text += f" | SELECTED: ID{self.selected_track_id}"
        
        cv2.putText(display, mode_text, 
                   (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 통계
        cv2.putText(display, f"Frames: {self.stats['total_frames']} | YOLO: {self.stats['yolo_calls']}", 
                   (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def run(self):
        """메인 실행 루프"""
        print("\n🎯 양쪽 클릭 가능한 듀얼 카메라 시스템!")
        print("사용법:")
        print("  🖱️  마우스 클릭:")
        print("     - 좌측 화면: 현재 카메라 차량 선택/해제")
        print("     - 우측 화면: 다음 카메라 탐지와 매칭 제안")
        print("  📺 'H': 선택된 차량 핸드오버 (north)")
        print("  📺 'J': 선택된 차량 핸드오버 (south)")
        print("  ✅ 'C': 핸드오버 완료 (매칭 제안 적용)")
        print("  ❌ 'D': 듀얼 모드 해제")
        print("  🔄 'R': 선택 해제")
        print("  📊 'S': 통계")
        print("  🚪 'Q': 종료")
        print("\n💡 듀얼 모드 워크플로우:")
        print("  1. 좌측에서 차량 선택 (보라색)")
        print("  2. 우측에서 매칭할 탐지 클릭 (노란색)")
        print("  3. 'C' 키로 핸드오버 완료!")
        
        # 창 초기 설정
        cv2.namedWindow("Fixed Dual Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Fixed Dual Camera", self.single_width, self.single_height)
        cv2.setMouseCallback("Fixed Dual Camera", self.mouse_callback)
        
        try:
            while True:
                loop_start = time.time()
                self.stats['total_frames'] += 1
                
                # 프레임 읽기
                current_frame, next_frame = self.get_frames()
                
                if current_frame is None:
                    print("⚠️ 프레임 없음")
                    time.sleep(0.1)
                    continue
                
                # 메인 루프에서 현재 프레임 저장
                self.current_frame = current_frame
                
                # 프레임 크기 업데이트
                self.handover_detector.update_frame_size(current_frame)
                
                # YOLO 실행
                tracks = []
                next_detections = []
                
                if self.should_run_yolo():
                    # 현재 카메라 탐지
                    detections = get_vehicle_detections(current_frame, conf_threshold=0.4, 
                                                      vehicle_classes=['car', 'truck'])
                    tracks = self.tracker.update(detections)
                    self.stats['yolo_calls'] += 1
                    
                    # 다음 카메라 탐지 (듀얼 모드일 때만)
                    if self.dual_mode and next_frame is not None:
                        next_detections = get_vehicle_detections(next_frame, conf_threshold=0.4,
                                                               vehicle_classes=['car', 'truck'])
                else:
                    # YOLO 없이 예측만
                    tracks = self.tracker.predict_only()
                
                # 선택된 차량이 사라졌는지 확인
                if self.selected_track_id:
                    track_exists = any(track[0] == self.selected_track_id for track in tracks)
                    if not track_exists:
                        print(f"⚠️ 선택된 차량 ID{self.selected_track_id} 사라짐")
                        self.selected_track_id = None
                
                # 듀얼 모드 타임아웃 확인
                if self.dual_mode and self.dual_mode_start_time > 0:
                    if time.time() - self.dual_mode_start_time > self.dual_mode_timeout:
                        print("⏰ 듀얼 모드 타임아웃")
                        self.deactivate_dual_mode()
                 
                # 화면 구성 (새로운 프레임으로 완전히 다시 그리기)
                display = self.create_display(current_frame, next_frame)
                
                # 트랙 그리기 (현재 카메라)
                self.draw_tracks(display, tracks)
                
                # 다음 카메라 탐지 결과 그리기
                self.draw_next_camera_detections(display, next_detections)
                
                # 성능 정보 표시
                self.draw_performance_info(display)
                
                # FPS 업데이트
                self.update_fps()
                
                # 메모리 정리
                self.memory_cleanup()
                
                # 화면 표시
                cv2.imshow("Fixed Dual Camera", display)
                
                # 키보드 입력
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    # 선택된 차량 핸드오버 (north)
                    if self.selected_track_id:
                        self.activate_dual_mode_for_selected('north')
                    else:
                        print("❌ 먼저 차량을 선택하세요")
                elif key == ord('j'):
                    # 선택된 차량 핸드오버 (south)
                    if self.selected_track_id:
                        self.activate_dual_mode_for_selected('south')
                    else:
                        print("❌ 먼저 차량을 선택하세요")
                elif key == ord('c'):
                    self.complete_handover()
                elif key == ord('d'):
                    self.deactivate_dual_mode()
                elif key == ord('r'):
                    self.selected_track_id = None
                    print("🔄 차량 선택 해제")
                elif key == ord('s'):
                    print(f"\n📊 시스템 통계:")
                    print(f"  총 프레임: {self.stats['total_frames']}")
                    print(f"  YOLO 호출: {self.stats['yolo_calls']}")
                    print(f"  현재 FPS: {self.fps_current:.1f}")
                    print(f"  선택된 차량: ID{self.selected_track_id if self.selected_track_id else 'None'}")
                    print(f"  핸드오버: {self.stats['successful_handovers']}/{self.stats['handover_attempts']}")
                
                # 프레임 레이트 제한
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
        print("\n🛑 시스템 종료 중...")
        
        if self.current_cap:
            self.current_cap.release()
        if self.next_cap:
            self.next_cap.release()
        
        cv2.destroyAllWindows()
        
        # 최종 통계
        if self.stats['total_frames'] > 0:
            total_time = time.time() - (self.fps_start_time if self.fps_frame_count == 0 else self.fps_start_time - self.fps_frame_count/30)
            avg_fps = self.stats['total_frames'] / total_time if total_time > 0 else 0
            print(f"\n📊 최종 성능:")
            print(f"  평균 FPS: {avg_fps:.1f}")
            print(f"  YOLO 효율: {self.stats['yolo_calls']}/{self.stats['total_frames']} ({self.stats['yolo_calls']/self.stats['total_frames']*100:.1f}%)")
            print(f"  핸드오버 성공률: {self.stats['successful_handovers']}/{self.stats['handover_attempts']}")
        
        print("✅ 시스템 종료 완료")

def main():
    load_dotenv()
    
    system = FixedDualCamera()
    start_camera = os.getenv("CURRENT_CCTV_NAME", "죽평")
    
    if system.start_with_camera(start_camera):
        print(f"✅ 시스템 시작 성공: {start_camera}")
        system.run()
    else:
        print(f"❌ 시스템 시작 실패: {start_camera}")

if __name__ == "__main__":
    main()