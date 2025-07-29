import cv2
import os
import time
import json
import numpy as np
from detector.yolo_detector import get_vehicle_detections
from tracker.tracker_v2 import PersistentMultiTracker
from handover.handover_logic import load_cctv_list
from handover.handover_detector import HandoverDetector
from reid.feature_extractor import ReIDSystem
from utils.stream import DualCameraManager
from dotenv import load_dotenv

class DualCameraTrackingSystem:
    """듀얼 카메라 차량 추적 시스템 (실제 CCTV 데이터 기반)"""
    
    def __init__(self):
        # 핵심 시스템들
        self.tracker = PersistentMultiTracker(max_age=150, iou_threshold=0.3)
        self.handover_detector = HandoverDetector()
        self.reid_system = ReIDSystem(similarity_threshold=0.65)
        self.camera_manager = DualCameraManager()
        
        # CCTV 정보 로드
        self.cctv_list = load_cctv_list()
        self.connections = self._load_connections()
        self.current_cctv = None
        self.next_cctv = None
        
        # 화면 레이아웃
        self.window_width = 1280
        self.window_height = 720
        self.dual_layout_active = False
        
        # 성능 측정
        self.fps_current = 0.0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        
        # 통계
        self.stats = {
            'total_frames': 0,
            'dual_mode_time': 0,
            'handover_attempts': 0,
            'successful_handovers': 0,
            'reid_matches': 0
        }
        
        print("🚀 듀얼 카메라 추적 시스템 초기화 완료")
        print(f"📡 사용 가능 CCTV: {len(self.cctv_list)}개")
        print(f"🔗 연결 관계: {len(self.connections)}개")
    
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
        
        # 연결 관계에서 현재 CCTV 찾기
        for connection in self.connections:
            if current_name == connection["cctvname"]:
                # 해당 방향의 연결 찾기
                for conn in connection["connections"]:
                    if conn["direction"] == direction:
                        target_name = conn["target"]
                        next_cctv = self.find_cctv_by_name(target_name)
                        if next_cctv:
                            print(f"🎯 다음 카메라 찾음: {current_name} → {target_name} ({direction})")
                            return next_cctv
        
        print(f"❌ 연결된 카메라 없음: {current_name} → {direction}")
        return None
    
    def start_with_camera(self, cctv_name):
        """특정 카메라로 시스템 시작"""
        # 정확한 매칭을 위해 부분 문자열 검색
        for cctv in self.cctv_list:
            if cctv_name in cctv["cctvname"] or cctv["cctvname"] in cctv_name:
                self.current_cctv = cctv
                break
        
        if not self.current_cctv:
            print(f"❌ CCTV를 찾을 수 없음: {cctv_name}")
            print(f"📋 사용 가능한 CCTV 목록:")
            for i, cctv in enumerate(self.cctv_list[:10]):  # 처음 10개만 표시
                print(f"   {i+1}. {cctv['cctvname']}")
            if len(self.cctv_list) > 10:
                print(f"   ... 외 {len(self.cctv_list)-10}개")
            return False
        
        # 카메라 시작
        success = self.camera_manager.set_current_camera(
            self.current_cctv["cctvname"], 
            self.current_cctv["cctvurl"]
        )
        
        if success:
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
        
        print(f"🔄 핸드오버 모드 활성화:")
        print(f"   현재: {self.current_cctv['cctvname']}")
        print(f"   다음: {next_cctv['cctvname']}")
        print(f"   방향: {direction}")
        
        # 듀얼 카메라 모드 시작
        success = self.camera_manager.activate_dual_mode(
            next_cctv["cctvname"],
            next_cctv["cctvurl"]
        )
        
        if success:
            self.next_cctv = next_cctv
            self.dual_layout_active = True
            self.stats['handover_attempts'] += 1
            print("✅ 듀얼 카메라 모드 시작")
            return True
        else:
            print("❌ 듀얼 카메라 모드 실패")
            return False
    
    def process_handover_events(self, current_frame):
        """핸드오버 이벤트 처리"""
        handover_events = []
        current_time = time.time()
        
        # 모든 트랙에 대해 핸드오버 조건 확인
        for track in self.tracker.tracks:
            track_info = {
                'id': track.id,
                'bbox': track.get_bbox(),
                'state': track.state,
                'confidence': track.confidence_score,
                'time_since_detection': current_time - track.last_detection_time,
                'velocity': track.get_velocity()
            }
            
            # 핸드오버 조건 확인
            conditions = self.handover_detector.check_handover_conditions(track_info, current_time)
            probability_info = self.handover_detector.evaluate_handover_probability(conditions)
            
            # 핸드오버 후보 등록/업데이트
            if probability_info['is_handover']:
                if track.id not in self.handover_detector.handover_candidates:
                    candidate = self.handover_detector.register_handover_candidate(
                        track.id, track_info, conditions, probability_info
                    )
                    
                    # 방향 매핑 및 듀얼 모드 활성화
                    exit_direction = conditions.get('exit_direction')
                    if exit_direction and not self.dual_layout_active:
                        # 화면 경계 방향을 연결 관계 방향으로 매핑
                        direction_mapping = {
                            'left': 'south',
                            'right': 'north',
                            'top': 'north', 
                            'bottom': 'south'
                        }
                        
                        connection_direction = direction_mapping.get(exit_direction, exit_direction)
                        self.activate_handover_mode(connection_direction)
                    
                    handover_events.append({
                        'type': 'NEW_CANDIDATE',
                        'track_id': track.id,
                        'direction': exit_direction,
                        'candidate': candidate
                    })
                else:
                    self.handover_detector.update_handover_candidate(track.id, track_info)
            
            # 핸드오버 확정 조건
            if track.id in self.handover_detector.handover_candidates:
                candidate = self.handover_detector.handover_candidates[track.id]
                if (current_time - candidate['registered_time'] > 3.0 and 
                    candidate['status'] == 'CANDIDATE'):
                    
                    # Re-ID 시스템에 등록
                    exit_direction = conditions.get('exit_direction')
                    if exit_direction:
                        bbox = track.get_bbox()
                        class_name = getattr(track, 'class_name', 'car')
                        self.reid_system.register_lost_vehicle(
                            track.id, current_frame, bbox, class_name,
                            {'direction': exit_direction, 'cctv': self.current_cctv['cctvname']}
                        )
                    
                    confirmed = self.handover_detector.confirm_handover(track.id)
                    handover_events.append({
                        'type': 'CONFIRMED',
                        'track_id': track.id,
                        'direction': exit_direction,
                        'candidate': confirmed
                    })
        
        # 오래된 후보 정리
        self.handover_detector.cleanup_old_candidates()
        
        return handover_events
    
    def create_dual_layout(self, current_frame, next_frame):
        """듀얼 카메라 레이아웃 생성"""
        half_width = self.window_width // 2
        half_height = self.window_height
        
        # 현재 카메라 프레임
        if current_frame is not None:
            current_resized = cv2.resize(current_frame, (half_width, half_height))
        else:
            current_resized = np.zeros((half_height, half_width, 3), dtype=np.uint8)
            cv2.putText(current_resized, "CURRENT CAMERA", (50, half_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 다음 카메라 프레임  
        if next_frame is not None:
            next_resized = cv2.resize(next_frame, (half_width, half_height))
        else:
            next_resized = np.zeros((half_height, half_width, 3), dtype=np.uint8)
            cv2.putText(next_resized, "NEXT CAMERA", (50, half_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 좌우로 결합
        combined_frame = np.hstack([current_resized, next_resized])
        
        # 구분선 그리기
        cv2.line(combined_frame, (half_width, 0), (half_width, half_height), (255, 255, 255), 3)
        
        # 카메라 이름 표시
        if self.current_cctv:
            current_name = self.current_cctv['cctvname']
            # 긴 이름을 줄임
            if len(current_name) > 20:
                current_name = current_name[:20] + "..."
            cv2.putText(combined_frame, f"CURRENT: {current_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if self.next_cctv:
            next_name = self.next_cctv['cctvname']
            if len(next_name) > 20:
                next_name = next_name[:20] + "..."
            cv2.putText(combined_frame, f"NEXT: {next_name}", 
                       (half_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return combined_frame, (current_resized, next_resized, half_width)
    
    def draw_detections_dual(self, frames_info, current_detections, next_detections):
        """듀얼 모드에서 탐지 결과 그리기"""
        combined_frame, (current_resized, next_resized, half_width) = frames_info
        
        # 현재 카메라 탐지 결과
        for track_id, x1, y1, x2, y2, state, confidence in current_detections:
            # 좌표를 축소된 크기에 맞게 조정
            original_height, original_width = self.handover_detector.frame_height, self.handover_detector.frame_width
            if original_width > 0 and original_height > 0:
                scale_x = current_resized.shape[1] / original_width
                scale_y = current_resized.shape[0] / original_height
                
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                
                # 상태별 색상
                if state == "DETECTED":
                    color = (0, 255, 0)
                elif state == "PREDICTING":
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                
                # 핸드오버 후보는 보라색
                if track_id in self.handover_detector.handover_candidates:
                    color = (255, 0, 255)
                    thickness = 3
                else:
                    thickness = 2
                
                cv2.rectangle(combined_frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, thickness)
                cv2.putText(combined_frame, f"ID{track_id}", (x1_scaled, y1_scaled-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 다음 카메라 탐지 결과
        if next_detections:
            original_height, original_width = self.handover_detector.frame_height, self.handover_detector.frame_width
            if original_width > 0 and original_height > 0:
                for i, detection in enumerate(next_detections):
                    if len(detection) >= 5:
                        x1, y1, x2, y2, conf = detection[:5]
                        class_name = detection[5] if len(detection) > 5 else "car"
                        
                        # 좌표를 축소된 크기에 맞게 조정하고 우측 화면으로 이동
                        scale_x = next_resized.shape[1] / original_width
                        scale_y = next_resized.shape[0] / original_height
                        
                        x1_next = half_width + int(x1 * scale_x)
                        y1_next = int(y1 * scale_y)
                        x2_next = half_width + int(x2 * scale_x)
                        y2_next = int(y2 * scale_y)
                        
                        # 새로운 탐지는 하얀색
                        cv2.rectangle(combined_frame, (x1_next, y1_next), (x2_next, y2_next), (255, 255, 255), 2)
                        cv2.putText(combined_frame, f"NEW_{i}", (x1_next, y1_next-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return combined_frame
    
    def process_reid_matching(self, next_detections, next_frame):
        """Re-ID 매칭 처리"""
        if not self.reid_system.lost_vehicles or not next_detections:
            return []
        
        matches = self.reid_system.search_in_new_camera(
            next_detections, next_frame, 
            self.next_cctv['cctvname'] if self.next_cctv else "Unknown"
        )
        
        # 높은 유사도 매치 자동 확정
        confirmed_matches = []
        for match in matches:
            if match['similarity'] > 0.8:
                self.reid_system.confirm_match(match)
                confirmed_matches.append(match)
                self.stats['reid_matches'] += 1
                print(f"🎯 자동 Re-ID 성공: ID{match['lost_id']} → 유사도 {match['similarity']:.3f}")
        
        return confirmed_matches
    
    def complete_handover(self):
        """핸드오버 완료 - 다음 카메라로 전환"""
        if not self.dual_layout_active or not self.next_cctv:
            return False
        
        print(f"🎯 핸드오버 완료: {self.current_cctv['cctvname']} → {self.next_cctv['cctvname']}")
        
        # 카메라 전환
        success = self.camera_manager.switch_to_next()
        
        if success:
            self.current_cctv = self.next_cctv
            self.next_cctv = None
            self.dual_layout_active = False
            self.stats['successful_handovers'] += 1
            
            print("✅ 핸드오버 완료 - 싱글 카메라 모드로 복귀")
            return True
        else:
            print("❌ 핸드오버 실패")
            return False
    
    def update_fps(self):
        """FPS 계산 및 업데이트"""
        self.fps_frame_count += 1
        
        if self.fps_frame_count >= 30:
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.fps_current = self.fps_frame_count / elapsed
            self.fps_start_time = current_time
            self.fps_frame_count = 0
    
    def draw_system_info(self, frame):
        """시스템 정보 오버레이"""
        info_y = 60
        
        # 기본 정보
        cv2.putText(frame, f"FPS: {self.fps_current:.1f} | Mode: {'DUAL' if self.dual_layout_active else 'SINGLE'}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        
        # 트래킹 통계
        detected = len([t for t in self.tracker.tracks if t.state == "DETECTED"])
        predicting = len([t for t in self.tracker.tracks if t.state == "PREDICTING"])
        lost = len([t for t in self.tracker.tracks if t.state == "LOST"])
        
        cv2.putText(frame, f"Tracks: {detected}D {predicting}P {lost}L | Handover: {len(self.handover_detector.handover_candidates)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        
        # Re-ID 정보
        cv2.putText(frame, f"Lost: {len(self.reid_system.lost_vehicles)} | ReID Success: {self.stats['reid_matches']}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 듀얼 모드 상태
        if self.dual_layout_active:
            dual_time = self.camera_manager.dual_mode_start_time
            if dual_time > 0:
                elapsed = time.time() - dual_time
                cv2.putText(frame, f"DUAL MODE: {elapsed:.1f}s", 
                           (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    def handle_user_input(self, key):
        """사용자 입력 처리"""
        if key == ord('q'):
            return 'quit'
        elif key == ord('r'):
            self.tracker.select_track_by_point(-1, -1)
            print("🔄 트랙 선택 해제")
        elif key == ord('h'):
            # 핸드오버 상태 출력
            print(f"\n🔄 핸드오버 상태:")
            print(f"  듀얼 모드: {self.dual_layout_active}")
            print(f"  현재 CCTV: {self.current_cctv['cctvname'] if self.current_cctv else 'None'}")
            print(f"  다음 CCTV: {self.next_cctv['cctvname'] if self.next_cctv else 'None'}")
            print(f"  후보: {list(self.handover_detector.handover_candidates.keys())}")
            print(f"  분실 차량: {list(self.reid_system.lost_vehicles.keys())}")
        elif key == ord('s'):
            # 통계 출력
            print(f"\n📊 시스템 통계:")
            print(f"  총 프레임: {self.stats['total_frames']}")
            print(f"  핸드오버 시도: {self.stats['handover_attempts']}")
            print(f"  핸드오버 성공: {self.stats['successful_handovers']}")
            print(f"  Re-ID 매칭: {self.stats['reid_matches']}")
            
            camera_status = self.camera_manager.get_status()
            print(f"  카메라 상태: {camera_status}")
        elif key == ord('c'):
            # 수동 핸드오버 완료
            if self.dual_layout_active:
                self.complete_handover()
            else:
                print("❌ 듀얼 모드가 아님")
        elif key == ord('d'):
            # 듀얼 모드 강제 해제
            if self.dual_layout_active:
                self.camera_manager.deactivate_dual_mode()
                self.dual_layout_active = False
                self.next_cctv = None
                print("🛑 듀얼 모드 강제 해제")
        elif key == ord('t'):
            # 테스트용 핸드오버 모드 활성화
            if not self.dual_layout_active:
                self.activate_handover_mode('north')  # 죽평 → 선평교
        elif key == ord('1'):
            # 수동 카메라 전환 테스트
            test_cameras = ["죽평", "선평교", "지본교", "순천"]
            current_name = self.current_cctv['cctvname'] if self.current_cctv else ""
            
            for i, name in enumerate(test_cameras):
                if name in current_name:
                    next_index = (i + 1) % len(test_cameras)
                    next_name = test_cameras[next_index]
                    next_cctv = self.find_cctv_by_name(next_name)
                    if next_cctv:
                        print(f"🔄 수동 전환: {name} → {next_name}")
                        self.camera_manager.set_current_camera(next_cctv["cctvname"], next_cctv["cctvurl"])
                        self.current_cctv = next_cctv
                    break
        
        return 'continue'
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.dual_layout_active:
                # 듀얼 모드에서는 좌측 화면만 클릭 가능
                half_width = self.window_width // 2
                if x < half_width:
                    # 좌측 화면 좌표로 변환
                    if self.handover_detector.frame_width > 0 and self.handover_detector.frame_height > 0:
                        original_x = int(x * self.handover_detector.frame_width / half_width)
                        original_y = int(y * self.handover_detector.frame_height / self.window_height)
                        self.tracker.select_track_by_point(original_x, original_y)
                        print(f"🖱️ 듀얼 모드 클릭: ({original_x}, {original_y})")
            else:
                # 싱글 모드에서는 전체 화면 클릭 가능
                if self.handover_detector.frame_width > 0 and self.handover_detector.frame_height > 0:
                    original_x = int(x * self.handover_detector.frame_width / self.window_width)
                    original_y = int(y * self.handover_detector.frame_height / self.window_height)
                    self.tracker.select_track_by_point(original_x, original_y)
                    print(f"🖱️ 싱글 모드 클릭: ({original_x}, {original_y})")
    
    def run(self):
        """메인 실행 루프"""
        print("\n🎯 듀얼 카메라 추적 시스템 시작!")
        print("사용법:")
        print("  - 마우스 클릭: 차량 선택")
        print("  - 'r': 선택 해제")
        print("  - 'h': 핸드오버 상태")
        print("  - 's': 시스템 통계") 
        print("  - 'c': 핸드오버 완료")
        print("  - 'd': 듀얼 모드 해제")
        print("  - 't': 테스트 듀얼 모드 (north)")
        print("  - '1': 수동 카메라 전환")
        print("  - 'q': 종료")
        
        # OpenCV 윈도우 설정
        cv2.namedWindow("Dual Camera Vehicle Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dual Camera Vehicle Tracking", self.window_width, self.window_height)
        cv2.setMouseCallback("Dual Camera Vehicle Tracking", self.mouse_callback)
        
        try:
            while True:
                loop_start = time.time()
                self.stats['total_frames'] += 1
                
                # 프레임 읽기
                current_frame, next_frame = self.camera_manager.get_frames()
                
                if current_frame is None:
                    print("⚠️ 현재 프레임 없음")
                    time.sleep(0.1)
                    continue
                
                # 프레임 크기 업데이트
                self.handover_detector.update_frame_size(current_frame)
                
                # YOLO 탐지 (현재 카메라)
                current_detections = get_vehicle_detections(current_frame, conf_threshold=0.3, 
                                                          vehicle_classes=['car', 'truck'])
                
                # YOLO 탐지 (다음 카메라 - 듀얼 모드일 때만)
                next_detections = []
                if self.dual_layout_active and next_frame is not None:
                    next_detections = get_vehicle_detections(next_frame, conf_threshold=0.3,
                                                           vehicle_classes=['car', 'truck'])
                
                # 트래커 업데이트 (현재 카메라만)
                tracks = self.tracker.update(current_detections)
                
                # 핸드오버 이벤트 처리
                handover_events = self.process_handover_events(current_frame)
                
                # Re-ID 처리 (듀얼 모드일 때)
                if self.dual_layout_active and next_frame is not None:
                    reid_matches = self.process_reid_matching(next_detections, next_frame)
                    
                    # 매칭이 충분히 확실하면 자동 핸드오버 완료
                    if reid_matches and len(reid_matches) > 0:
                        best_match = max(reid_matches, key=lambda x: x['similarity'])
                        if best_match['similarity'] > 0.85:
                            print(f"🎯 높은 유사도로 자동 핸드오버: {best_match['similarity']:.3f}")
                            time.sleep(2.0)  # 2초 대기 후 전환
                            self.complete_handover()
                
                # 듀얼 모드 타임아웃 확인
                self.camera_manager.check_timeout()
                if not self.camera_manager.dual_mode and self.dual_layout_active:
                    self.dual_layout_active = False
                    self.next_cctv = None
                
                # 화면 구성
                if self.dual_layout_active and next_frame is not None:
                    # 듀얼 레이아웃
                    frames_info = self.create_dual_layout(current_frame, next_frame)
                    display_frame = self.draw_detections_dual(frames_info, tracks, next_detections)
                else:
                    # 싱글 레이아웃
                    display_frame = cv2.resize(current_frame, (self.window_width, self.window_height))
                    
                    # 트랙 그리기
                    selected_bbox = self.tracker.get_selected_bbox()
                    
                    for track_id, x1, y1, x2, y2, state, confidence in tracks:
                        # 좌표 스케일링
                        scale_x = self.window_width / current_frame.shape[1]
                        scale_y = self.window_height / current_frame.shape[0]
                        
                        x1_scaled = int(x1 * scale_x)
                        y1_scaled = int(y1 * scale_y)
                        x2_scaled = int(x2 * scale_x)
                        y2_scaled = int(y2 * scale_y)
                        
                        # 상태별 색상
                        if state == "DETECTED":
                            color = (0, 255, 0)
                        elif state == "PREDICTING":
                            color = (0, 255, 255)
                        else:
                            color = (0, 0, 255)
                        
                        # 선택된 트랙은 보라색
                        if selected_bbox and (x1, y1, x2, y2) == selected_bbox:
                            color = (255, 0, 255)
                            thickness = 4
                        else:
                            thickness = 2
                        
                        # 핸드오버 후보는 점선 효과
                        if track_id in self.handover_detector.handover_candidates:
                            color = (255, 128, 0)  # 주황색
                            thickness = 3
                        
                        cv2.rectangle(display_frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, thickness)
                        
                        label = f"ID{track_id} {state[:4]}"
                        if track_id in self.handover_detector.handover_candidates:
                            label += " [H]"
                        
                        cv2.putText(display_frame, label, (x1_scaled, y1_scaled-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 시스템 정보 오버레이
                self.draw_system_info(display_frame)
                
                # FPS 업데이트
                self.update_fps()
                
                # 화면 표시
                cv2.imshow("Dual Camera Vehicle Tracking", display_frame)
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # 키가 눌렸을 때
                    action = self.handle_user_input(key)
                    if action == 'quit':
                        break
                
                # 성능 제한 (최대 30fps)
                loop_time = time.time() - loop_start
                if loop_time < 0.033:  # 33ms = 30fps
                    time.sleep(0.033 - loop_time)
        
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
        print("\n🛑 듀얼 카메라 시스템 종료 중...")
        
        # 카메라 매니저 종료
        self.camera_manager.shutdown()
        
        # OpenCV 윈도우 정리
        cv2.destroyAllWindows()
        
        # 최종 통계 출력
        print("\n📊 최종 통계:")
        print(f"  총 처리 프레임: {self.stats['total_frames']}")
        print(f"  핸드오버 시도: {self.stats['handover_attempts']}")
        print(f"  핸드오버 성공: {self.stats['successful_handovers']}")
        print(f"  Re-ID 성공: {self.stats['reid_matches']}")
        
        success_rate = 0
        if self.stats['handover_attempts'] > 0:
            success_rate = self.stats['successful_handovers'] / self.stats['handover_attempts'] * 100
        
        print(f"  핸드오버 성공률: {success_rate:.1f}%")
        print("🎯 시스템 종료 완료")


def main():
    """메인 함수"""
    load_dotenv()
    
    # 시스템 초기화
    system = DualCameraTrackingSystem()
    
    # 시작 카메라 설정 (.env에서 읽기)
    start_camera = os.getenv("CURRENT_CCTV_NAME", "죽평")
    
    if system.start_with_camera(start_camera):
        print(f"✅ 시스템 시작 성공: {start_camera}")
        system.run()
    else:
        print(f"❌ 시스템 시작 실패: {start_camera}")
        print("💡 .env 파일의 CURRENT_CCTV_NAME을 확인하세요")


if __name__ == "__main__":
    main()