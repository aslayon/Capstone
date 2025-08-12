"""
이벤트 기반 ReID 듀얼카메라 시스템
파일명: test_event_reid_dual.py

- 평상시: YOLO + 칼만필터만 (부드러움 유지)
- 객체 선택시: ReID 등록 
- 분실시: ReID 검색
"""
import cv2
import os
import time
import json
import numpy as np
from detector.yolo_detector import get_vehicle_detections
from tracker.tracker_test import MultiTracker, check_boundary_event
from handover.handover_logic import load_cctv_list
from reid.feature_extractor import ReIDSystem
from dotenv import load_dotenv

class EventBasedReIDDualSystem:
    """이벤트 기반 ReID 듀얼 카메라 시스템"""
    
    def __init__(self):
        # 기본 추적 시스템
        self.tracker = MultiTracker()
        
        # ReID 시스템 (선택적 사용)
        self.reid_system = ReIDSystem(similarity_threshold=0.7)
        self.selected_track_id = None  # 사용자가 선택한 트랙 ID
        self.reid_registered_tracks = set()  # ReID에 등록된 트랙들
        
        # 카메라 설정
        self.current_cap = None
        self.next_cap = None
        self.dual_mode = False
        self.dual_mode_start_time = 0
        
        # CCTV 정보
        self.cctv_list = load_cctv_list()
        self.connections = self._load_connections()
        self.current_cctv = None
        self.next_cctv = None
        
        # 성능 설정
        self.detection_interval = 2  # 2프레임마다 YOLO
        self.frame_counter = 0
        self.last_detections = []
        
        # 이벤트 기반 처리
        self.lost_track_events = {}  # 분실된 트랙 정보
        self.reid_search_active = False  # ReID 검색 활성화 여부
        
        # FPS 측정
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # 성능 통계
        self.stats = {
            'total_frames': 0,
            'yolo_calls': 0,
            'reid_registrations': 0,
            'reid_searches': 0,
            'successful_matches': 0
        }
        
        print("🚀 이벤트 기반 ReID 듀얼카메라 시스템 초기화")
        print("💡 ReID는 객체 선택/분실시에만 실행됩니다")
    
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
        """현재 카메라 시작"""
        self.current_cctv = self.find_cctv_by_name(cctv_name)
        if not self.current_cctv:
            print(f"❌ CCTV 찾기 실패: {cctv_name}")
            return False
        
        print(f"📡 카메라 연결 중: {cctv_name}")
        self.current_cap = cv2.VideoCapture(stream_url)
        
        if not self.current_cap.isOpened():
            print("❌ 카메라 연결 실패")
            return False
        
        # 최적화 설정
        self.current_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.current_cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✅ 카메라 연결 성공: {cctv_name}")
        return True
    
    def activate_dual_mode(self, direction):
        """듀얼 모드 활성화"""
        if self.dual_mode:
            return False
        
        next_cctv = self.find_next_camera(direction)
        if not next_cctv:
            print(f"❌ {direction} 방향 카메라 없음")
            return False
        
        print(f"🔄 듀얼 모드 활성화: {direction} → {next_cctv['cctvname']}")
        
        # 다음 카메라 연결
        stream_url = os.getenv("CURRENT_CCTV_URL", "")
        self.next_cap = cv2.VideoCapture(stream_url)
        
        if not self.next_cap.isOpened():
            print("❌ 다음 카메라 연결 실패")
            return False
        
        self.next_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.next_cctv = next_cctv
        self.dual_mode = True
        self.dual_mode_start_time = time.time()
        
        # ReID 검색 활성화 (분실된 객체가 있다면)
        if self.lost_track_events:
            self.reid_search_active = True
            print("🔍 ReID 검색 활성화 - 분실된 객체를 찾습니다")
        
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
        self.dual_mode = False
        self.dual_mode_start_time = 0
        self.reid_search_active = False
    
    def get_frames(self):
        """프레임 읽기"""
        current_frame = None
        next_frame = None
        
        # 현재 카메라 프레임
        if self.current_cap:
            ret, current_frame = self.current_cap.read()
            if not ret:
                current_frame = None
        
        # 다음 카메라 프레임 (듀얼 모드일 때만)
        if self.dual_mode and self.next_cap:
            ret, next_frame = self.next_cap.read()
            if not ret:
                next_frame = None
        
        return current_frame, next_frame
    
    def process_detections(self, frame):
        """탐지 처리 (성능 최적화)"""
        self.frame_counter += 1
        self.stats['total_frames'] += 1
        
        # 2프레임마다만 YOLO 실행
        if self.frame_counter % self.detection_interval == 0:
            # 640x480으로 축소하여 빠른 탐지
            small_frame = cv2.resize(frame, (640, 480))
            
            start_time = time.time()
            detections = get_vehicle_detections(
                small_frame, 
                conf_threshold=0.4,
                vehicle_classes=['car']
            )
            detection_time = time.time() - start_time
            
            self.stats['yolo_calls'] += 1
            
            # 좌표를 원본 크기로 복원
            if detections:
                scale_x = frame.shape[1] / 640
                scale_y = frame.shape[0] / 480
                
                scaled_detections = []
                for det in detections:
                    if len(det) >= 4:
                        x1, y1, x2, y2 = det[:4]
                        scaled_detections.append((
                            int(x1 * scale_x), int(y1 * scale_y),
                            int(x2 * scale_x), int(y2 * scale_y)
                        ) + det[4:])
                self.last_detections = scaled_detections
            else:
                self.last_detections = []
            
            # 성능 로그 (가끔씩만)
            if self.frame_counter % 60 == 0:  # 60프레임마다
                print(f"🔧 YOLO 성능: {detection_time*1000:.1f}ms, {len(detections)}개 탐지")
        
        return self.last_detections
    
    def on_track_selected(self, track_id, frame, bbox):
        """트랙 선택 이벤트 - ReID 등록"""
        if track_id in self.reid_registered_tracks:
            print(f"🔍 ID{track_id}는 이미 ReID에 등록됨")
            return
        
        print(f"🎯 트랙 선택: ID{track_id} - ReID 등록 시작")
        
        # 트랙 영역 크롭
        x1, y1, x2, y2 = bbox
        
        # 안전한 크롭 영역
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        vehicle_crop = frame[y1:y2, x1:x2]
        
        if vehicle_crop.size > 0:
            # ReID 시스템에 등록
            self.reid_system.register_lost_vehicle(
                track_id, 
                vehicle_crop, 
                bbox, 
                'car',
                {
                    'direction': 'unknown',
                    'cctv': self.current_cctv['cctvname'],
                    'selection_time': time.time()
                }
            )
            
            self.reid_registered_tracks.add(track_id)
            self.selected_track_id = track_id
            self.stats['reid_registrations'] += 1
            
            print(f"✅ ID{track_id} ReID 등록 완료")
        else:
            print(f"❌ ID{track_id} 크롭 실패")
    
    def on_track_lost(self, track_id):
        """트랙 분실 이벤트"""
        if track_id not in self.reid_registered_tracks:
            return  # ReID에 등록되지 않은 트랙은 무시
        
        print(f"📉 트랙 분실: ID{track_id}")
        
        # 분실 이벤트 기록
        self.lost_track_events[track_id] = {
            'lost_time': time.time(),
            'lost_cctv': self.current_cctv['cctvname']
        }
        
        # 듀얼 모드가 아니라면 활성화 시도
        if not self.dual_mode:
            print("🔄 분실된 객체로 인한 듀얼 모드 시도")
            # 방향은 간단히 결정 (실제로는 이동 방향 분석 필요)
            self.activate_dual_mode('north')
    
    def process_reid_search(self, next_detections, next_frame):
        """ReID 검색 처리 (듀얼 모드에서만)"""
        if not self.reid_search_active or not self.lost_track_events:
            return []
        
        if not next_detections or next_frame is None:
            return []
        
        print(f"🔍 ReID 검색 시작: {len(next_detections)}개 탐지에서 {len(self.lost_track_events)}개 분실 객체 찾기")
        
        start_time = time.time()
        
        # ReID 검색 실행
        matches = self.reid_system.search_in_new_camera(
            next_detections,
            next_frame, 
            self.next_cctv['cctvname'] if self.next_cctv else "Unknown"
        )
        
        search_time = time.time() - start_time
        self.stats['reid_searches'] += 1
        
        print(f"⏱️ ReID 검색 완료: {search_time*1000:.1f}ms, {len(matches)}개 매칭")
        
        # 높은 유사도 매칭 처리
        confirmed_matches = []
        for match in matches:
            if match['similarity'] > 0.75:  # 높은 임계값
                print(f"🎯 매칭 발견: ID{match['lost_id']} → 유사도 {match['similarity']:.3f}")
                
                # 분실 이벤트에서 제거
                if match['lost_id'] in self.lost_track_events:
                    del self.lost_track_events[match['lost_id']]
                
                confirmed_matches.append(match)
                self.stats['successful_matches'] += 1
        
        # 분실된 객체를 모두 찾았으면 검색 비활성화
        if not self.lost_track_events:
            self.reid_search_active = False
            print("✅ 모든 분실 객체 발견 - ReID 검색 비활성화")
        
        return confirmed_matches
    
    def update_tracker_with_events(self, detections):
        """트래커 업데이트 및 이벤트 처리"""
        # 기존 트랙 상태 저장
        old_tracks = {track.id: track.state for track in self.tracker.tracks}
        
        # 트래커 업데이트
        tracks = self.tracker.update(detections)
        
        # 새로운 트랙 상태와 비교하여 이벤트 감지
        current_track_ids = {track_id for track_id, *_ in tracks}
        
        # 분실된 트랙 감지
        for old_id, old_state in old_tracks.items():
            if old_id not in current_track_ids and old_state != "LOST":
                self.on_track_lost(old_id)
        
        return tracks
    
    def draw_layout_with_reid_info(self, current_frame, next_frame=None, reid_matches=None):
        """ReID 정보가 포함된 레이아웃"""
        if self.dual_mode and next_frame is not None:
            # 듀얼 모드: 좌우 분할
            h, w = current_frame.shape[:2]
            half_w = w // 2
            
            current_half = cv2.resize(current_frame, (half_w, h))
            next_half = cv2.resize(next_frame, (half_w, h))
            
            # ReID 매칭 정보 표시 (다음 카메라에)
            if reid_matches:
                for match in reid_matches:
                    # 매칭된 영역에 특별한 표시
                    # (실제 구현에서는 매칭된 bbox 정보 필요)
                    cv2.putText(next_half, f"MATCH ID{match['lost_id']}", 
                               (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            combined = np.hstack([current_half, next_half])
            
            # 구분선과 레이블
            cv2.line(combined, (half_w, 0), (half_w, h), (255, 255, 255), 2)
            cv2.putText(combined, "CURRENT", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "NEXT", (half_w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # ReID 상태 정보
            if self.reid_search_active:
                cv2.putText(combined, "ReID SEARCHING...", (half_w + 10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            return combined
        else:
            return current_frame.copy()
    
    def draw_tracks_with_reid_status(self, frame, tracks):
        """ReID 상태가 포함된 트랙 그리기"""
        for track_id, x1, y1, x2, y2 in tracks:
            # 색상 선택
            if track_id == self.selected_track_id:
                color = (255, 0, 255)  # 보라색 (선택된 트랙)
                thickness = 3
            elif track_id in self.reid_registered_tracks:
                color = (0, 255, 255)  # 노란색 (ReID 등록됨)
                thickness = 2
            else:
                color = (255, 0, 0)   # 파란색 (일반)
                thickness = 2
            
            # 듀얼 모드에서 좌표 조정
            if self.dual_mode and x1 >= frame.shape[1] // 2:
                continue  # 우측 영역은 스킵
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # 레이블
            label = f"ID{track_id}"
            if track_id == self.selected_track_id:
                label += " [SELECTED]"
            elif track_id in self.reid_registered_tracks:
                label += " [ReID]"
            
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def on_mouse_click(self, event, x, y, flags, param):
        """마우스 클릭으로 트랙 선택"""
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param
            
            # 듀얼 모드에서 좌측만 클릭 가능
            if self.dual_mode and x >= frame.shape[1] // 2:
                return
            
            # 원본 좌표로 변환 (듀얼 모드 고려)
            if self.dual_mode:
                # 좌측 절반을 원본 크기로 확대
                original_x = int(x * 2)
                original_y = y
            else:
                original_x = x
                original_y = y
            
            # 트랙 선택
            selected_track = self.tracker.select_track_by_point(original_x, original_y)
            
            if selected_track:
                track_id = selected_track['id']
                bbox = selected_track['bbox']
                
                # 현재 프레임에서 ReID 등록
                current_frame, _ = self.get_frames()
                if current_frame is not None:
                    self.on_track_selected(track_id, current_frame, bbox)
    
    def print_performance_stats(self):
        """성능 통계 출력"""
        print(f"\n📊 성능 통계:")
        print(f"  총 프레임: {self.stats['total_frames']}")
        print(f"  YOLO 호출: {self.stats['yolo_calls']}")
        print(f"  ReID 등록: {self.stats['reid_registrations']}")
        print(f"  ReID 검색: {self.stats['reid_searches']}")
        print(f"  성공 매칭: {self.stats['successful_matches']}")
        print(f"  현재 FPS: {self.current_fps:.1f}")
        
        if self.stats['yolo_calls'] > 0:
            yolo_rate = self.stats['yolo_calls'] / self.stats['total_frames'] * 100
            print(f"  YOLO 실행률: {yolo_rate:.1f}%")
        
        if self.stats['reid_searches'] > 0:
            reid_rate = self.stats['reid_searches'] / self.stats['total_frames'] * 100
            print(f"  ReID 실행률: {reid_rate:.1f}% (매우 낮음 = 좋음)")
    
    def update_fps(self):
        """FPS 업데이트"""
        self.fps_counter += 1
        
        if self.fps_counter >= 30:
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            
            if elapsed > 0:
                self.current_fps = self.fps_counter / elapsed
            
            self.fps_start_time = current_time
            self.fps_counter = 0
    
    def run(self):
        """메인 실행 루프"""
        print("\n🎬 이벤트 기반 ReID 듀얼카메라 시스템 시작!")
        print("사용법:")
        print("  - 마우스 클릭: 차량 선택 (ReID 등록)")
        print("  - 'd': 듀얼 모드 토글")
        print("  - 's': 카메라 전환")
        print("  - 'h': 성능 통계")
        print("  - 'q': 종료")
        
        cv2.namedWindow("Event-based ReID Dual Camera", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Event-based ReID Dual Camera", self.on_mouse_click)
        
        try:
            while True:
                # 프레임 읽기
                current_frame, next_frame = self.get_frames()
                
                if current_frame is None:
                    print("⚠️ 프레임 읽기 실패")
                    time.sleep(0.01)
                    continue
                
                # 탐지 처리 (2프레임마다만)
                detections = self.process_detections(current_frame)
                
                # 트래커 업데이트 (이벤트 처리 포함)
                tracks = self.update_tracker_with_events(detections)
                
                # ReID 검색 (듀얼 모드이고 검색 활성화시만)
                reid_matches = []
                if self.dual_mode and self.reid_search_active and next_frame is not None:
                    # 다음 카메라에서도 탐지 (필요시만)
                    next_detections = get_vehicle_detections(
                        cv2.resize(next_frame, (640, 480)), 
                        conf_threshold=0.5
                    )
                    reid_matches = self.process_reid_search(next_detections, next_frame)
                
                # 화면 구성
                display_frame = self.draw_layout_with_reid_info(current_frame, next_frame, reid_matches)
                
                # 트랙 그리기
                self.draw_tracks_with_reid_status(display_frame, tracks)
                
                # 정보 표시
                self.update_fps()
                
                info_text = f"FPS: {self.current_fps:.1f}"
                if self.selected_track_id:
                    info_text += f" | Selected: ID{self.selected_track_id}"
                if self.lost_track_events:
                    info_text += f" | Lost: {len(self.lost_track_events)}"
                if self.reid_search_active:
                    info_text += " | ReID Active"
                
                cv2.putText(display_frame, info_text, (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 화면 출력
                cv2.imshow("Event-based ReID Dual Camera", display_frame)
                
                # 키보드 입력
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    if self.dual_mode:
                        self.deactivate_dual_mode()
                    else:
                        self.activate_dual_mode('north')
                elif key == ord('s') and self.dual_mode:
                    self.switch_to_next()
                elif key == ord('h'):
                    self.print_performance_stats()
                elif key == ord('c'):
                    # 선택 해제
                    self.selected_track_id = None
                    print("🔄 트랙 선택 해제")
        
        except KeyboardInterrupt:
            print("\n⌨️ 사용자 중단")
        
        finally:
            self.shutdown()
    
    def switch_to_next(self):
        """다음 카메라로 전환"""
        if not self.dual_mode or not self.next_cap:
            return False
        
        print(f"🔄 카메라 전환: {self.current_cctv['cctvname']} → {self.next_cctv['cctvname']}")
        
        # 현재 카메라 해제
        if self.current_cap:
            self.current_cap.release()
        
        # 다음을 현재로
        self.current_cap = self.next_cap
        self.current_cctv = self.next_cctv
        
        # 듀얼 모드 종료
        self.next_cap = None
        self.next_cctv = None
        self.dual_mode = False
        self.reid_search_active = False
        
        print("✅ 카메라 전환 완료")
        return True
    
    def shutdown(self):
        """정리"""
        print("\n🛑 시스템 종료 중...")
        
        # 최종 통계
        self.print_performance_stats()
        
        if self.current_cap:
            self.current_cap.release()
        
        if self.next_cap:
            self.next_cap.release()
        
        cv2.destroyAllWindows()
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
    
    system = EventBasedReIDDualSystem()
    
    if system.start_with_camera(cctv_name, stream_url):
        system.run()
    else:
        print("❌ 시스템 시작 실패")


if __name__ == "__main__":
    main()