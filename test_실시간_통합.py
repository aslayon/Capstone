import cv2
import os
import time
from detector.yolo_detector import get_vehicle_detections
from tracker.tracker_v2 import PersistentMultiTracker  # ⭐ v2 사용
from handover.handover_logic import load_cctv_list
from handover.handover_detector import HandoverDetector  # ⭐ 핸드오버 감지
from handover.camera_switcher import IntegratedHandoverSystem  # ⭐ 카메라 전환
from reid.feature_extractor import ReIDSystem  # ⭐ Re-ID 시스템
from dotenv import load_dotenv

# 전역 시스템들
tracker = PersistentMultiTracker(max_age=150, iou_threshold=0.3)  # v2 트래커
handover_detector = HandoverDetector()
reid_system = ReIDSystem(similarity_threshold=0.65)
handover_system = None
current_cctv = None

def mouse_callback(event, x, y, flags, param):
    """마우스 클릭으로 차량 선택"""
    if event == cv2.EVENT_LBUTTONDOWN:
        tracker.select_track_by_point(x, y)
        print(f"🖱️ 클릭: ({x}, {y})")

def process_handover_events(frame):
    """핸드오버 이벤트 처리"""
    global current_cctv, handover_system, reid_system
    
    handover_events = []
    current_time = time.time()
    
    # 모든 트랙에 대해 핸드오버 조건 확인
    for track in tracker.tracks:
        track_info = {
            'id': track.id,
            'bbox': track.get_bbox(),
            'state': track.state,
            'confidence': track.confidence_score,
            'time_since_detection': current_time - track.last_detection_time,
            'velocity': track.get_velocity()
        }
        
        # 핸드오버 조건 확인
        conditions = handover_detector.check_handover_conditions(track_info, current_time)
        probability_info = handover_detector.evaluate_handover_probability(conditions)
        
        # 핸드오버 후보 등록/업데이트
        if probability_info['is_handover']:
            if track.id not in handover_detector.handover_candidates:
                candidate = handover_detector.register_handover_candidate(
                    track.id, track_info, conditions, probability_info
                )
                handover_events.append({
                    'type': 'NEW_CANDIDATE',
                    'track_id': track.id,
                    'direction': handover_detector.get_handover_direction(track.id),
                    'candidate': candidate
                })
            else:
                handover_detector.update_handover_candidate(track.id, track_info)
        
        # 핸드오버 확정 확인 (3초 이상 후보 상태)
        if track.id in handover_detector.handover_candidates:
            candidate = handover_detector.handover_candidates[track.id]
            if (current_time - candidate['registered_time'] > 3.0 and 
                candidate['status'] == 'CANDIDATE'):
                
                # 분실 차량으로 Re-ID 시스템에 등록
                direction = handover_detector.get_handover_direction(track.id)
                if direction:
                    # 특징 추출 및 저장
                    bbox = track.get_bbox()
                    class_name = getattr(track, 'class_name', 'unknown')
                    reid_system.register_lost_vehicle(
                        track.id, frame, bbox, class_name,
                        {'direction': direction, 'cctv': current_cctv['cctvname']}
                    )
                
                confirmed = handover_detector.confirm_handover(track.id)
                handover_events.append({
                    'type': 'CONFIRMED',
                    'track_id': track.id,
                    'direction': direction,
                    'candidate': confirmed
                })
    
    # 오래된 후보 정리
    handover_detector.cleanup_old_candidates()
    
    return handover_events

def main():
    global tracker, handover_detector, reid_system, handover_system, current_cctv
    
    load_dotenv()
    
    print("🚀 통합 시스템 초기화 중...")
    
    # CCTV 설정
    current_cctv_name = os.getenv("CURRENT_CCTV_NAME")
    cctv_list = load_cctv_list()
    current_cctv = next((c for c in cctv_list if current_cctv_name in c["cctvname"]), None)
    
    if not current_cctv:
        print("❌ CCTV를 찾을 수 없습니다.")
        return
    
    # 통합 핸드오버 시스템 초기화
    try:
        handover_system = IntegratedHandoverSystem()
        if not handover_system.start_with_camera(current_cctv_name):
            print("❌ 핸드오버 시스템 시작 실패")
            return
        print("✅ 통합 핸드오버 시스템 시작됨")
    except Exception as e:
        print(f"⚠️ 핸드오버 시스템 오류, 기본 스트림 사용: {e}")
        # 기본 스트림으로 폴백
        stream_url = current_cctv["cctvurl"]
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print("❌ 스트림 연결 실패")
            return
        handover_system = None
    
    print(f"📺 현재 CCTV: {current_cctv_name}")
    
    # OpenCV 설정
    cv2.namedWindow("Integrated Vehicle Tracking", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Integrated Vehicle Tracking", mouse_callback)
    
    # 성능 측정 변수
    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0
    
    yolo_times = []
    handover_times = []
    
    print("\n🎯 통합 테스트 시작!")
    print("사용법:")
    print("  - 마우스 클릭: 차량 선택")
    print("  - 'r': 선택 해제")
    print("  - 'h': 핸드오버 상태 출력")
    print("  - 's': 시스템 통계")
    print("  - 'q': 종료")
    
    try:
        while True:
            loop_start = time.time()
            frame_count += 1
            fps_frame_count += 1
            
            # 프레임 읽기
            if handover_system:
                frame = handover_system.get_current_frame()
                if frame is None:
                    print("⚠️ 프레임 없음")
                    time.sleep(0.1)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ 프레임 읽기 실패")
                    time.sleep(0.1)
                    continue
            
            # 프레임 크기 업데이트
            handover_detector.update_frame_size(frame)
            
            # YOLO 탐지
            yolo_start = time.time()
            detections = get_vehicle_detections(frame, conf_threshold=0.3, vehicle_classes=['car', 'truck'])
            yolo_time = time.time() - yolo_start
            yolo_times.append(yolo_time)
            
            # 재활성화 시도
            reactivated = tracker.try_reactivate_lost_tracks(detections)
            if reactivated:
                print(f"🎉 재활성화: {reactivated}")
            
            # 트래커 업데이트
            tracks_v2 = tracker.update(detections)
            
            # Re-ID 탐색 (분실 차량이 있는 경우)
            if reid_system.lost_vehicles:
                matches = reid_system.search_in_new_camera(detections, frame, current_cctv['cctvname'])
                for match in matches:
                    if match['similarity'] > 0.8:  # 높은 유사도만 자동 확정
                        reid_system.confirm_match(match)
                        print(f"🎯 자동 Re-ID 성공: ID{match['lost_id']}")
            
            # 핸드오버 이벤트 처리
            handover_start = time.time()
            handover_events = process_handover_events(frame)
            handover_time = time.time() - handover_start
            handover_times.append(handover_time)
            
            # 핸드오버 이벤트 처리
            for event in handover_events:
                if event['type'] == 'CONFIRMED' and handover_system:
                    direction = event['direction']
                    if direction:
                        print(f"🔄 자동 카메라 전환 시도: {direction}")
                        # 실제 카메라 전환은 안정성을 위해 비활성화
                        # handover_system.process_handover_event(event)
            
            # 시각화
            vis_start = time.time()
            selected_bbox = tracker.get_selected_bbox()
            
            for track_id, x1, y1, x2, y2, state, confidence in tracks_v2:
                # 클래스 정보 가져오기
                class_name = "unknown"
                for i, det in enumerate(detections):
                    if len(det) >= 6:
                        det_bbox = det[:4]
                        if (abs(det_bbox[0] - x1) < 10 and abs(det_bbox[1] - y1) < 10):
                            class_name = det[5]
                            break
                
                # 상태별 색상
                if state == "DETECTED":
                    base_color = (0, 255, 0)      # 초록색
                elif state == "PREDICTING":
                    base_color = (0, 255, 255)    # 노란색
                elif state == "LOST":
                    base_color = (0, 0, 255)      # 빨간색
                else:
                    base_color = (128, 128, 128)  # 회색
                
                # 선택된 차량은 보라색
                current_bbox = (x1, y1, x2, y2)
                if selected_bbox and current_bbox == selected_bbox:
                    color = (255, 0, 255)  # 보라색
                    thickness = 4
                else:
                    color = base_color
                    thickness = 2
                
                # 핸드오버 후보는 점선으로
                if track_id in handover_detector.handover_candidates:
                    # 점선 효과 (간단히 두께 변경)
                    thickness = max(1, thickness - 1)
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # 라벨
                label = f"{class_name.upper()} ID{track_id}"
                if state != "DETECTED":
                    label += f" ({state[:4]})"
                if track_id in handover_detector.handover_candidates:
                    label += " [H]"  # 핸드오버 표시
                
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            vis_time = time.time() - vis_start
            
            # 정보 오버레이
            info_y = 30
            cv2.putText(frame, f"Frame: {frame_count} | FPS: {current_fps:.1f}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            
            # 상태별 통계
            detected = len([t for t in tracks_v2 if t[5] == "DETECTED"])
            predicting = len([t for t in tracks_v2 if t[5] == "PREDICTING"])
            lost = len([t for t in tracks_v2 if t[5] == "LOST"])
            
            cv2.putText(frame, f"Tracks: {detected}D {predicting}P {lost}L | Handover: {len(handover_detector.handover_candidates)}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            
            cv2.putText(frame, f"YOLO: {yolo_time*1000:.0f}ms | Lost: {len(reid_system.lost_vehicles)} | ReID: {reid_system.stats['successful_matches']}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # FPS 계산
            if fps_frame_count >= 30:
                current_time = time.time()
                current_fps = fps_frame_count / (current_time - fps_start_time)
                fps_start_time = current_time
                fps_frame_count = 0
            
            # 화면 표시
            cv2.imshow("Integrated Vehicle Tracking", frame)
            
            # 키보드 입력
            key = cv2.waitKey(33) & 0xFF  # 30fps 제한
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker.select_track_by_point(-1, -1)
                print("🔄 선택 해제")
            elif key == ord('h'):
                print(f"\n🔄 핸드오버 상태:")
                print(f"  후보: {list(handover_detector.handover_candidates.keys())}")
                stats = handover_detector.get_statistics()
                print(f"  통계: {stats}")
            elif key == ord('s'):
                print(f"\n📊 시스템 통계:")
                print(f"  프레임: {frame_count}")
                print(f"  평균 YOLO: {sum(yolo_times[-30:])/min(30, len(yolo_times))*1000:.0f}ms")
                print(f"  평균 핸드오버: {sum(handover_times[-30:])/min(30, len(handover_times))*1000:.1f}ms")
                
                reid_stats = reid_system.get_statistics()
                print(f"  Re-ID: {reid_stats}")
                
                if handover_system:
                    system_status = handover_system.get_status()
                    print(f"  시스템: {system_status}")
            elif key == ord('t'):
                # 수동 Re-ID 테스트
                if tracks_v2 and selected_bbox:
                    for track_id, x1, y1, x2, y2, state, confidence in tracks_v2:
                        if (x1, y1, x2, y2) == selected_bbox:
                            class_name = "car"  # 임시
                            reid_system.register_lost_vehicle(track_id, frame, selected_bbox, class_name)
                            print(f"🧪 수동 Re-ID 테스트: ID{track_id} 등록")
                            break
            
            # 메모리 관리
            if len(yolo_times) > 100:
                yolo_times = yolo_times[-50:]
            if len(handover_times) > 100:
                handover_times = handover_times[-50:]
    
    except Exception as e:
        print(f"💥 예외 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🛑 시스템 종료 중...")
        
        if handover_system:
            handover_system.shutdown()
        else:
            cap.release()
        
        cv2.destroyAllWindows()
        
        # 최종 통계
        print("\n📊 최종 통계:")
        print(f"  총 처리 프레임: {frame_count}")
        if yolo_times:
            print(f"  평균 YOLO: {sum(yolo_times)/len(yolo_times)*1000:.0f}ms")
        if handover_times:
            print(f"  평균 핸드오버: {sum(handover_times)/len(handover_times)*1000:.1f}ms")
        
        reid_stats = reid_system.get_statistics()
        print(f"  Re-ID 성공률: {reid_stats['success_rate']*100:.1f}%")
        
        handover_stats = handover_detector.get_statistics()
        print(f"  핸드오버 후보: {handover_stats['total_handovers']}회")

if __name__ == "__main__":
    main()