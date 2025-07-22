import cv2
import os
import time
from detector.yolo_detector import get_vehicle_detections
from tracker.tracker_test import MultiTracker
from handover.handover_logic import load_cctv_list
from dotenv import load_dotenv

tracker = MultiTracker()

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        tracker.select_track_by_point(x, y)

def main():
    load_dotenv()
    
    current_cctv_name = os.getenv("CURRENT_CCTV_NAME")
    cctv_list = load_cctv_list()
    current_cctv = next((c for c in cctv_list if current_cctv_name in c["cctvname"]), None)
    
    if not current_cctv:
        print("❌ CCTV를 찾을 수 없습니다.")
        return
    
    stream_url = current_cctv["cctvurl"]
    print(f"CCTV: {current_cctv_name}")
    print(f"URL: {stream_url[:50]}...")
    
    # 스트림 연결 테스트
    print("📡 스트림 연결 중...")
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("❌ 스트림 연결 실패")
        return
    
    print("✅ 스트림 연결 성공")
    
    # 첫 몇 프레임 테스트
    for i in range(5):
        ret, frame = cap.read()
        print(f"프레임 {i+1}: {'성공' if ret else '실패'} - {frame.shape if ret else 'None'}")
        if not ret:
            print("❌ 초기 프레임 읽기 실패")
            return
    
    cv2.namedWindow("Debug Tracking", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Debug Tracking", mouse_callback)
    
    print("\n🚀 디버깅 시작!")
    print("로그를 주의깊게 봐주세요...")
    
    frame_count = 0
    last_time = time.time()
    yolo_times = []
    frame_times = []
    
    try:
        while True:
            loop_start = time.time()
            
            # 1. 프레임 읽기 시간 측정
            read_start = time.time()
            ret, frame = cap.read()
            read_time = time.time() - read_start
            
            if not ret:
                print(f"⚠️ 프레임 {frame_count}: 읽기 실패")
                time.sleep(0.1)  # 잠깐 대기 후 재시도
                continue
            
            frame_count += 1
            
            # 2. YOLO 매 프레임 실행 (차량+트럭 탐지)
            yolo_start = time.time()
            detections = get_vehicle_detections(frame, conf_threshold=0.3, vehicle_classes=['car', 'truck'])
            yolo_time = time.time() - yolo_start
            yolo_times.append(yolo_time)
            
            # 3. 트래커 업데이트 시간 측정
            track_start = time.time()
            tracks = tracker.update(detections)
            track_time = time.time() - track_start
            
            # 4. 시각화 시간 측정 (클래스별 색상)
            vis_start = time.time()
            selected_bbox = tracker.get_selected_bbox()
            
            for i, (track_id, x1, y1, x2, y2) in enumerate(tracks):
                # 탐지 정보에서 클래스 이름 가져오기
                class_name = "unknown"
                if i < len(detections):
                    if len(detections[i]) >= 6:  # (x1, y1, x2, y2, conf, class_name)
                        class_name = detections[i][5]
                
                # 선택된 차량 확인
                current_bbox = (x1, y1, x2, y2)
                if selected_bbox and current_bbox == selected_bbox:
                    color = (0, 255, 255)  # 노란색 (선택됨)
                    thickness = 4
                else:
                    # 클래스별 색상
                    if class_name == 'car':
                        color = (0, 255, 0)  # 초록색
                    elif class_name == 'truck':
                        color = (0, 0, 255)  # 빨간색  
                    else:
                        color = (255, 0, 0)  # 파란색
                    thickness = 2
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # 라벨에 클래스 정보 포함
                label = f"{class_name.upper()} {track_id}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            vis_time = time.time() - vis_start
            
            # 5. 정보 오버레이
            info_y = 30
            cv2.putText(frame, f"Frame: {frame_count} (30fps)", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(frame, f"Cars: {len(tracks)}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(frame, f"YOLO: {yolo_time*1000:.0f}ms (EVERY FRAME)", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            info_y += 25
            cv2.putText(frame, f"Read: {read_time*1000:.1f}ms", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 6. 화면 표시 시간 측정
            display_start = time.time()
            cv2.imshow("Debug Tracking", frame)
            display_time = time.time() - display_start
            
            # 7. 전체 루프 시간 계산
            loop_time = time.time() - loop_start
            frame_times.append(loop_time)
            
            # 8. 주기적으로 성능 통계 출력
            if frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - last_time) if frame_count > 30 else 0
                avg_yolo = sum(yolo_times[-30:]) / min(30, len(yolo_times)) * 1000
                avg_loop = sum(frame_times[-30:]) / min(30, len(frame_times)) * 1000
                
                print(f"\n📊 프레임 {frame_count} 통계:")
                print(f"  FPS: {fps:.1f}")
                print(f"  평균 YOLO: {avg_yolo:.0f}ms")
                print(f"  평균 루프: {avg_loop:.0f}ms")
                print(f"  읽기: {read_time*1000:.1f}ms")
                print(f"  추적: {track_time*1000:.1f}ms") 
                print(f"  시각화: {vis_time*1000:.1f}ms")
                print(f"  화면표시: {display_time*1000:.1f}ms")
                
                last_time = current_time
            
            # 9. 너무 오래 걸리면 경고
            if loop_time > 0.2:  # 200ms 이상
                print(f"🐌 느린 프레임 {frame_count}: {loop_time*1000:.0f}ms")
                print(f"  YOLO: {yolo_time*1000:.0f}ms, 읽기: {read_time*1000:.1f}ms")
            
            # 10. 30fps로 제한 (스트림 안정화)
            key = cv2.waitKey(33) & 0xFF  # 33ms = 약 30fps
            if key == ord('q'):
                print("👋 사용자 종료 요청")
                break
            elif key == ord('i'):
                # 트래커 v2 정보 출력
                print(f"\n📍 Tracker v2 상태:")
                print(f"  활성 트랙: {len(tracker.tracks)}")
                print(f"  분실 트랙: {len(tracker.lost_tracks)}")
                print(f"  선택된 ID: {tracker.selected_id}")
                
                if tracker.selected_id:
                    info = tracker.get_track_info(tracker.selected_id)
                    if info:
                        print(f"\n🎯 선택된 트랙 상세 정보:")
                        print(f"  ID: {info['id']}")
                        print(f"  상태: {info['state']}")
                        print(f"  신뢰도: {info['confidence']:.3f}")
                        print(f"  속도: {info['velocity']}")
                        print(f"  마지막 탐지 후 경과: {info['time_since_detection']:.1f}초")
                        print(f"  총 탐지 횟수: {info['total_detections']}")
                        print(f"  예측 횟수: {info['predictions_count']}")
                
                # 상태별 통계
                states = {}
                for track_id, x1, y1, x2, y2, state, confidence in tracks_v2:
                    states[state] = states.get(state, 0) + 1
                
                print(f"\n📊 상태별 트랙 분포:")
                for state, count in states.items():
                    print(f"  {state}: {count}개")
            
            # 11. 메모리 누수 방지 (통계 배열 크기 제한)
            if len(yolo_times) > 100:
                yolo_times = yolo_times[-50:]
            if len(frame_times) > 100:
                frame_times = frame_times[-50:]
    
    except Exception as e:
        print(f"💥 예외 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 최종 통계:")
        print(f"  총 처리 프레임: {frame_count}")
        if yolo_times:
            print(f"  평균 YOLO 시간: {sum(yolo_times)/len(yolo_times)*1000:.0f}ms")
        if frame_times:
            print(f"  평균 프레임 처리: {sum(frame_times)/len(frame_times)*1000:.0f}ms")

if __name__ == "__main__":
    main()