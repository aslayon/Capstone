"""
HLS 스트림 매니저가 통합된 메인 실행 파일
파일명: main_integrated.py
"""

import os
import sys
import time
from dotenv import load_dotenv

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 기존 모듈들 import
from test_dualCamera_plt import IntegratedHandoverSystem
from core.stream_manager import HLSStreamManager

def main():
    """개선된 메인 함수"""
    load_dotenv()
    
    # 환경 변수 확인
    api_key = os.getenv("ITS_API_KEY")
    stream_url = os.getenv("CURRENT_CCTV_URL")
    cctv_name = os.getenv("CURRENT_CCTV_NAME", "죽평")
    
    if not api_key:
        print("❌ ITS_API_KEY 환경변수가 설정되지 않았습니다.")
        print("💡 .env 파일에 다음을 추가하세요:")
        print("   ITS_API_KEY=your-api-key-here")
        return
    
    if not stream_url:
        print("❌ CURRENT_CCTV_URL 환경변수가 설정되지 않았습니다.")
        return
    
    print("="*60)
    print("🎬 HLS 스트림 개선 통합 시스템")
    print("="*60)
    print(f"📹 카메라: {cctv_name}")
    print(f"🔗 초기 URL: {stream_url[:50]}...")
    print(f"🔑 API 키: {api_key[:10]}...")
    print("="*60)
    
    # 1. 기존 시스템 초기화
    print("\n[1/4] 기존 시스템 초기화...")
    system = IntegratedHandoverSystem()
    
    # 2. HLS 스트림 매니저 생성 및 통합
    print("[2/4] HLS 스트림 매니저 생성...")
    stream_manager = HLSStreamManager(api_key, update_interval=20)
    
    # 시스템에 스트림 매니저 연결
    system.hls_stream_manager = stream_manager
    
    # get_frames 메서드 개선
    original_get_frames = system.get_frames
    
    def improved_get_frames():
        """HLS 스트림 매니저를 사용하는 개선된 프레임 가져오기"""
        if hasattr(system, 'hls_stream_manager') and system.hls_stream_manager:
            frame = system.hls_stream_manager.get_frame(timeout=0.05)
            if frame is not None:
                # 핸드오버용 보조 프레임 처리
                secondary_frame = None
                if system.secondary_cap:
                    ret, secondary_frame = system.secondary_cap.read()
                    if not ret:
                        secondary_frame = None
                
                return frame, secondary_frame
        
        # 폴백: 기존 방식
        return original_get_frames()
    
    system.get_frames = improved_get_frames
    
    # 3. 시스템 초기화 계속
    print("[3/4] 모듈 초기화...")
    system.initialize_modules()
    system.setup_matplotlib()
    
    # 디버깅 모드 (선택사항)
    # system.apply_debug_mode()
    
    # 4. 스트림 시작
    print("[4/4] 스트림 시작...")
    
    # HLS 스트림 매니저로 먼저 시작
    if not stream_manager.start(cctv_name, stream_url):
        print("❌ HLS 스트림 시작 실패")
        return
    
    # 기존 시스템 카메라 정보 설정
    system.current_cctv = {
        "cctvname": cctv_name,
        "connections": []  # 실제로는 DB에서 로드
    }
    
    # UI 설정
    system.ui_system.set_single_mode(cctv_name)
    system.data_manager.set_ui_mode("single", cctv_name)
    
    print("\n✅ 시스템 시작 준비 완료!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("사용법:")
    print("  🖱️  마우스 클릭: 차량 선택")
    print("  ⌨️  ESC 또는 'q': 종료")
    print("  📊  스트림 상태가 주기적으로 표시됩니다")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # 메인 루프 실행
    try:
        run_improved_loop(system, stream_manager)
    except KeyboardInterrupt:
        print("\n⌨️ 사용자 중단")
    except Exception as e:
        print(f"\n💥 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 정리
        print("\n🛑 시스템 종료 중...")
        stream_manager.stop()
        system.shutdown()
        print("✅ 종료 완료")


def run_improved_loop(system, stream_manager):
    """개선된 메인 루프"""
    frame_count = 0
    fps_start = time.time()
    last_health_check = time.time()
    
    while True:
        # 프레임 가져오기 (개선된 방식)
        current_frame, secondary_frame = system.get_frames()
        
        if current_frame is None:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        
        # 기존 처리 로직
        system.handover_manager.update_frame(
            system.current_cctv["cctvname"], current_frame
        )
        
        detections = system.process_detections(current_frame)
        tracks = system.tracker.update(detections)
        
        system.process_handover_logic(current_frame, tracks)
        system.update_handover_state()
        
        display_frame = system.create_display_frame(current_frame, secondary_frame)
        system.draw_tracks_on_matplotlib(display_frame, tracks)
        
        # 주기적 상태 체크 (5초마다)
        current_time = time.time()
        if current_time - last_health_check > 5:
            health = stream_manager.get_stream_health()
            
            # 상태 표시
            status_line = (
                f"📊 스트림 상태: "
                f"버퍼={health['buffer_size']}프레임 "
                f"[{health['buffer_health'].upper()}] | "
                f"URL갱신={health['url_updates']}회 | "
                f"재연결={health['reconnects']}회"
            )
            
            # 버퍼 상태가 critical이면 경고
            if health['buffer_health'] == 'critical':
                status_line = "⚠️ " + status_line + " - 버퍼 부족!"
            
            print(status_line)
            last_health_check = current_time
        
        # FPS 계산 (30프레임마다)
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start
            fps = 30 / elapsed if elapsed > 0 else 0
            
            print(f"📹 FPS: {fps:.1f} | 프레임: {frame_count} | "
                  f"탐지: {len(detections)} | 추적: {len(tracks) if tracks else 0}")
            
            fps_start = time.time()


def test_stream_only():
    """스트림만 테스트하는 간단한 함수"""
    load_dotenv()
    
    api_key = os.getenv("ITS_API_KEY")
    stream_url = os.getenv("CURRENT_CCTV_URL")
    cctv_name = os.getenv("CURRENT_CCTV_NAME", "죽평")
    
    if not api_key or not stream_url:
        print("❌ 환경 변수를 설정하세요")
        return
    
    print("🧪 스트림 테스트 모드")
    print("="*60)
    
    # 스트림 매니저 생성
    manager = HLSStreamManager(api_key, update_interval=15)
    
    if not manager.start(cctv_name, stream_url):
        print("❌ 스트림 시작 실패")
        return
    
    print("📹 스트리밍 시작 (ESC로 종료)")
    
    import cv2
    frame_count = 0
    start_time = time.time()
    
    while True:
        frame = manager.get_frame()
        
        if frame is not None:
            frame_count += 1
            
            # 정보 오버레이
            h, w = frame.shape[:2]
            info_text = f"Frame: {frame_count} | Buffer: {manager.frame_buffer.qsize()}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 크기 조정 (화면에 맞게)
            if w > 1280:
                scale = 1280 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            cv2.imshow("HLS Stream Test", frame)
            
            # 10초마다 통계
            if frame_count % 300 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                health = manager.get_stream_health()
                
                print(f"📊 평균 FPS: {fps:.1f} | "
                      f"URL 갱신: {health['url_updates']}회 | "
                      f"총 프레임: {health['total_frames']}")
        
        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()
    manager.stop()
    
    # 최종 통계
    total_time = time.time() - start_time
    print(f"\n📊 테스트 완료:")
    print(f"  실행 시간: {total_time:.1f}초")
    print(f"  총 프레임: {frame_count}")
    print(f"  평균 FPS: {frame_count/total_time:.1f}")
    
    final_health = manager.get_stream_health()
    print(f"  URL 갱신: {final_health['url_updates']}회")
    print(f"  버퍼 언더런: {final_health['buffer_underruns']}회")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 스트림만 테스트
        test_stream_only()
    else:
        # 전체 시스템 실행
        main()