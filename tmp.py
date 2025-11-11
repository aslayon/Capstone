"""
🚀 즉시 실행 가능한 통합 솔루션 (개선 버전)

사용법:
    python run_web_streaming_fixed.py

접속:
    http://localhost:5000

특징:
    - Pipeline 자동 시작 (헤드리스 모드)
    - OpenCV 창 없음 (웹 전용)
    - 실시간 상태 모니터링
    - 🆕 웹 API로 tri-mode 전환 가능!

API 엔드포인트:
    POST /api/toggle-tri-mode   # tri_prepare 토글
    GET  /api/mode              # 현재 모드 확인
"""

import os
import sys
import time
import threading

# ===== 1. 헤드리스 모드 활성화 =====
os.environ["HEADLESS"] = "1"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

print("\n" + "="*70)
print("🚀 CCTV 웹 스트리밍 서버 (개선 버전)")
print("="*70)

# ===== 2. BUS 확인 =====
try:
    from core.frame_bus import BUS
    print("✅ BUS 모듈 로드 성공")
except ImportError as e:
    print(f"❌ BUS 모듈 로드 실패: {e}")
    sys.exit(1)

# ===== 3. Pipeline 시작 =====
def start_pipeline():
    """Pipeline을 별도 스레드로 실행"""
    print("\n[PIPELINE] 시작 중...")
    
    def run_pipeline():
        try:
            from core.pipeline import run_detect
            
            # 헤드리스 모드 재확인
            import core.pipeline as pipeline_module
            if hasattr(pipeline_module, 'HEADLESS'):
                print("[PIPELINE] 헤드리스 모드 확인됨")
            
            run_detect()
            
        except Exception as e:
            print(f"[PIPELINE] ❌ 에러: {e}")
            import traceback
            traceback.print_exc()
    
    t = threading.Thread(target=run_pipeline, daemon=True, name="PipelineThread")
    t.start()
    print(f"[PIPELINE] 스레드 시작 (ID: {t.ident})")
    
    # 첫 프레임 대기
    print("[PIPELINE] 첫 프레임 대기 중", end="")
    for i in range(100):  # 10초
        frame = BUS.latest()
        if frame is not None:
            print(f"\n[PIPELINE] ✅ 첫 프레임 수신! shape={frame.shape}")
            return True
        
        if i % 10 == 9:
            print(".", end="", flush=True)
        
        time.sleep(0.1)
    
    print("\n[PIPELINE] ⚠️  프레임 수신 실패 (10초 타임아웃)")
    print("[PIPELINE] 가능한 원인:")
    print("  - 카메라 연결 실패")
    print("  - .env 설정 오류")
    print("  - pipeline.py 내부 에러")
    return False

# ===== 4. Flask App 설정 =====
try:
    print("\n[FLASK] Flask 앱 로드 중...")
    import app as app_module
    print("[FLASK] ✅ Flask 앱 로드 완료")
    
except ImportError as e:
    print(f"[FLASK] ❌ Flask 앱 로드 실패: {e}")
    sys.exit(1)

# ===== 5. 상태 모니터링 =====
def monitor_status():
    """실시간 상태 모니터링"""
    while True:
        time.sleep(5)
        frame = BUS.latest()
        if frame is not None:
            print(f"[MONITOR] 📊 프레임 수신 중: {frame.shape}")
        else:
            print(f"[MONITOR] ⚠️  프레임 없음")

# ===== 6. 메인 실행 =====
def main():
    # Pipeline 시작
    pipeline_ok = start_pipeline()
    
    if not pipeline_ok:
        print("\n" + "⚠️ "*25)
        print("경고: Pipeline 시작 실패!")
        print("웹 서버는 시작되지만 비디오가 안 나올 수 있습니다.")
        print("\n대안: 터미널 2개로 분리 실행")
        print("  터미널1: python -c 'from core.pipeline import run_detect; run_detect()'")
        print("  터미널2: python app.py")
        print("⚠️ "*25 + "\n")
        
        response = input("계속하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # 상태 모니터 시작
    monitor_thread = threading.Thread(target=monitor_status, daemon=True, name="MonitorThread")
    monitor_thread.start()
    print("[MONITOR] 상태 모니터링 시작")
    
    # Flask 서버 시작
    print("\n[FLASK] 웹 서버 시작")
    print("="*70)
    print("🌐 접속 주소:")
    print("   - http://localhost:5000")
    print("   - http://127.0.0.1:5000")
    print("\n📡 API 사용법:")
    print("   - tri-mode 토글: curl -X POST http://localhost:5000/api/toggle-tri-mode")
    print("   - 현재 모드 확인: curl http://localhost:5000/api/mode")
    print("="*70)
    print("\n종료: Ctrl+C\n")
    
    try:
        app_module.run()
    except KeyboardInterrupt:
        print("\n\n[EXIT] 서버 종료 중...")
        print("✅ 종료 완료")

if __name__ == "__main__":
    main()
