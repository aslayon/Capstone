"""
matplotlib vs OpenCV 성능 비교
파일명: test_display_performance.py

matplotlib과 OpenCV의 화면 출력 성능 비교
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

def test_matplotlib_performance():
    """matplotlib 성능 테스트"""
    print("📊 matplotlib 성능 테스트")
    
    # matplotlib 설정
    plt.ion()  # interactive mode
    fig, ax = plt.subplots()
    
    times = []
    
    for i in range(100):
        # 테스트 이미지 생성
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 박스 몇 개 그리기
        cv2.rectangle(test_image, (50, 50), (150, 100), (255, 0, 0), 2)
        cv2.rectangle(test_image, (200, 150), (300, 200), (0, 255, 0), 2)
        
        start_time = time.time()
        
        # matplotlib 출력
        ax.clear()
        ax.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {i}")
        ax.axis('off')
        plt.pause(0.01)  # 핵심!
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        if i % 20 == 0:
            print(f"  Frame {i}: {(end_time - start_time)*1000:.1f}ms")
    
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"📈 matplotlib 결과:")
    print(f"  평균 시간: {avg_time*1000:.1f}ms")
    print(f"  FPS: {fps:.1f}")
    
    plt.close()
    return avg_time, fps


def test_opencv_performance():
    """OpenCV 성능 테스트"""
    print("\n🖼️ OpenCV 성능 테스트")
    
    cv2.namedWindow("OpenCV Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("OpenCV Test", 640, 480)
    
    times = []
    
    for i in range(100):
        # 테스트 이미지 생성
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 박스 몇 개 그리기 (matplotlib과 동일)
        cv2.rectangle(test_image, (50, 50), (150, 100), (255, 0, 0), 2)
        cv2.rectangle(test_image, (200, 150), (300, 200), (0, 255, 0), 2)
        cv2.putText(test_image, f"Frame {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        start_time = time.time()
        
        # OpenCV 출력
        cv2.imshow("OpenCV Test", test_image)
        cv2.waitKey(1)  # 핵심!
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        if i % 20 == 0:
            print(f"  Frame {i}: {(end_time - start_time)*1000:.1f}ms")
    
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"📈 OpenCV 결과:")
    print(f"  평균 시간: {avg_time*1000:.1f}ms")
    print(f"  FPS: {fps:.1f}")
    
    cv2.destroyAllWindows()
    return avg_time, fps


def test_real_camera_matplotlib():
    """실제 카메라로 matplotlib 테스트"""
    print("\n📹 실제 카메라 + matplotlib 테스트")
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    stream_url = os.getenv("CURRENT_CCTV_URL", "")
    
    if not stream_url:
        print("❌ CURRENT_CCTV_URL이 설정되지 않음")
        return
    
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("❌ 카메라 연결 실패")
        return
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # matplotlib 설정 (test.py와 동일)
    plt.ion()
    fig = plt.figure()
    
    frame_times = []
    total_times = []
    
    print("ESC 키로 종료...")
    
    for i in range(200):  # 200프레임 테스트
        loop_start = time.time()
        
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            continue
        
        # RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # matplotlib 출력 (test.py와 정확히 동일)
        display_start = time.time()
        plt.imshow(frame_rgb)
        plt.title(f"Real Camera Test - Frame {i}")
        plt.axis("off")
        plt.pause(0.01)  # test.py와 동일
        plt.clf()
        display_end = time.time()
        
        loop_end = time.time()
        
        frame_times.append(display_end - display_start)
        total_times.append(loop_end - loop_start)
        
        if i % 50 == 0:
            avg_display = sum(frame_times[-50:]) / min(len(frame_times), 50)
            avg_total = sum(total_times[-50:]) / min(len(total_times), 50)
            fps = 1.0 / avg_total if avg_total > 0 else 0
            print(f"  Frame {i}: 출력 {avg_display*1000:.1f}ms, 전체 {avg_total*1000:.1f}ms, FPS {fps:.1f}")
    
    # 통계
    avg_display_time = sum(frame_times) / len(frame_times)
    avg_total_time = sum(total_times) / len(total_times)
    real_fps = 1.0 / avg_total_time if avg_total_time > 0 else 0
    
    print(f"\n📊 실제 카메라 + matplotlib 결과:")
    print(f"  평균 출력 시간: {avg_display_time*1000:.1f}ms")
    print(f"  평균 전체 시간: {avg_total_time*1000:.1f}ms")
    print(f"  실제 FPS: {real_fps:.1f}")
    
    cap.release()
    plt.close()
    
    return avg_display_time, real_fps


def test_real_camera_opencv():
    """실제 카메라로 OpenCV 테스트"""
    print("\n📹 실제 카메라 + OpenCV 테스트")
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    stream_url = os.getenv("CURRENT_CCTV_URL", "")
    
    if not stream_url:
        print("❌ CURRENT_CCTV_URL이 설정되지 않음")
        return
    
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("❌ 카메라 연결 실패")
        return
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cv2.namedWindow("Real Camera OpenCV", cv2.WINDOW_NORMAL)
    
    frame_times = []
    total_times = []
    
    print("ESC 키로 종료...")
    
    for i in range(200):  # 200프레임 테스트
        loop_start = time.time()
        
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 정보 추가
        cv2.putText(frame, f"Real Camera Test - Frame {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # OpenCV 출력
        display_start = time.time()
        cv2.imshow("Real Camera OpenCV", frame)
        key = cv2.waitKey(1) & 0xFF
        display_end = time.time()
        
        if key == 27:  # ESC
            break
        
        loop_end = time.time()
        
        frame_times.append(display_end - display_start)
        total_times.append(loop_end - loop_start)
        
        if i % 50 == 0:
            avg_display = sum(frame_times[-50:]) / min(len(frame_times), 50)
            avg_total = sum(total_times[-50:]) / min(len(total_times), 50)
            fps = 1.0 / avg_total if avg_total > 0 else 0
            print(f"  Frame {i}: 출력 {avg_display*1000:.1f}ms, 전체 {avg_total*1000:.1f}ms, FPS {fps:.1f}")
    
    # 통계
    if frame_times and total_times:
        avg_display_time = sum(frame_times) / len(frame_times)
        avg_total_time = sum(total_times) / len(total_times)
        real_fps = 1.0 / avg_total_time if avg_total_time > 0 else 0
        
        print(f"\n📊 실제 카메라 + OpenCV 결과:")
        print(f"  평균 출력 시간: {avg_display_time*1000:.1f}ms")
        print(f"  평균 전체 시간: {avg_total_time*1000:.1f}ms")
        print(f"  실제 FPS: {real_fps:.1f}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        return avg_display_time, real_fps
    
    cap.release()
    cv2.destroyAllWindows()
    return 0, 0


def main():
    """메인 비교 함수"""
    print("🎯 matplotlib vs OpenCV 성능 비교")
    print("=" * 60)
    
    # 1. 더미 이미지로 순수 출력 성능 비교
    print("1️⃣ 더미 이미지 성능 비교")
    
    plt_time, plt_fps = test_matplotlib_performance()
    cv_time, cv_fps = test_opencv_performance()
    
    print(f"\n📊 더미 이미지 비교 결과:")
    print(f"  matplotlib: {plt_time*1000:.1f}ms, {plt_fps:.1f} FPS")
    print(f"  OpenCV:     {cv_time*1000:.1f}ms, {cv_fps:.1f} FPS")
    
    if plt_fps > cv_fps:
        print(f"  🏆 matplotlib이 {plt_fps/cv_fps:.1f}배 빠름!")
    else:
        print(f"  🏆 OpenCV가 {cv_fps/plt_fps:.1f}배 빠름!")
    
    # 2. 실제 카메라로 비교
    print(f"\n" + "="*60)
    print("2️⃣ 실제 카메라 성능 비교")
    
    input("matplotlib 테스트를 시작합니다. Enter 키를 누르세요...")
    plt_real_time, plt_real_fps = test_real_camera_matplotlib()
    
    input("OpenCV 테스트를 시작합니다. Enter 키를 누르세요...")
    cv_real_time, cv_real_fps = test_real_camera_opencv()
    

    if plt_real_fps > 0 and cv_real_fps > 0:
        print(f"\n📊 실제 카메라 비교 결과:")
        print(f"  matplotlib: {plt_real_time*1000:.1f}ms, {plt_real_fps:.1f} FPS")
        print(f"  OpenCV:     {cv_real_time*1000:.1f}ms, {cv_real_fps:.1f} FPS")
        
        if plt_real_fps > cv_real_fps:
            print(f"  🏆 실제 환경에서도 matplotlib이 {plt_real_fps/cv_real_fps:.1f}배 빠름!")
            print(f"  💡 test.py가 부드러운 이유를 찾았습니다!")
        else:
            print(f"  🏆 실제 환경에서는 OpenCV가 {cv_real_fps/plt_real_fps:.1f}배 빠름!")
    
    # 최종 결론
    print(f"\n" + "="*60)
    print("🎯 최종 결론")
    print("="*60)
    
    if plt_fps > cv_fps * 1.2:  # 20% 이상 차이나면
        print("✅ matplotlib(plt.pause)가 OpenCV(cv2.imshow)보다 빠릅니다!")
        print("💡 해결책: 듀얼카메라에서도 matplotlib 사용")
        print("📝 test.py 스타일로 화면 출력 변경 필요")
    else:
        print("⚠️ 출력 방식은 큰 차이가 없습니다.")
        print("💡 다른 원인을 찾아야 합니다.")
    
    print(f"\n🚀 권장사항:")
    print(f"  1. matplotlib 방식 채택 (plt.pause(0.01))")
    print(f"  2. 복잡한 멀티스레딩 제거")
    print(f"  3. test.py와 동일한 단순 구조 유지")


if __name__ == "__main__":
    main()