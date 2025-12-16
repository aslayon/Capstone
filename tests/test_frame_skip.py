
import cv2
import os
import time

def main():
    stream_url = os.getenv("ITS_BASE_URL")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("❌ 스트림 열기 실패")
        return

    prev_time = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임 읽기 실패 또는 스트림 종료")
            break

        # 현재 프레임의 영상 시간 (ms 단위, float)
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

        if prev_time is not None:
            gap = pos_msec - prev_time
            if gap > 200:  # 0.2초 이상 건너뛴 경우 경고
                print(f"⚠️  {frame_idx}번 프레임에서 시간 점프 감지! (+{gap:.1f} ms)")
        else:
            print(f"▶ 첫 프레임 시간: {pos_msec:.1f} ms")

        prev_time = pos_msec
        frame_idx += 1

        

        

    cap.release()
    print("✅ 스트림 테스트 종료")

if __name__ == "__main__":
    main()
