"""
MP4 파일을 입력으로 받아 YOLO 추적 결과를 시각화하는 실험용 스크립트.
사용자는 아래 CONFIG 섹션의 변수만 수정하면 된다.
"""

from pathlib import Path
from typing import Optional, Tuple
import time

import cv2

from core.config import load_config, parse_roi
from detectors.yolo_tracker import YOLOTracker

# ============================================================
# CONFIG: 필요에 맞게 아래 값을 수정하세요.
# ============================================================
SOURCE_PATH = r"C:\Users\joker\Downloads\95315.mp4"  # 또는 "0" 등 숫자로 웹캠 사용

# ROI 설정
USE_CONFIG_ROI = True  # True면 .env의 ROI_RECT 사용
CUSTOM_ROI: Optional[Tuple[int, int, int, int]] = (4, 62, 712, 375)
ROI_ACTIVE_AT_START = True  # 실행 시 ROI 적용 여부

# 저장 설정 (None이면 저장 안 함)
SAVE_VIDEO_PATH: Optional[str] = None  # 예: r"outputs\annotated.mp4"
SAVE_VIDEO_FPS: float = 30.0  # 0이면 입력 스트림 FPS 사용
# ============================================================


def resolve_roi() -> Optional[Tuple[int, int, int, int]]:
    if USE_CONFIG_ROI:
        cfg = load_config()
        return parse_roi(cfg.get("ROI_RECT", ""))
    return CUSTOM_ROI


def draw_overlay(frame, tracks, roi, roi_active: bool):
    for track_id, x1, y1, x2, y2 in tracks:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 220),
            1,
            cv2.LINE_AA,
        )

    if roi and roi_active:
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 140, 0), 2)
        cv2.putText(
            frame,
            "ROI ON",
            (roi[0] + 5, roi[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 140, 0),
            1,
            cv2.LINE_AA,
        )
    elif roi:
        cv2.putText(
            frame,
            "ROI OFF",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )


def main():
    source = SOURCE_PATH
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    roi = resolve_roi()
    roi_active = bool(roi) and ROI_ACTIVE_AT_START

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {SOURCE_PATH}")

    tracker = YOLOTracker(model_path="yolo11n.pt")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    video_writer = None
    if SAVE_VIDEO_PATH:
        out_fps = SAVE_VIDEO_FPS or input_fps or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        Path(SAVE_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(
            SAVE_VIDEO_PATH, fourcc, out_fps, (frame_w, frame_h)
        )

    cv2.namedWindow("mp4-tracker", cv2.WINDOW_NORMAL)
    paused = False
    last_time = time.time()

    print("[INFO] Controls: q=exit, space=pause, r=ROI toggle")
    if roi:
        print(f"[INFO] ROI {roi} (active={roi_active})")
    else:
        print("[INFO] ROI not set. Edit CUSTOM_ROI or enable USE_CONFIG_ROI.")

    frame = None
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] stream ended")
                break

            tracks = tracker.update(frame, roi if roi_active else None)
            draw_overlay(frame, tracks, roi, roi_active)

            now = time.time()
            fps = 1.0 / (now - last_time) if last_time else 0.0
            last_time = now
            cv2.putText(
                frame,
                f"FPS {fps:.1f}",
                (10, frame_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if video_writer:
                video_writer.write(frame)

            cv2.imshow("mp4-tracker", frame)
        else:
            if frame is not None:
                cv2.imshow("mp4-tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("r") and roi:
            roi_active = not roi_active
            print(f"[INFO] ROI toggled -> {'ON' if roi_active else 'OFF'}")

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
