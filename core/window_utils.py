# core/window_utils.py
# OpenCV 창 크기 자동 맞춤 유틸
import cv2

def fit_window_to_image(win_name: str, img) -> None:
    """현재 이미지 크기를 기준으로 OpenCV 창 크기를 화면 해상도 내에서 자동 맞춤."""
    if img is None:
        return
    h, w = img.shape[:2]

    # 화면 해상도 (Windows 우선, 실패 시 폴백)
    try:
        import ctypes
        user32 = ctypes.windll.user32
        screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        screen_w, screen_h = 1920, 1080

    # 화면 안에 들어오도록 축소 (확대는 하지 않음)
    scale = min(screen_w / w, screen_h / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)

    # 창 모드를 NORMAL로 만들어야 resizeWindow가 적용됨
    try:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    except cv2.error:
        pass  # 이미 생성된 경우 무시

    cv2.resizeWindow(win_name, new_w, new_h)
