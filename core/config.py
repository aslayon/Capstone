# config.py
#.env 로드 + CURRENT_CCTV_URL 자동 갱신 함수
import os
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(".env")

def load_config():
    # .env 로드
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    cfg = {
        "ITS_API_URL": os.getenv("ITS_API_URL", ""),
        "ITS_API_KEY": os.getenv("ITS_API_KEY", ""),
        "CURRENT_CCTV_NAME": os.getenv("CURRENT_CCTV_NAME", ""),
        "CURRENT_CCTV_URL": os.getenv("CURRENT_CCTV_URL", ""),
        "ROI_RECT": os.getenv("ROI_RECT", ""),  # "x1,y1,x2,y2"
        "DISPLAY_W": int(os.getenv("DISPLAY_WIDTH", "1440")),
        "DISPLAY_H": int(os.getenv("DISPLAY_HEIGHT", "960")),
        "YOLO_W": int(os.getenv("YOLO_WIDTH", "640")),
        "YOLO_H": int(os.getenv("YOLO_HEIGHT", "640")),
        "TRACKER_MAX_AGE": int(os.getenv("TRACKER_MAX_AGE", "30")),
        "TRACKER_IOU_TH": float(os.getenv("TRACKER_IOU_THRESHOLD", "0.3")),
        "DET_CONF": float(os.getenv("DETECTION_CONFIDENCE", "0.2")),
        "DEBUG_MODE": os.getenv("DEBUG_MODE", "false").lower() == "true",
    }
    return cfg

def parse_roi(roi_str):
    try:
        x1, y1, x2, y2 = map(int, roi_str.split(","))
        return (x1, y1, x2, y2)
    except:
        return None

def save_current_cctv_url(new_url: str):
    """
    .env 의 CURRENT_CCTV_URL 값을 new_url 로 덮어씀.
    """
    lines = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    key = "CURRENT_CCTV_URL="
    updated = False
    out = []
    for ln in lines:
        if ln.startswith(key):
            out.append(f"{key}{new_url}")
            updated = True
        else:
            out.append(ln)
    if not updated:
        out.append(f"{key}{new_url}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    # 프로세스 환경도 최신화
    os.environ["CURRENT_CCTV_URL"] = new_url
