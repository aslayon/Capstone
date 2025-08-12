import requests
import os
import math
import time
import hashlib
from datetime import datetime

TARGET_NAME = "지본교"
NUM_NEIGHBORS = 5
SAVE_DIR = "videos"
CHECK_INTERVAL = 10  # 초

downloaded_urls = set()

def fetch_cctv_list():
    url = "https://openapi.its.go.kr:9443/cctvInfo"
    params = {
        "apiKey": os.getenv("ITS_API_KEY"),
        "type": "all",
        "cctvType": "5",  # mp4
        "minX": "126.8", "maxX": "127.89",
        "minY": "34.9", "maxY": "35.1",
        "getType": "json"
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        return data["response"]["data"]
    except Exception as e:
        print("⚠️ API 요청 실패:", e)
        return []

def calc_dist(c1, c2):
    return math.sqrt((c1["coordx"] - c2["coordx"])**2 + (c1["coordy"] - c2["coordy"])**2)

def download_video(name, url):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = hashlib.md5(url.encode()).hexdigest()[:8]
    save_path = os.path.join(SAVE_DIR, name, f"{ts}_{fname}.mp4")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        r = requests.get(url, timeout=10)
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"✅ 저장 완료: {save_path}")
    except Exception as e:
        print(f"⚠️ 다운로드 실패 ({name}):", e)

def main_loop():
    global downloaded_urls

    base_list = fetch_cctv_list()
    target = next((c for c in base_list if TARGET_NAME in c["cctvname"]), None)
    if not target:
        print("⚠️ 지본교 CCTV 찾기 실패")
        return

    # 인접 CCTV 미리 고정
    neighbors = sorted(base_list, key=lambda c: calc_dist(c, target))[:NUM_NEIGHBORS]
    neighbor_names = set(c["cctvname"] for c in neighbors)
    print("🎯 수집 대상 CCTV:", neighbor_names)

    while True:
        print(f"\n🕒 {datetime.now().strftime('%H:%M:%S')} - 최신 영상 수집 중...")
        cctvs = fetch_cctv_list()
        for c in cctvs:
            name, url = c["cctvname"], c["cctvurl"]
            if name not in neighbor_names:
                continue
            if url in downloaded_urls:
                print(f"⏩ 이미 받은 영상 (중복): {name}")
                continue
            downloaded_urls.add(url)
            download_video(name, url)

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main_loop()
