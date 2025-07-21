'''
import os
import json
import requests
import time
import subprocess
from datetime import datetime
from geopy.distance import geodesic

SAVE_BASE = "downloads"
TARGET_NAME = "지본교"
RADIUS_KM = 2.0
MAX_NEIGHBORS = 4
CCTV_JSON_PATH = "cctv_list_5.json"
REPEAT = 20  # 30초씩 20번 = 10분
SLEEP_INTERVAL = 30

def load_cctv_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_adjacent_cctvs_by_distance(base_cctv, cctv_list, radius_km=2.0):
    base_coord = (base_cctv["coordy"], base_cctv["coordx"])
    candidates = []
    for cctv in cctv_list:
        if cctv["cctvname"] == base_cctv["cctvname"]:
            continue
        coord = (cctv["coordy"], cctv["coordx"])
        dist = geodesic(base_coord, coord).km
        if dist <= radius_km:
            candidates.append((dist, cctv))
    candidates.sort()
    return [c[1] for c in candidates]

def download_clip(name, url, output_path):
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✅ 저장됨: {output_path}")
    except Exception as e:
        print(f"⚠️ 다운로드 실패: {name} ({url}) - {e}")

def main():
    cctv_list = load_cctv_list(CCTV_JSON_PATH)
    current = next((c for c in cctv_list if TARGET_NAME in c["cctvname"]), None)
    if not current:
        print(f"❌ 기준 CCTV '{TARGET_NAME}'를 찾을 수 없습니다.")
        return

    nearby = find_adjacent_cctvs_by_distance(current, cctv_list, RADIUS_KM)
    targets = [current] + nearby[:MAX_NEIGHBORS]

    now_str = datetime.now().strftime("%Y%m%d_%H%M")
    session_dir = os.path.join(SAVE_BASE, f"{TARGET_NAME}_{now_str}")
    os.makedirs(session_dir, exist_ok=True)

    for cctv in targets:
        folder = os.path.join(session_dir, cctv["cctvname"].replace("/", "_"))
        os.makedirs(folder, exist_ok=True)

    print(f"[INFO] 다운로드 시작 (총 {REPEAT}회, {SLEEP_INTERVAL}초 간격)...")

    for round in range(1, REPEAT + 1):
        print(f"▶ 회차 {round}/{REPEAT}")
        for cctv in targets:
            name = cctv["cctvname"].replace("/", "_")
            url = cctv["cctvurl"].replace("\/", "/")
            folder = os.path.join(session_dir, name)
            output_file = os.path.join(folder, f"clip_{round:02d}.mp4")
            download_clip(name, url, output_file)
        if round < REPEAT:
            time.sleep(SLEEP_INTERVAL)

    print(f"[DONE] 다운로드 완료: {session_dir}")

if __name__ == "__main__":
    main()

'''

import os
import json
import requests
import cv2
from datetime import datetime
from geopy.distance import geodesic
import time

SAVE_BASE = "downloads"
TARGET_NAME = "지본교"
RADIUS_KM = 2.0
MAX_NEIGHBORS = 4
CCTV_JSON_PATH = "cctv_list_5.json"
REPEAT = 600  # 10분 동안 1초 간격 = 600회
SLEEP_INTERVAL = 1

def load_cctv_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_adjacent_cctvs_by_distance(base_cctv, cctv_list, radius_km=2.0):
    base_coord = (base_cctv["coordy"], base_cctv["coordx"])
    candidates = []
    for cctv in cctv_list:
        if cctv["cctvname"] == base_cctv["cctvname"]:
            continue
        coord = (cctv["coordy"], cctv["coordx"])
        dist = geodesic(base_coord, coord).km
        if dist <= radius_km:
            candidates.append((dist, cctv))
    candidates.sort()
    return [c[1] for c in candidates]

def download_clip(name, url, output_path):
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        size = os.path.getsize(output_path)
        print(f"✅ 저장됨: {output_path} ({size} bytes)")
    except Exception as e:
        print(f"⚠️ 다운로드 실패: {name} ({url}) - {e}")

def main():
    cctv_list = load_cctv_list(CCTV_JSON_PATH)
    current = next((c for c in cctv_list if TARGET_NAME in c["cctvname"]), None)
    if not current:
        print(f"❌ 기준 CCTV '{TARGET_NAME}'를 찾을 수 없습니다.")
        return

    nearby = find_adjacent_cctvs_by_distance(current, cctv_list, RADIUS_KM)
    targets = [current] + nearby[:MAX_NEIGHBORS]

    now_str = datetime.now().strftime("%Y%m%d_%H%M")
    session_dir = os.path.join(SAVE_BASE, f"{TARGET_NAME}_{now_str}_1sec")
    os.makedirs(session_dir, exist_ok=True)

    for cctv in targets:
        folder = os.path.join(session_dir, cctv["cctvname"].replace("/", "_"))
        os.makedirs(folder, exist_ok=True)

    print(f"[INFO] 1초 간격 다운로드 시작 (총 {REPEAT}회)...")

    for round in range(1, REPEAT + 1):
        print(f"▶ 회차 {round}/{REPEAT}")
        for cctv in targets:
            name = cctv["cctvname"].replace("/", "_")
            url = cctv["cctvurl"].replace("\/", "/")
            folder = os.path.join(session_dir, name)
            output_file = os.path.join(folder, f"clip_{round:03d}.mp4")
            download_clip(name, url, output_file)
        if round < REPEAT:
            time.sleep(SLEEP_INTERVAL)

    print(f"[DONE] 다운로드 완료: {session_dir}")

if __name__ == "__main__":
    main()
