
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
CCTV_JSON_PATH = "cctv_list_2.json"
REPEAT = 20  # 30초씩 20번 = 10분
SLEEP_INTERVAL = 30

def load_cctv_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_adjacent_cctvs(base_cctv, cctv_list, radius_km=2.0):
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

def merge_clips(folder, output_path):
    list_path = os.path.join(folder, "files.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for i in range(1, REPEAT + 1):
            clip = os.path.join(folder, f"clip_{i:02d}.mp4")
            if os.path.exists(clip):
                f.write(f"file '{clip}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", list_path, "-c", "copy", output_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"🎬 병합 완료: {output_path}")

def main():
    cctv_list = load_cctv_list(CCTV_JSON_PATH)
    current = next((c for c in cctv_list if TARGET_NAME in c["cctvname"]), None)
    if not current:
        print(f"❌ 기준 CCTV '{TARGET_NAME}'를 찾을 수 없습니다.")
        return

    nearby = find_adjacent_cctvs(current, cctv_list, RADIUS_KM)
    targets = [current] + nearby[:MAX_NEIGHBORS]

    now_str = datetime.now().strftime("%Y%m%d_%H%M")
    session_dir = os.path.join(SAVE_BASE, f"{TARGET_NAME}_{now_str}")
    os.makedirs(session_dir, exist_ok=True)

    # 각 CCTV별 하위 폴더 생성
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

    print("⏳ 병합 중...")
    for cctv in targets:
        name = cctv["cctvname"].replace("/", "_")
        folder = os.path.join(session_dir, name)
        merged_path = os.path.join(session_dir, f"{name}_merged.mp4")
        merge_clips(folder, merged_path)

    print(f"[DONE] 모든 CCTV 10분 영상 저장 완료: {session_dir}")

if __name__ == "__main__":
    main()
