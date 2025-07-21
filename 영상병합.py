
import os
import re
import subprocess

REPEAT = 20
BASE_PATH = "downloads"
TARGET_DIR = "지본교_20250721_1841"  # 기존 다운로드된 세션 디렉토리명
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"  # ffmpeg 경로
def sanitize(name):
    return re.sub(r'[\\/:*?"<>|\[\]]', "_", name)

def merge_clips(folder, output_path):
    clip_paths = []
    for i in range(1, REPEAT + 1):
        if os.path.exists(os.path.join(folder, f"clip_{i:02d}.mp4")):
            clip_paths.append(f"file 'clip_{i:02d}.mp4'")  # 상대경로

    if not clip_paths:
        print(f"⚠️ 병합 스킵: {folder} 안에 클립 없음")
        return

    list_path = os.path.join(folder, "files.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(clip_paths))  # ✅ 진짜 줄바꿈

    subprocess.run([
        FFMPEG_PATH, "-y", "-f", "concat", "-safe", "0",
        "-i", list_path, "-c", "copy", output_path
    ], check=True)

    print(f"🎬 병합 완료: {output_path}")



def main():
    session_path = os.path.abspath(os.path.join(BASE_PATH, TARGET_DIR))
    if not os.path.exists(session_path):
        print(f"❌ 세션 폴더 없음: {session_path}")
        return

    for name in os.listdir(session_path):
        folder = os.path.abspath(os.path.join(session_path, name))
        if not os.path.isdir(folder):
            continue
        safe_name = sanitize(name)
        merged_path = os.path.join(session_path, f"{safe_name}_merged.mp4")
        print(f"▶ 병합 대상: {folder}")
        merge_clips(folder, merged_path)

if __name__ == "__main__":
    main()
