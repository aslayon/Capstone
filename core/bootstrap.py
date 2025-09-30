# core/bootstrap.py
import os, sys, json, subprocess, time
from pathlib import Path
from typing import Optional
from core.config import load_config, save_current_cctv_url
from core.cctv_graph import load_cctv_list, find_url_by_name

# API 스크립트 경로 (레포 루트 기준)
FETCHER_PATH = Path("API에서영상받아오기.py")
OUT_JSON = Path("cctv_list_4.json")  # fetcher가 생성하는 파일명과 맞춤

def _run_fetcher_once(python_exe: Optional[str] = None, timeout: int = 20) -> bool:
    """API에서영상받아오기.py를 한 번 실행해서 최신 JSON을 생성."""
    if not FETCHER_PATH.exists():
        return False
    py = python_exe or sys.executable
    try:
        proc = subprocess.run(
            [py, str(FETCHER_PATH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            encoding="utf-8",
            errors="ignore",
        )
        if proc.returncode != 0:
            print("[bootstrap] fetcher error:", proc.stderr.strip())
            return False
        return OUT_JSON.exists()
    except Exception as e:
        print("[bootstrap] fetcher exception:", e)
        return False

def _load_json_list(path: Path) -> list:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            # pandas로 저장된 경우 기본이 UTF-8이지만, 인코딩 문제 대비
            return json.loads(path.read_text(encoding="cp949"))
        except Exception:
            return []

def refresh_initial_url() -> Optional[str]:
    """
    1) fetcher 실행 시도 → 최신 cctv_list_4.json 생성
    2) CURRENT_CCTV_NAME에 해당하는 URL 찾기
    3) .env 의 CURRENT_CCTV_URL 자동 갱신
    실패 시 기존 JSON 또는 기존 URL 유지
    """
    cfg = load_config()
    current_name = cfg.get("CURRENT_CCTV_NAME", "") or ""
    if not current_name:
        print("[bootstrap] CURRENT_CCTV_NAME 없음 → 건너뜀")
        return None

    # 1) fetcher 실행 (1회만; 쿼터 고려)
    fetched = _run_fetcher_once()

    # 2) JSON 읽기 (fetch 실패 시 기존 파일 사용)
    if not OUT_JSON.exists():
        print("[bootstrap] 최신 JSON 없음 → 기존 URL 유지")
        return None

    # 일부 케이스에서 파일 생성 직후 잠깐 락 걸릴 수 있어 짧게 재시도
    for _ in range(3):
        cctv_list = _load_json_list(OUT_JSON)
        if cctv_list: break
        time.sleep(0.2)

    if not cctv_list:
        print("[bootstrap] JSON 로드 실패 → 기존 URL 유지")
        return None

    # 3) 지점명으로 URL 매칭 (정확 일치 우선)
    url = find_url_by_name(cctv_list, current_name)
    if not url:
        # 부분 일치(contains) 폴백
        for item in cctv_list:
            name = item.get("cctvname", "")
            if current_name in name:
                url = item.get("cctvurl")
                if url: break

    if not url:
        print(f"[bootstrap] '{current_name}'에 해당하는 URL 없음 → 기존 URL 유지")
        return None

    # 4) .env 갱신
    save_current_cctv_url(url)
    print(f"[bootstrap] ✅ CURRENT_CCTV_URL 갱신 완료: {url}")
    return url
