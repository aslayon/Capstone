# Capstone CCTV Tracking Platform

고삼터널 구간을 타깃으로 하는 CCTV 차량 추적 파이프라인입니다. Flask 기반 웹 UI와 YOLO + ReID 파이프라인을 통합해 단일/tri 모드 추적, 로그 모니터링, 카메라 핸드오버를 지원합니다.

## 주요 기능
- **실시간 추적 파이프라인**: `core/pipeline.py`에서 YOLO 추적 + ReID 기반 추적 유지.
- **웹 UI**: Flask (`examples/flask_demo/flask_template/app.py`)로 로그인, CCTV 선택, 로그 페이지 제공.
- **Tri 모드 핸드오버**: `core/switch_controller.py`가 좌/우 이웃 카메라를 로딩하고 tri 모드 전환을 지원.
- **로그 페이지**: `/logs` 페이지에서 이벤트 로그, 차량별 시작/현재 카메라, tri 모드 경로를 시각화.

## 디렉터리 구조
- `core/` - 파이프라인, 상태 관리, ROI, 웹 연동.
- `examples/flask_demo/flask_template/` - Flask 앱과 템플릿.
- `detectors/` - YOLO 추적기, 탐지 코드.
- `reid/`, `handover/`, `ui/` 등 세부 기능 모듈.
- `config/` - CCTV 그래프, 카메라 통계 JSON.
- `tracking_logs/` - 날짜별 로그 파일.

## Quick Start
1. Python 3.11 환경 준비 후 의존성 설치:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. `.env` 복사 및 수정:
   ```bash
   copy .env.example .env
   # API 키, CCTV URL 등 환경값 채우기
   ```
3. 파이프라인 실행:
   ```bash
   python app.py
   ```
4. 웹 접속 후 로그인 → `/cctv`에서 카메라 선택 → `/logs`에서 로그 확인.

## 주요 환경 변수
`.env`에 정의된 핵심 값:
- `ITS_API_KEY` / `ITS_API_URL`: 국토부 ITS API 정보.
- `CURRENT_CCTV_NAME` / `CURRENT_CCTV_URL`: 시작 카메라.
- `ROI_RECT`, `TRI_ROI_*`: 탐지 영역 정의.
- 탐지/추적 파라미터 (`DETECTION_CONFIDENCE`, `TRACKER_MAX_AGE` 등).
- 해상도, HLS 버퍼 설정, ReID 임계치 등.

## 테스트/유틸
- 단일/간단 테스트는 `tests/` 디렉터리의 스크립트 사용.
- 샘플 영상 다운로드 유틸: `python scripts/download_sample_videos.py`.

## 참고 노트
- tri 모드 경로/이웃 카메라는 `config/cctv_graph_connections.json` 기반.
- 로그 파일은 `tracking_logs/YYYY-MM-DD/session_*_vehicle_track.txt`.
- 웹 UI에서 수동/자동 새로고침이 가능하며, tri 경로는 선택 이벤트를 기준으로 재시작됩니다.
