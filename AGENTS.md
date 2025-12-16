# Repository Guidelines

## Project Structure & Module Organization
- `core/` — pipeline, config, stream management, history, ROI, handover helpers.
- `detectors/` — YOLO detectors and tracking utilities.
- `handover/` — cross‑camera handover logic and helpers.
- `reid/` — feature extraction and Re‑ID bank utilities.
- `ui/` — simple UI components and visual helpers.
- `utils/` — streaming helpers.
- `config/` — camera graph and related configs.
- `tests/` — small, runnable test and integration scripts.
- `scripts/` — one‑off utilities (e.g., video fetch/download).
- Root: `app.py` (entrypoint), `requirements.txt`, `.env`/`.env.example`, `instance/` (runtime DB), `assets/`.

## Build, Test, and Development Commands
- Create venv (Windows): `python -m venv .venv && .\.venv\Scripts\activate`
- Install deps: `pip install -r requirements.txt`
- Run pipeline locally: `python app.py`
- Quick tests: `python tests\test_sanity.py`, `python tests\tracker_test.py`
- Utilities: `python scripts\download_sample_videos.py` (sample data), `python scripts\api_video_fetch.py` (fetch via API)

## Coding Style & Naming Conventions
- Python 3, PEP 8, 4‑space indentation; prefer type hints.
- Files/functions use `snake_case`; classes use `PascalCase`.
- Keep module and file names in English; avoid spaces and non‑ASCII in new code.
- Use `pathlib` for paths and `logging` instead of prints in library code.

## Testing Guidelines
- Place tests under `tests/` using `test_*.py` or `*_test.py` naming.
- Keep tests small and runnable; print key metrics or output paths when helpful.
- Prefer deterministic inputs (local files) over network/video streams for unit tests.
- Example: run a single check `python tests\test_frame_skip.py`.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (scope optional), e.g., `feat(core): add ROI utils`.
- PRs: include purpose, key changes, how to run, and evidence (logs/screenshots). Link related issues.
- Keep PRs focused; avoid unrelated refactors.

## Security & Configuration Tips
- Do not hardcode credentials. Use `.env` with `python-dotenv`; never commit secrets.
- Treat `instance/app.db`, large media, and model weights (`*.pt`) as generated artifacts.
- Review `config/cam_stats.json` and `config/cctv_graph_connections.json` before changing camera/graph topology.

## Agent-Specific Instructions (에이전트 지침)
- 이 저장소 관련 에이전트 응답은 기본적으로 한국어로 제공합니다.
- 코드, 파일 경로, 식별자, 명령어는 영어 표기를 유지합니다.
- 사용자가 다른 언어를 명시하지 않는 한, 한국어로 답변합니다.
