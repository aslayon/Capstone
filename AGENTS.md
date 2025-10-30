# Repository Guidelines

This repository implements a multi-camera CV pipeline (YOLO detection, tracking, Re-ID, and handover) with optional Flask demo. Use this guide to navigate structure, run locally, and contribute consistently.

## Project Structure & Module Organization
- `core/` — pipeline, configuration, stream management, history, ROI utils.
- `detectors/` and `detector/` — YOLO-based detectors and tracker utilities.
- `handover/` — cross-camera handover logic and helpers.
- `reid/` — feature extraction and Re-ID bank utilities.
- `ui/` — simple UI components for handover/debug.
- `utils/` — streaming helpers.
- `테스트함수들/` — ad-hoc test scripts and integration checks.
- `DongOh/CCTV/CCTV_web/flask_template/` — minimal Flask demo (templates, static, app).
- `app.py` — primary local entrypoint; `requirements.txt` — Python deps; `instance/app.db` — runtime DB.

## Build, Test, and Development Commands
- Create venv (Windows): `python -m venv .venv && .\.venv\Scripts\activate`
- Install deps: `pip install -r requirements.txt`
- Run pipeline: `python app.py`
- Run Flask demo: `python DongOh\CCTV\CCTV_web\flask_template\app.py`
- Quick tests (examples):
  - `python 테스트함수들\test.py`
  - `python detectors\tracker_test.py`

## Coding Style & Naming Conventions
- Python 3, PEP 8, 4-space indentation; prefer type hints.
- Use `snake_case` for files/functions, `PascalCase` for classes.
- Keep module names and new files in English; avoid spaces and non-ASCII in new code.
- Use `pathlib` for paths and `logging` instead of prints in library code.

## Testing Guidelines
- Place new tests under `테스트함수들/` or as `test_*.py`/`*_test.py` near the code.
- Provide small, runnable scripts that demonstrate behavior and expected console output or saved image paths.
- For vision code, attach sample inputs/outputs or screenshots in PRs.

## Commit & Pull Request Guidelines
- Commits: concise imperative subject, e.g., `feat(core): add ROI utils`.
- PRs: include purpose, key changes, run instructions, and evidence (logs/screenshots). Link related issues. Keep PRs focused and under ~400 LOC when possible.

## Security & Configuration Tips
- Do not hardcode credentials; use environment variables (e.g., `.env`) and `python-dotenv`.
- Treat `instance/app.db` and large media as generated artifacts; avoid manual edits and large binary diffs.
- Review `cam_stats.json` and `cctv_graph_connections.json` before changing graph/stream topology.

