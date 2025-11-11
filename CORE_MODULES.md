# Core Module Overview

| Path | Purpose |
| --- | --- |
| `core/pipeline.py` | Main detection/tri-mode loop connecting YOLO trackers, switch controller, selection handler, and web integration. |
| `core/pipeline_components/` | Support modules extracted from the pipeline (`logging_utils.py`, `match_utils.py`, `selection.py`). |
| `core/bootstrap.py` | Helpers to refresh stream config/state before launching the pipeline. |
| `core/cam_stats.py` | Persisted statistics and feature builders for cameras (color/shape fingerprints). |
| `core/cctv_graph.py` | Load and query CCTV graph/list definitions (neighbors, URLs). |
| `core/config.py` | Global configuration parsing (ROI rectangles, env loading). |
| `core/crop_saver.py` | Save cropped detections for Re-ID debugging. |
| `core/frame_bus.py` | Thread-safe frame buffer used by Flask video feed. |
| `core/handover_features.py` | Feature extraction utilities specific to camera handover. |
| `core/history.py` | TrackHistory buffer for storing recent crops/bboxes. |
| `core/mouse_handler.py` | Legacy OpenCV mouse selector callbacks (superseded by `selection.py`, candidate for archive). |
| `core/mouse_tri.py` | Legacy tri-view click handler (now unused after SelectionHandler). |
| `core/reid_bank.py` | Core Re-ID bank implementation (banded Bhattacharyya distance). |
| `core/reid_bank_ext.py` | Extended bank utilities (distance aggregation helpers). |
| `core/reid_threshold_config.py` | Threshold configs for varying camera/segment combinations. |
| `core/roi_utils.py` | ROI transformations, segment helpers, ROI-based filters. |
| `core/stream_manager.py` | HLS stream lifecycle management per camera. |
| `core/switch_controller.py` | Manages current/neighbor stream switching, keyboard commands. |
| `core/tri_concat.py` | Concatenate multiple camera frames into tri-view. |
| `core/tri_detect.py` | Segment-aware detection splitting utilities. |
| `core/web_util.py` | Bridge between Flask web inputs and OpenCV loop (key/click queues, stats updates). |
| `core/window_utils.py` | Utility to resize OpenCV windows to fit frame dimensions. |

## Suggested Next Steps
- Move `core/mouse_handler.py` / `core/mouse_tri.py` into `archive/` since SelectionHandler replaced them.
- Delete or archive `core/pipeline copy*.py` and `pipeline1.py` after verifying no needed diffs.
- Expand `core/pipeline_components/` with more granular pieces (e.g., ROI drawing, track HUD rendering) as refactors continue.

### Mon, Nov 10, 2025  5:55:04 PM

### 2025-11-10 17:55:31
- Archived legacy mouse handlers to archive/core/ since SelectionHandler replaced them.
- Moved pipeline copy variants into archive/core/ for reference.
