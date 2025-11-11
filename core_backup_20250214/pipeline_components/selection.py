"""
Mouse/selection handling extracted from the legacy pipeline.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import cv2

from core.pipeline_components.match_utils import find_closest_track_in_history


class SelectionHandler:
    """
    Encapsulate click-handling logic for both single and tri modes.
    """

    def __init__(
        self,
        tracker,
        selected_bank,
        switcher,
        track_history_center,
        track_history_left,
        track_history_right,
        log_vehicle_tracking: Callable[..., None],
        update_web_stats: Callable[..., None],
    ) -> None:
        self.tracker = tracker
        self.selected_bank = selected_bank
        self.switcher = switcher
        self.track_history_center = track_history_center
        self.track_history_left = track_history_left
        self.track_history_right = track_history_right
        self.log_vehicle_tracking = log_vehicle_tracking
        self.update_web_stats = update_web_stats

        self.tri_prepare = False
        self.scale = 1.0
        self.tri_ui_state: Dict[str, int] = {}
        self.tracks: Sequence[Tuple[int, int, int, int, int]] = []
        self.tracks_L: Sequence[Tuple[int, int, int, int, int]] = []
        self.tracks_C: Sequence[Tuple[int, int, int, int, int]] = []
        self.tracks_R: Sequence[Tuple[int, int, int, int, int]] = []
        self.tracking_session_id: Optional[str] = None

    # --------------------------------------------------------------------- #
    # Mutable state helpers
    # --------------------------------------------------------------------- #
    def update_state(
        self,
        *,
        tri_prepare: Optional[bool] = None,
        tri_ui_state: Optional[Dict[str, int]] = None,
        tracks: Optional[Sequence] = None,
        tracks_L: Optional[Sequence] = None,
        tracks_C: Optional[Sequence] = None,
        tracks_R: Optional[Sequence] = None,
        scale: Optional[float] = None,
        tracking_session_id: Optional[str] = None,
    ) -> None:
        if tri_prepare is not None:
            self.tri_prepare = tri_prepare
        if tri_ui_state is not None:
            self.tri_ui_state = tri_ui_state
        if tracks is not None:
            self.tracks = tracks
        if tracks_L is not None:
            self.tracks_L = tracks_L
        if tracks_C is not None:
            self.tracks_C = tracks_C
        if tracks_R is not None:
            self.tracks_R = tracks_R
        if scale is not None:
            self.scale = scale
        if tracking_session_id is not None:
            self.tracking_session_id = tracking_session_id

    # ------------------------------------------------------------------ #
    def __call__(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.tri_prepare:
            self._handle_tri_click(x, y)
        else:
            self._handle_single_click(x, y)

    # ------------------------------------------------------------------ #
    def _handle_tri_click(self, x: int, y: int) -> None:
        state = self.tri_ui_state or {}
        orig_w = state.get("orig_w", 0)
        orig_h = state.get("orig_h", 0)
        disp_w = state.get("disp_w", 0)
        disp_h = state.get("disp_h", 0)
        offset_x = state.get("offset_x", 0)
        offset_y = state.get("offset_y", 0)
        seg_w = state.get("seg_w", 0)

        if orig_w <= 0 or disp_w <= 0 or seg_w <= 0:
            print("[Mouse] tri state invalid")
            return

        x_in = x - offset_x
        y_in = y - offset_y
        if not (0 <= x_in < disp_w and 0 <= y_in < disp_h):
            print("[Mouse] 여백 클릭")
            return

        x_tri = int(x_in * orig_w / disp_w)
        y_tri = int(y_in * orig_h / disp_h)

        seg_idx = min(max(x_tri // seg_w, 0), 2)
        clicked_seg = ("L", "C", "R")[seg_idx]
        x_local = x_tri - seg_idx * seg_w
        y_local = y_tri

        print(
            f"[Mouse] tri click: seg={clicked_seg}, "
            f"local=({x_local},{y_local}), tri=({x_tri},{y_tri})"
        )

        seg_tracks = {
            "L": self.tracks_L,
            "C": self.tracks_C,
            "R": self.tracks_R,
        }.get(clicked_seg, []) or []

        clicked_id = None
        for tid, x1, y1, x2, y2 in seg_tracks:
            if x1 <= x_local <= x2 and y1 <= y_local <= y2:
                clicked_id = tid
                print(f"[DIRECT_HIT] ✅ 현재 프레임에서 ID {tid} 직접 선택")
                break

        if clicked_id is None:
            print("[HISTORY_SEARCH] 현재 프레임에 트랙 없음, 히스토리 검색 시작...")
            history = self._history_for_segment(clicked_seg)
            clicked_id, _, _ = find_closest_track_in_history(
                history,
                x_local,
                y_local,
                max_frames_back=15,
                max_distance=150,
            )

        if clicked_id is None:
            self._clear_selection()
            print("[INFO] ❌ tri 모드 클릭 영역 내 차량 없음 (현재+히스토리)")
            return

        self.tracker.selected_id = clicked_id
        self.update_web_stats(selected_id=clicked_id)
        print(f"[INFO] 🎯 tri 모드 차량 선택됨: seg={clicked_seg}, ID={clicked_id}")

        history = self._history_for_segment(clicked_seg)
        cam_name = self._camera_name_for_segment(clicked_seg)
        self._log_selection(clicked_seg, cam_name, clicked_id)
        self._collect_history_crops(history, clicked_seg, cam_name, clicked_id)

    # ------------------------------------------------------------------ #
    def _handle_single_click(self, x: int, y: int) -> None:
        scale = self.scale if self.scale and self.scale > 0 else 1.0
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        print(
            f"[Mouse] 클릭: 표시=({x},{y}), "
            f"원본=({orig_x},{orig_y}), scale={scale:.3f}"
        )

        clicked_id = None
        for tid, x1, y1, x2, y2 in self.tracks or []:
            if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                clicked_id = tid
                print(
                    f"[DIRECT_HIT] ✅ ID {tid} 선택 "
                    f"(bbox={x1},{y1},{x2},{y2})"
                )
                break

        if clicked_id is None:
            print("[HISTORY_SEARCH] 현재 프레임에 트랙 없음, 히스토리 검색 시작...")
            clicked_id, _, _ = find_closest_track_in_history(
                self.track_history_center,
                orig_x,
                orig_y,
                max_frames_back=15,
                max_distance=150,
            )

        if clicked_id is None:
            self._clear_selection()
            print("[INFO] ❌ 클릭 영역 내 차량 없음 (현재+히스토리)")
            return

        self.tracker.selected_id = clicked_id
        if self.update_web_stats:
            self.update_web_stats(selected_id=clicked_id)
        print(
            f"[INFO] 🎯 차량 선택됨 (ID={clicked_id}) - "
            "다음 프레임부터 빨간 박스로 표시됩니다"
        )
        cam_name = getattr(self.switcher, "current_name", None)
        self._collect_history_crops(
            self.track_history_center, "C", cam_name, clicked_id
        )

    # ------------------------------------------------------------------ #
    def _history_for_segment(self, seg: str):
        if seg == "L":
            return self.track_history_left
        if seg == "R":
            return self.track_history_right
        return self.track_history_center

    def _camera_name_for_segment(self, seg: str) -> Optional[str]:
        if not self.switcher:
            return None
        if seg == "L":
            return getattr(self.switcher, "left_name", None)
        if seg == "R":
            return getattr(self.switcher, "right_name", None)
        return getattr(self.switcher, "current_name", None)

    def _collect_history_crops(
        self,
        history,
        segment: str,
        cam_name: Optional[str],
        track_id: Optional[int],
    ) -> None:
        if history is None or track_id is None:
            return

        if not hasattr(history, "get_history"):
            return

        history_frames = history.get_history(track_id)
        target = getattr(
            getattr(self.selected_bank, "items_band5", None), "maxlen", 0
        )

        print(
            f"[HISTORY] 📦 {len(history_frames)}개 프레임 발견, "
            f"{target}장 수집 시도..."
        )

        collected = 0
        skip_reasons = {"too_small": 0, "add_failed": 0, "quality": 0}

        for crop, bbox, frame_idx in reversed(history_frames):
            if collected >= target:
                break
            if crop is None or getattr(crop, "size", 0) == 0:
                skip_reasons["quality"] += 1
                continue

            h, w = crop.shape[:2]
            if h < 30 or w < 30:
                skip_reasons["too_small"] += 1
                continue

            if collected == 0:
                print(f"[HISTORY] crop 시도: size={w}x{h}, fidx={frame_idx}")

            try:
                added = self.selected_bank.add_from_frame_banded5_improved(
                    crop,
                    (0, 0, w, h),
                    pad=0,
                    center_ratio=1.0,
                    origin_seg=segment,
                    origin_cam=cam_name,
                    cam_id=cam_name,
                    use_whitening=True,
                )
                if added:
                    collected += 1
                    if collected <= 2:
                        print(
                            f"[HISTORY] ✅ {collected}/{target} 수집 "
                            f"(size={w}x{h})"
                        )
                else:
                    skip_reasons["add_failed"] += 1
            except Exception as exc:  # pragma: no cover - defensive
                skip_reasons["add_failed"] += 1
                if collected == 0:
                    print(f"[HISTORY] ⚠️  add 실패: {exc}")

        if collected > 0:
            print(f"[HISTORY] ✅ 최종 {collected}/{target}장 수집 완료")
        else:
            print(f"[HISTORY] ❌ 히스토리 수집 실패: {skip_reasons}")
            print("[HISTORY] → 실시간 수집으로 전환")

    def _log_selection(self, segment: str, cam_name: Optional[str], track_id: int):
        if not self.log_vehicle_tracking:
            return
        self.log_vehicle_tracking(
            session_id=self.tracking_session_id,
            event_type="INITIAL_SELECT",
            data={
                "camera": cam_name,
                "tid": track_id,
                "segment": segment,
            },
        )

    def _clear_selection(self) -> None:
        self.selected_bank.clear()
        self.tracker.selected_id = None
        if self.update_web_stats:
            self.update_web_stats(selected_id=None)

    def clear_selection(self) -> None:
        """External hook to clear current selection."""
        self._clear_selection()
