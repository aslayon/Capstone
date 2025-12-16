"""
Matching utilities extracted from the monolithic pipeline.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from core.history import TrackHistory


def evaluate_match_confidence(distance: float) -> str:
    """
    Convert Bhattacharyya distance to a qualitative confidence label.
    """
    if distance < 0.60:
        return "HIGH"
    if distance < 0.70:
        return "MEDIUM"
    if distance < 0.80:
        return "LOW"
    return "REJECT"


class ConsecutiveMatchValidator:
    """
    Ensure a track meets a minimum number of consecutive confirmations before
    being treated as a valid match.
    """

    def __init__(self, required_count: int = 3) -> None:
        self.required_count = required_count
        self.match_history: Dict[int, int] = {}
        self.last_frame_idx: Dict[int, int] = {}

    def validate(
        self, track_id: int, distance: float, threshold: float, frame_idx: int
    ) -> Tuple[bool, str, int]:
        if distance > threshold:
            self.match_history[track_id] = 0
            return False, "REJECT", 0

        last_frame = self.last_frame_idx.get(track_id, -999)
        if frame_idx - last_frame > 5:
            count = 1
        else:
            count = self.match_history.get(track_id, 0) + 1

        self.match_history[track_id] = count
        self.last_frame_idx[track_id] = frame_idx

        if count >= self.required_count:
            return True, "CONFIRMED", count
        return False, "PENDING", count

    def reset(self) -> None:
        self.match_history.clear()
        self.last_frame_idx.clear()

    def cleanup(self, current_frame_idx: int, max_age: int = 30) -> None:
        to_remove = [
            tid
            for tid, last_frame in self.last_frame_idx.items()
            if current_frame_idx - last_frame > max_age
        ]
        for tid in to_remove:
            self.match_history.pop(tid, None)
            self.last_frame_idx.pop(tid, None)


def find_closest_track_in_history(
    track_history: TrackHistory,
    click_x: int,
    click_y: int,
    max_frames_back: int = 15,
    max_distance: int = 150,
) -> Tuple[Optional[int], Optional[float], int]:
    """
    Search the recent history buffer for the track whose centroid is closest
    to the requested click coordinate.
    """
    if not getattr(track_history, "history", None):
        return None, None, 0

    best_tid: Optional[int] = None
    best_distance = float(max_distance)
    best_frame_idx = -1
    total_checked = 0

    for tid, frames in track_history.history.items():
        if not frames:
            continue

        recent_frames = list(frames)[-min(len(frames), max_frames_back):]
        for crop, bbox, frame_idx in recent_frames:
            total_checked += 1
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            distance = math.hypot(cx - click_x, cy - click_y)

            if distance < best_distance:
                best_distance = distance
                best_tid = tid
                best_frame_idx = frame_idx

    if best_tid is None:
        print(
            f"[HISTORY_MATCH] ❌ {total_checked}개 프레임 검색했으나 "
            f"{max_distance}px 이내 트랙 없음"
        )
        return None, None, total_checked

    print(
        "[HISTORY_MATCH] ✅ "
        f"{total_checked}개 프레임 검색 → ID {best_tid} 발견 "
        f"(거리: {best_distance:.1f}px, 프레임: {best_frame_idx})"
    )
    return best_tid, best_distance, total_checked
