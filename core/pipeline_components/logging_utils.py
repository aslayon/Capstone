"""
Utility helpers for pipeline logging and auditing.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def log_vehicle_tracking(
    session_id: str,
    event_type: str,
    data: Dict[str, Any],
    root: str = "tracking_logs",
) -> None:
    """
    Persist structured vehicle tracking events.

    Args:
        session_id: tracking session identifier (usually selection timestamp)
        event_type: semantic event key (INITIAL_SELECT, MATCH_FOUND, etc.)
        data: event payload
        root: directory to store log files
    """
    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(root) / today
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"session_{session_id}_vehicle_track.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    if event_type == "INITIAL_SELECT":
        message = (
            "[추적 시작] 차량 선택 - "
            f"카메라: {data.get('camera', '?')}, "
            f"ID: {data.get('tid', '?')}, "
            f"세그먼트: {data.get('segment', '?')}"
        )
    elif event_type == "MATCH_FOUND":
        message = (
            "[동일 차량 발견] "
            f"{data.get('cam_from', '?')}(ID:{data.get('tid_from', '?')}) → "
            f"{data.get('cam_to', '?')}(ID:{data.get('tid_to', '?')}) | "
            f"유사도: {data.get('distance', 0):.3f} | "
            f"신뢰도: {data.get('confidence', 'UNKNOWN')}"
        )
    elif event_type == "CAMERA_SWITCH":
        message = (
            "[카메라 전환] "
            f"{data.get('from_cam', '?')} → {data.get('to_cam', '?')}"
        )
    else:
        message = f"[{event_type}] {data}"

    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} | {message}\n")
