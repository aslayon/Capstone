# core/tri_detect.py
# 카메라 3대 영상을 합친 tri 이미지(2160xH)에서
# YOLO 검출 결과를 좌/중/우로 분배하는 유틸
from typing import List, Tuple, Dict

BBox = Tuple[float, float, float, float, float, int]  # x1,y1,x2,y2,conf,cls

def split_dets_by_segment(dets: List[BBox], seg_w: int = 720) -> Dict[str, List[BBox]]:
    """
    tri 이미지(2160xH)에 대한 YOLO 결과 dets를 좌/중/우로 분배.
    반환 좌표는 '각 세그먼트 로컬 좌표'로 변환됨.
    """
    out = {"L": [], "C": [], "R": []}
    for x1, y1, x2, y2, conf, cls in dets:
        cx = 0.5 * (x1 + x2)
        if cx < seg_w:
            # Left: 로컬 좌표 => 그대로
            out["L"].append((x1, y1, x2, y2, conf, cls))
        elif cx < seg_w * 2:
            # Center: offset 제거
            out["C"].append((x1 - seg_w, y1, x2 - seg_w, y2, conf, cls))
        else:
            # Right
            out["R"].append((x1 - seg_w * 2, y1, x2 - seg_w * 2, y2, conf, cls))
    return out
