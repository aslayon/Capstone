import math
import numpy as np
import cv2

def classify_directions(center_cctv, all_cctvs, top_k=8):
    def get_angle(dx, dy):
        angle = math.degrees(math.atan2(dy, dx))
        angle = (angle + 360) % 360  # 0~360도
        return angle

    direction_map = {
        ( -1, -1 ): (225, 315),  # NW
        (  0, -1 ): (315,  45),  # N
        (  1, -1 ): ( 45,  90),  # NE
        (  1,  0 ): ( 90, 135),  # E
        (  1,  1 ): (135, 180),  # SE
        (  0,  1 ): (180, 225),  # S
        ( -1,  1 ): (225, 270),  # SW
        ( -1,  0 ): (270, 315),  # W
    }

    result = {}
    cx, cy = center_cctv["coordx"], center_cctv["coordy"]
    distances = []

    for c in all_cctvs:
        if c["cctvname"] == center_cctv["cctvname"]:
            continue
        dx = c["coordx"] - cx
        dy = c["coordy"] - cy
        dist = math.sqrt(dx**2 + dy**2)
        angle = get_angle(dx, -dy)  # y축 반대

        distances.append((dist, angle, c["cctvname"]))

    # 가까운 top_k 개만 사용
    distances.sort()
    for dist, angle, name in distances[:top_k]:
        for (dx, dy), (a1, a2) in direction_map.items():
            # 북쪽(315~45도)은 예외처리
            if a1 > a2:
                if angle >= a1 or angle <= a2:
                    result[(dx, dy)] = name
                    break
            else:
                if a1 <= angle <= a2:
                    result[(dx, dy)] = name
                    break

    return result


def find_nearest_cctv_by_bbox(bbox, cctv_list):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2

    min_dist = float("inf")
    closest = None
    for c in cctv_list:
        dx = c["coordx"] - cx
        dy = c["coordy"] - cy
        dist = (dx**2 + dy**2)**0.5
        if dist < min_dist:
            min_dist = dist
            closest = c
    return closest





def compose_cctv_grid(direction_map, frame_dict):
    # 중심 해상도 기준으로 전체 그리드 크기 계산
    if "center" not in frame_dict:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    H, W = frame_dict["center"].shape[:2]
    grid = np.zeros((H * 3, W * 3, 3), dtype=np.uint8)

    for (dx, dy), name in direction_map.items():
        fx, fy = 1 + dx, 1 + dy
        if name in frame_dict:
            f = frame_dict[name]
            h, w = f.shape[:2]
            if h != H or w != W:
                f = cv2.resize(f, (W, H))
            grid[fy * H : (fy+1) * H, fx * W : (fx+1) * W] = f

    # 중심
    center_frame = frame_dict["center"]
    h, w = center_frame.shape[:2]
    if h != H or w != W:
        center_frame = cv2.resize(center_frame, (W, H))
    grid[H:2*H, W:2*W] = center_frame

    return grid

