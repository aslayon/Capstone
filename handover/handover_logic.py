import json
from geopy.distance import geodesic

def load_cctv_list(path="cctv_list.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_adjacent_cctvs(current_x, current_y, cctv_list, direction="right", radius_km=2.0):
    candidates = []
    for cctv in cctv_list:
        x, y = cctv["coordx"], cctv["coordy"]
        if direction == "right" and x <= current_x:
            continue
        if direction == "left" and x >= current_x:
            continue
        dist = geodesic((current_y, current_x), (y, x)).km
        if dist <= radius_km:
            candidates.append((dist, cctv))
    candidates.sort()
    return [c[1] for c in candidates]
