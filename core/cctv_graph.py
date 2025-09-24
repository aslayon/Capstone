# cctv_graph.py
# 연결/이웃 + URL 조회
import json
from pathlib import Path

def load_graph(path="cctv_graph_connections.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_cctv_list(path="cctv_list_4.json"):
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))

def get_neighbors(graph, current_name: str):
    """
    그래프에서 current_name 의 이웃 2개(북/남 or 좌/우)를 반환.
    반환: {"left": name or None, "right": name or None} 형태로 맞춰두자.
    그래프에는 'north/south'로 들어있으니, 여기선 규칙적으로 left/right 로 맵핑.
    """
    left = right = None
    for node in graph:
        if node["cctvname"] == current_name:
            for c in node["connections"]:
                if c["direction"] == "north":
                    right = c["target"]
                elif c["direction"] == "south":
                    left = c["target"]
            break
    return {"left": left, "right": right}

def find_url_by_name(cctv_list, target_name: str):
    for item in cctv_list:
        if item.get("cctvname") == target_name:
            return item.get("cctvurl")
    return None
