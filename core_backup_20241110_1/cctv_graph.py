# cctv_graph.py
# 연결/이웃 + URL 조회
import json
from pathlib import Path
from core.config import CCTV_GRAPH_PATH

def load_graph(path: str | Path = CCTV_GRAPH_PATH):
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
    그래프에는 'north/south'로 들어있으니, 여기센 규칙적으로 left/right 로 맵핑.
    
    매핑 규칙:
    - north (포천 방향) → left (A 키, 역방향)
    - south (세종 방향) → right (D 키, 정방향)
    """
    left = right = None
    found = False
    
    for node in graph:
        if node["cctvname"] == current_name:
            found = True
            for c in node["connections"]:
                if c["direction"] == "south":
                    left = c["target"]   # ✅ 수정: north → left
                elif c["direction"] == "north":
                    right = c["target"]  # ✅ 수정: south → right
            break
    
    # ✅ 디버깅 출력 추가
    if not found:
        print(f"[get_neighbors] ⚠️  '{current_name}' NOT FOUND in graph")
        # 유사한 이름 찾기
        import difflib
        candidates = [n.get("cctvname", "") for n in graph]
        close = difflib.get_close_matches(current_name, candidates, n=3, cutoff=0.6)
        if close:
            print(f"[get_neighbors] 유사한 이름: {close}")
    else:
        if left or right:
            print(f"[get_neighbors] ✅ {current_name}")
            print(f"  ← (A/north) {left or '(없음)'}")
            print(f"  → (D/south) {right or '(없음)'}")
        else:
            print(f"[get_neighbors] ⚠️  '{current_name}' 에 연결된 이웃 없음")
    
    return {"left": left, "right": right}

def find_url_by_name(cctv_list, target_name: str):
    for item in cctv_list:
        if item.get("cctvname") == target_name:
            return item.get("cctvurl")
    return None