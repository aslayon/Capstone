#!/usr/bin/env python3
# test_graph_connections.py
# 그래프 연결 및 현재 CCTV 확인

import json
import sys
from pathlib import Path

def load_graph(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_neighbors(graph, current_name: str):
    """개선된 get_neighbors"""
    left = right = None
    found = False
    
    for node in graph:
        if node["cctvname"] == current_name:
            found = True
            for c in node["connections"]:
                if c["direction"] == "north":
                    left = c["target"]
                elif c["direction"] == "south":
                    right = c["target"]
            break
    
    if not found:
        print(f"❌ '{current_name}' NOT FOUND in graph")
        # 유사한 이름 찾기
        import difflib
        candidates = [n.get("cctvname", "") for n in graph]
        close = difflib.get_close_matches(current_name, candidates, n=5, cutoff=0.5)
        if close:
            print(f"\n유사한 이름들:")
            for i, c in enumerate(close, 1):
                print(f"  {i}. {c}")
    else:
        print(f"✅ '{current_name}' FOUND")
        print(f"  ← (A/north): {left or '(없음)'}")
        print(f"  → (D/south): {right or '(없음)'}")
    
    return {"left": left, "right": right, "found": found}

def main():
    # 그래프 로드
    graph_path = "config/cctv_graph_connections.json"
    if not Path(graph_path).exists():
        print(f"❌ {graph_path} 파일이 없습니다")
        return
    
    graph = load_graph(graph_path)
    print(f"📊 그래프 노드 수: {len(graph)}")
    print(f"📋 그래프 내 CCTV 목록:")
    for i, node in enumerate(graph, 1):
        print(f"  {i}. {node['cctvname']}") 
    print()
    
    # .env에서 현재 CCTV 이름 확인
    env_path = ".env"
    current_name = None
    
    if Path(env_path).exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("CURRENT_CCTV_NAME="):
                    current_name = line.split("=", 1)[1].strip().strip('"')
                    break
    
    if current_name:
        print(f"🎯 .env의 CURRENT_CCTV_NAME: {current_name}")
        print("="*70)
        result = get_neighbors(graph, current_name)
    else:
        print("⚠️  .env 파일에서 CURRENT_CCTV_NAME을 찾을 수 없습니다")
        print("\n테스트: 첫 번째 노드로 테스트")
        print("="*70)
        test_name = graph[0]["cctvname"]
        result = get_neighbors(graph, test_name)

if __name__ == "__main__":
    main()