# core/switch_controller.py
# - 실행 중 .env 변경 감지 → 부드러운 전환  -> 카메라 전환 관련
# - 키보드 'a','d' → 좌/우 이웃으로 전환


import os, time
from typing import Optional, Dict
from core.cctv_graph import load_graph, load_cctv_list, get_neighbors, find_url_by_name
from core.config import load_config, save_current_cctv_url
from core.stream_manager import HLSStreamManager

class SwitchController:
    """
    - 실행 중 .env 변경 감지 → 부드러운 전환
    - 키보드 'a','d' → 좌/우 이웃으로 전환
    - 3화면(2160x480)에서 클릭 x좌표로 전환 (좌/중/우)
    - HLSStreamManager 기반으로 끊김 최소화
    """
    def __init__(self,
                 current_name: str,
                 current_url: str,
                 api_key: str,
                 graph_path: str = "data/cctv_graph_connections.json",
                 list_path: str  = "data/cctv_list_4.json",
                 env_path: str   = ".env"):
        self.env_path = env_path
        self.env_mtime = os.path.getmtime(env_path) if os.path.exists(env_path) else 0
        self._last_url_updates = 0
        self.api_key = api_key
        self.graph = load_graph(graph_path)
        self.cctv_list = load_cctv_list(list_path)

        # 현재 중심(메인) CCTV 상태
        self.current_name = current_name
        self.current_url  = current_url
        self.center_sm: Optional[HLSStreamManager] = None

        # 이웃 CCTV 상태 (tri-mode 대비)
        self.left_name: Optional[str] = None
        self.right_name: Optional[str] = None
        self.left_sm: Optional[HLSStreamManager] = None
        self.right_sm: Optional[HLSStreamManager] = None

        self.tri_mode = False  # 3화면 모드 여부 (파이프라인 쪽에서 켜고 끔)
        print(f"[SwitchController] cctv_list loaded: {len(self.cctv_list)} items (path={list_path})")      # 디버그

        # 초기 이웃 계산
        self._update_neighbors()

    # ---------- attach / start / stop ----------
    # [수정] attach_center_manager
    def attach_center_manager(self, sm: HLSStreamManager):
        self.center_sm = sm
        if not getattr(sm, "running", False):
            sm.start(self.current_name, self.current_url)
        # 초기 한 번, 현재 좌/우 타깃 안내
        self._log_next_keys()
        # url_updates 초기화
        try:
            self._last_url_updates = sm.stats.get("url_updates", 0)
        except:
            self._last_url_updates = 0


    def ensure_neighbor_managers(self):
        """tri-prepare용: 좌/우 이웃 스트림 매니저 준비/시작."""
        if self.left_name and self.left_sm is None:
            lu = self._find_url(self.left_name)
            if lu:
                self.left_sm = HLSStreamManager(self.api_key, update_interval=20)
                self.left_sm.start(self.left_name, lu)
        if self.right_name and self.right_sm is None:
            ru = self._find_url(self.right_name)
            if ru:
                self.right_sm = HLSStreamManager(self.api_key, update_interval=20)
                self.right_sm.start(self.right_name, ru)

    def stop_all(self):
        for sm in (self.left_sm, self.right_sm, self.center_sm):
            try:
                if sm: sm.stop()
            except: pass

    # ---------- periodic tick ----------
    # [수정] tick(): 실행 중 .env 변경 + URL 갱신 감지시에도 안내
    def tick(self):
        # 1) .env 변경 감지 → 필요시 스위칭 (기존 그대로)
        if os.path.exists(self.env_path):
            mtime = os.path.getmtime(self.env_path)
            if mtime > self.env_mtime:
                self.env_mtime = mtime
                cfg = load_config()
                new_name = cfg.get("CURRENT_CCTV_NAME", "") or self.current_name
                new_url  = cfg.get("CURRENT_CCTV_URL", "")  or self.current_url
                if (new_name != self.current_name) or (new_url != self.current_url):
                    self._switch_to(new_name, new_url)
                    # _switch_to 안에서 이웃 갱신 + 안내가 이미 출력됨
                    return

        # 2) HLSStreamManager 내부 URL 갱신 감지 → 안내
        if self.center_sm:
            try:
                cur = self.center_sm.stats.get("url_updates", 0)
                if cur != self._last_url_updates:
                    self._last_url_updates = cur
                    # 같은 지점 내 URL만 바뀐 경우라도, 사용자 안내 재출력
                    self._log_next_keys(prefix=f"[URL 업데이트 #{cur}] ")
            except:
                pass
 
        # [추가] 공통 안내 함수
    def _log_next_keys(self, prefix: str = ""):     # 디버그
        left_name  = self.left_name  or "(없음)"
        right_name = self.right_name or "(없음)"

        # URL 조회
        left_url  = self._find_url(self.left_name)  if self.left_name  else None
        right_url = self._find_url(self.right_name) if self.right_name else None

        left_info  = f"{left_name} ({left_url})"   if left_url  else left_name
        right_info = f"{right_name} ({right_url})" if right_url else right_name

        print(f"{prefix}[A] 누르면 → {left_name}  |  [D] 누르면 → {right_name}")

        


    # ---------- key / click ----------
    # [수정] on_key: 누르는 순간 무엇으로 갈지 한 번 더 보여주기
    def on_key(self, key: int):
        if key == ord('a'):
            print(f"[Key] A → {self.left_name or '(없음)'}")
            self.switch_left()
        elif key == ord('d'):
            print(f"[Key] D → {self.right_name or '(없음)'}")
            self.switch_right()


    def on_triple_click(self, x: int, total_width: int):
        """
        3화면(2160x480 등)에서의 사용자 클릭 x좌표로 전환.
        좌/중/우 중 클릭된 영역으로 스위칭.
        """
        if total_width <= 0: return
        seg = total_width // 3
        if x < seg:
            # 좌
            if self.left_name:
                url = self._find_url(self.left_name)
                if url: self._switch_to(self.left_name, url)
        elif x < seg*2:
            # 중 (그대로)
            pass
        else:
            # 우
            if self.right_name:
                url = self._find_url(self.right_name)
                if url: self._switch_to(self.right_name, url)

    # ---------- explicit switches ----------
    def switch_left(self):
        if not self.left_name: return
        url = self._find_url(self.left_name)
        if url: self._switch_to(self.left_name, url)

    def switch_right(self):
        if not self.right_name: return
        url = self._find_url(self.right_name)
        if url: self._switch_to(self.right_name, url)

    # ---------- internals ----------
    def _switch_to(self, new_name: str, new_url: str):
        if not self.center_sm:
            print("[SwitchController] ❌ center_sm not attached")
            return

        print(f"[SwitchController] switching...\n  from: {self.current_name}\n  to  : {new_name}\n  url : {new_url}")
        if not self.center_sm.switch_to(new_name, new_url):
            print("[SwitchController] ⚠️ switch_to failed; keep current")
            return

        self.current_name = new_name
        self.current_url  = new_url
        try:
            from core.config import save_current_cctv_url, save_current_cctv
            
            save_current_cctv(new_name, new_url)
        except:
            pass

        self._update_neighbors()
        if self.tri_mode:
            self._reset_neighbors_if_changed()
            self.ensure_neighbor_managers()

        print(f"[SwitchController] ✅ switched to: {self.current_name}")
        self._log_next_keys()




    def _update_neighbors(self):
        neigh = get_neighbors(self.graph, self.current_name)
        self.left_name  = neigh.get("left")
        self.right_name = neigh.get("right")

    def _reset_neighbors_if_changed(self):
        # left
        if self.left_sm and (self.left_sm.cctv_name != self.left_name):
            try: self.left_sm.stop()
            except: pass
            self.left_sm = None
        # right
        if self.right_sm and (self.right_sm.cctv_name != self.right_name):
            try: self.right_sm.stop()
            except: pass
            self.right_sm = None

    def _find_url(self, name):
        if not name or not self.cctv_list:
            print(f"[find_url] list empty or name missing (path={getattr(self,'list_path','?')})")
            return None

        def norm(s: str) -> str:
            for ch in "[]()":
                s = s.replace(ch, "")
            return "".join(s.split())  # 공백 제거

        # 1) 정확 일치
        for it in self.cctv_list:
            if it.get("cctvname") == name:
                return it.get("cctvurl")

        # 2) 정규화 일치
        nname = norm(name)
        for it in self.cctv_list:
            if norm(it.get("cctvname","")) == nname:
                return it.get("cctvurl")

        # 3) 부분 포함 (양방향)
        for it in self.cctv_list:
            nm = it.get("cctvname","")
            if name in nm or nm in name:
                return it.get("cctvurl")

        # 디버깅: 가까운 후보 보여주기
        try:
            import difflib
            cand = [it.get("cctvname","") for it in self.cctv_list]
            close = difflib.get_close_matches(name, cand, n=5, cutoff=0.4)
            print(f"[find_url] '{name}' not found. close candidates: {close[:5]}")
        except Exception:
            pass
        return None


