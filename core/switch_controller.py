# core/switch_controller.py
# MP4 파일 지원 버전 (참고용)
# 
# 변경사항:
# 1. UniversalStreamManager import (기존 HLSStreamManager와 호환)
# 2. 소스 타입 자동 감지
# 3. 그 외 코드는 동일

import os, time
from typing import Optional, Dict
from core.cctv_graph import load_graph, load_cctv_list, get_neighbors, find_url_by_name
from core.config import load_config, save_current_cctv_url

# ✅ 기존 코드와 호환되는 import
# universal_stream_manager.py에서 HLSStreamManager = UniversalStreamManager로 정의됨
from core.stream_manager import HLSStreamManager

class SwitchController:
    """
    - 실행 중 .env 변경 감지 → 부드러운 전환
    - 키보드 'a','d' → 좌/우 이웃으로 전환
    - 3화면(2160x480)에서 클릭 x좌표로 전환 (좌/중/우)
    - ✅ HLS/MP4/RTSP 모두 지원 (자동 감지)
    """
    def __init__(self,
                 current_name: str,
                 current_url: str,
                 api_key: str,
                 graph_path: str = "config/cctv_graph_connections.json",
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

        self.tri_mode = False
        print(f"[SwitchController] cctv_list loaded: {len(self.cctv_list)} items")

        # 초기 이웃 계산
        self._update_neighbors()

    def attach_center_manager(self, sm: HLSStreamManager):
        """중앙 스트림 매니저 연결"""
        self.center_sm = sm
        if not getattr(sm, "running", False):
            sm.start(self.current_name, self.current_url)
        
        self._log_next_keys()
        
        try:
            self._last_url_updates = sm.stats.get("url_updates", 0)
        except:
            self._last_url_updates = 0

    def ensure_neighbor_managers(self):
        """좌/우 이웃 스트림 매니저 준비"""
        if self.left_name and self.left_sm is None:
            lu = self._find_url(self.left_name)
            if lu:
                # ✅ MP4도 자동 지원
                self.left_sm = HLSStreamManager(self.api_key, update_interval=20)
                self.left_sm.start(self.left_name, lu)
                
                # 소스 타입 출력 (디버깅용)
                source_type = self.left_sm.stats.get('source_type', 'unknown')
                print(f"[LEFT] {self.left_name}: {source_type}")
        
        if self.right_name and self.right_sm is None:
            ru = self._find_url(self.right_name)
            if ru:
                self.right_sm = HLSStreamManager(self.api_key, update_interval=20)
                self.right_sm.start(self.right_name, ru)
                
                source_type = self.right_sm.stats.get('source_type', 'unknown')
                print(f"[RIGHT] {self.right_name}: {source_type}")

    def stop_all(self):
        """모든 스트림 매니저 중지"""
        for sm in (self.left_sm, self.right_sm, self.center_sm):
            try:
                if sm: sm.stop()
            except: pass

    def tick(self):
        """주기적 업데이트 (환경 변경 감지 등)"""
        # .env 변경 감지
        if os.path.exists(self.env_path):
            mtime = os.path.getmtime(self.env_path)
            if mtime > self.env_mtime:
                self.env_mtime = mtime
                cfg = load_config()
                new_name = cfg.get("CURRENT_CCTV_NAME", "") or self.current_name
                new_url  = cfg.get("CURRENT_CCTV_URL", "")  or self.current_url
                if (new_name != self.current_name) or (new_url != self.current_url):
                    self._switch_to(new_name, new_url)
                    return

        # URL 갱신 감지 (HLS만 해당, MP4는 갱신 없음)
        if self.center_sm:
            try:
                cur = self.center_sm.stats.get("url_updates", 0)
                if cur != self._last_url_updates:
                    self._last_url_updates = cur
                    self._log_next_keys(prefix=f"[URL 업데이트 #{cur}] ")
            except:
                pass

    def _log_next_keys(self, prefix: str = ""):
        """다음 전환 가능한 카메라 안내"""
        left_name  = self.left_name  or "(없음)"
        right_name = self.right_name or "(없음)"

        left_url  = self._find_url(self.left_name)  if self.left_name  else None
        right_url = self._find_url(self.right_name) if self.right_name else None

        # ✅ 소스 타입 표시
        left_type = self._get_source_type(left_url) if left_url else ""
        right_type = self._get_source_type(right_url) if right_url else ""

        left_info  = f"{left_name} [{left_type}]" if left_type else left_name
        right_info = f"{right_name} [{right_type}]" if right_type else right_name

        print(f"{prefix}[A] → {left_info}  |  [D] → {right_info}")

    def _get_source_type(self, url: str) -> str:
        """URL로부터 소스 타입 간단 판별"""
        if not url:
            return ""
        
        url_lower = url.lower()
        
        if os.path.isfile(url):
            return "MP4"
        elif url_lower.endswith('.m3u8'):
            return "HLS"
        elif url_lower.startswith('rtsp://'):
            return "RTSP"
        elif url.isdigit():
            return "WEBCAM"
        else:
            return "STREAM"

    def on_key(self, key: int):
        """키보드 입력 처리"""
        if key == ord('a'):
            print(f"[Key] A → {self.left_name or '(없음)'}")
            self.switch_left()
        elif key == ord('d'):
            print(f"[Key] D → {self.right_name or '(없음)'}")
            self.switch_right()

    def on_triple_click(self, x: int, total_width: int):
        """3화면 클릭 처리"""
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

    def switch_left(self):
        """좌측 카메라로 전환"""
        if not self.left_name: return
        url = self._find_url(self.left_name)
        if url: self._switch_to(self.left_name, url)

    def switch_right(self):
        """우측 카메라로 전환"""
        if not self.right_name: return
        url = self._find_url(self.right_name)
        if url: self._switch_to(self.right_name, url)

    def _switch_to(self, new_name: str, new_url: str):
        """새 카메라/소스로 전환"""
        if not self.center_sm:
            print("[SwitchController] ❌ center_sm not attached")
            return

        # ✅ 소스 타입 표시
        source_type = self._get_source_type(new_url)
        print(f"[SwitchController] switching...")
        print(f"  from: {self.current_name}")
        print(f"  to  : {new_name} [{source_type}]")
        print(f"  url : {new_url}")
        
        # UniversalStreamManager가 자동으로 타입 감지하여 처리
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
        """이웃 카메라 정보 갱신"""
        neigh = get_neighbors(self.graph, self.current_name)
        self.left_name  = neigh.get("left")
        self.right_name = neigh.get("right")

    def _reset_neighbors_if_changed(self):
        """이웃이 변경되면 스트림 매니저 재설정"""
        if self.left_sm and (self.left_sm.cctv_name != self.left_name):
            try: self.left_sm.stop()
            except: pass
            self.left_sm = None
        
        if self.right_sm and (self.right_sm.cctv_name != self.right_name):
            try: self.right_sm.stop()
            except: pass
            self.right_sm = None

    def _find_url(self, name):
        """카메라 이름으로 URL 찾기"""
        if not name or not self.cctv_list:
            return None

        def norm(s: str) -> str:
            for ch in "[]()":
                s = s.replace(ch, "")
            return "".join(s.split())

        # 1) 정확 일치
        for it in self.cctv_list:
            if it.get("cctvname") == name:
                return it.get("cctvurl")

        # 2) 정규화 일치
        nname = norm(name)
        for it in self.cctv_list:
            if norm(it.get("cctvname","")) == nname:
                return it.get("cctvurl")

        # 3) 부분 포함
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
        except:
            pass
        
        return None


# ============================================================
# 사용 예시
# ============================================================
if __name__ == "__main__":
    from core.config import load_config
    
    cfg = load_config()
    
    # HLS 또는 MP4 자동 지원
    switcher = SwitchController(
        current_name=cfg["CURRENT_CCTV_NAME"],
        current_url=cfg["CURRENT_CCTV_URL"],
        api_key=cfg["ITS_API_KEY"],
        graph_path="config/cctv_graph_connections.json",
        list_path="data/cctv_list_4.json"
    )
    
    # 스트림 매니저 연결
    sm = HLSStreamManager(api_key=cfg["ITS_API_KEY"])
    switcher.attach_center_manager(sm)
    
    # 테스트
    print("\n" + "="*60)
    print("키보드 전환 테스트")
    print("="*60)
    print("A: 좌측으로 전환")
    print("D: 우측으로 전환")
    print("Q: 종료")
    print("="*60 + "\n")
    
    import cv2
    
    while True:
        switcher.tick()
        
        frame = switcher.center_sm.get_frame()
        if frame is not None:
            # 소스 타입 표시
            health = switcher.center_sm.get_stream_health()
            source_type = health.get('source_type', 'unknown')
            
            cv2.putText(frame, f"{switcher.current_name} [{source_type}]",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('a'), ord('d')]:
            switcher.on_key(key)
    
    switcher.stop_all()
    cv2.destroyAllWindows()
