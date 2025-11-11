
from typing import Optional, Tuple

_web_state = None

def setup_web_integration(state_obj):
    """
    웹 통합 초기화
    app.py의 PipelineState 객체를 전달받아 설정
    """
    global _web_state
    _web_state = state_obj
    print("[WEB_INTEGRATION] ✅ 웹 통합 활성화")


def get_web_key(timeout=0.001) -> Optional[int]:
    """
    웹에서 전송된 키 입력 가져오기
    cv2.waitKey()를 대체
    
    Returns:
        int: 키 코드 (cv2.waitKey와 동일한 형식)
        None: 키 입력 없음
    """
    import cv2
    # 항상 cv2 이벤트 루프를 유지해 창이 응답하도록 함
    local_key = cv2.waitKey(1) & 0xFF
    if local_key not in (-1, 255):
        return local_key
    
    if _web_state is None:
        return -1  # 웹 통합 미사용
    
    try:
        # 웹에서 키 가져오기 (non-blocking)
        key_str = _web_state.get_key(block=False)
        
        if key_str is None:
            return -1  # 키 없음
        
        # 특수 키 처리
        if key_str == 'clear':
            return ord('c')  # clear를 'c' 키로 변환
        
        # 문자를 ASCII 코드로 변환
        if len(key_str) == 1:
            return ord(key_str)
        
        return -1
        
    except Exception as e:
        print(f"[WEB_KEY] 에러: {e}")
        return -1


def get_web_click() -> Optional[Tuple[int, int]]:
    """
    웹에서 전송된 클릭 좌표 가져오기
    
    Returns:
        (x, y): 클릭 좌표
        None: 클릭 없음
    """
    if _web_state is None:
        return None
    
    try:
        click = _web_state.get_click(block=False)
        return click
    except Exception as e:
        print(f"[WEB_CLICK] 에러: {e}")
        return None


def update_web_stats(**kwargs):
    """
    웹에 통계 업데이트
    
    예:
        update_web_stats(fps=30.5, total_tracks=5, selected_id=12, mode='tri')
    """
    if _web_state is None:
        return
    
    try:
        _web_state.update_stats(**kwargs)
    except Exception as e:
        print(f"[WEB_STATS] 업데이트 실패: {e}")
