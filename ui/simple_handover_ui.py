import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from enum import Enum

class UIMode(Enum):
    """UI 표시 모드 (간소화)"""
    SINGLE = "single"       # 단일 카메라 모드
    DUAL = "dual"          # 핸드오버용 듀얼 모드

class SimpleHandoverUI:
    """간소화된 핸드오버 UI 시스템"""
    
    def __init__(self):
    # 720x480 기준으로 변경
        self.single_width = 720
        self.single_height = 480
        self.dual_width = 1440  # 720 * 2
        self.dual_height = 480
        
        # 표시용 크기 (2배 확대)
        self.display_single_width = 1440
        self.display_single_height = 960
        self.display_dual_width = 1440
        self.display_dual_height = 480
        
        # 나머지는 그대로...
        self.current_mode = UIMode.SINGLE
        self.primary_camera = None
        self.secondary_camera = None
    
    def set_single_mode(self, camera_name: str):
        """단일 카메라 모드로 전환"""
        self.current_mode = UIMode.SINGLE
        self.primary_camera = camera_name
        self.secondary_camera = None
        self.handover_active = False
        print(f"단일 모드: {camera_name}")
    
    def set_dual_mode(self, primary_camera: str, secondary_camera: str):
        """듀얼 카메라 모드로 전환 (핸드오버용)"""
        self.current_mode = UIMode.DUAL
        self.primary_camera = primary_camera
        self.secondary_camera = secondary_camera
        self.handover_active = True
        print(f"듀얼 모드: {primary_camera} + {secondary_camera}")
    
    def create_display_frame(self, frame_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        현재 모드에 따라 표시할 프레임 생성
        
        Args:
            frame_dict: {"카메라명": 프레임} 딕셔너리
            
        Returns:
            표시용 프레임, 변환 정보
        """
        if self.current_mode == UIMode.SINGLE:
            return self._create_single_frame(frame_dict)
        else:  # DUAL
            return self._create_dual_frame(frame_dict)
    
    def _create_single_frame(self, frame_dict: Dict) -> Tuple[np.ndarray, Dict]:
        """단일 카메라 프레임 생성"""
        # 검은 배경
        display_frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        if self.primary_camera and self.primary_camera in frame_dict:
            # 프레임을 전체 화면에 맞춤
            frame = frame_dict[self.primary_camera]
            display_frame = cv2.resize(frame, (self.display_width, self.display_height))
            
            # 카메라 이름 표시
            self._draw_text(display_frame, self.primary_camera, (20, 50), 
                          color=(0, 255, 0), scale=1.2)
        else:
            # 카메라 없음
            self._draw_text(display_frame, "NO CAMERA", (self.display_width//2-100, self.display_height//2),
                          color=(0, 0, 255), scale=2.0)
        
        transform_info = {
            "mode": "single",
            "camera": self.primary_camera
        }
        
        return display_frame, transform_info
    
    def _create_dual_frame(self, frame_dict: Dict) -> Tuple[np.ndarray, Dict]:
        """듀얼 카메라 프레임 생성 (좌우 분할)"""
        half_width = self.display_width // 2
        display_frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # 왼쪽: Primary 카메라
        if self.primary_camera and self.primary_camera in frame_dict:
            primary_frame = frame_dict[self.primary_camera]
            primary_resized = cv2.resize(primary_frame, (half_width, self.display_height))
            display_frame[:, :half_width] = primary_resized
            
            # Primary 라벨
            self._draw_text(display_frame, f"현재: {self.primary_camera}", (20, 50),
                          color=(0, 255, 0), scale=0.8)
        
        # 오른쪽: Secondary 카메라
        if self.secondary_camera and self.secondary_camera in frame_dict:
            secondary_frame = frame_dict[self.secondary_camera]
            secondary_resized = cv2.resize(secondary_frame, (half_width, self.display_height))
            display_frame[:, half_width:] = secondary_resized
            
            # Secondary 라벨
            self._draw_text(display_frame, f"다음: {self.secondary_camera}", 
                          (half_width + 20, 50), color=(255, 255, 0), scale=0.8)
        
        # 가운데 구분선
        cv2.line(display_frame, (half_width, 0), (half_width, self.display_height), 
                (255, 255, 255), 4)
        
        # 핸드오버 상태 표시
        if self.handover_active:
            self._draw_handover_status(display_frame)
        
        transform_info = {
            "mode": "dual",
            "primary_camera": self.primary_camera,
            "secondary_camera": self.secondary_camera,
            "split_x": half_width
        }
        
        return display_frame, transform_info
    
    def _draw_handover_status(self, frame: np.ndarray):
        """핸드오버 상태 그리기"""
        # 상태 메시지 (화면 상단 중앙)
        message = self.handover_message or "차량 매칭 중..."
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (self.display_width - text_size[0]) // 2
        
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (text_x - 20, 80), (text_x + text_size[0] + 20, 130), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 텍스트
        self._draw_text(frame, message, (text_x, 115), color=(0, 255, 255), scale=1.0)
        
        # 진행 바 (화면 상단 중앙)
        bar_width = 400
        bar_height = 15
        bar_x = (self.display_width - bar_width) // 2
        bar_y = 140
        
        # 진행 바 배경
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (60, 60, 60), -1)
        
        # 진행률
        if self.handover_progress > 0:
            progress_width = int(bar_width * min(self.handover_progress, 1.0))
            color = (0, 255, 0) if self.handover_progress < 0.8 else (0, 165, 255)
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + progress_width, bar_y + bar_height), color, -1)
        
        # 테두리
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (255, 255, 255), 2)
        
        # 시간 표시
        time_text = f"{self.handover_time:.1f}초"
        self._draw_text(frame, time_text, (bar_x + bar_width + 20, bar_y + 12),
                       color=(255, 255, 255), scale=0.6)
    
    def _draw_text(self, frame: np.ndarray, text: str, position: Tuple[int, int],
                  color: Tuple[int, int, int] = (255, 255, 255), scale: float = 1.0):
        """텍스트 그리기 (그림자 효과 포함)"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(scale * 2))
        
        # 그림자 (약간 오프셋)
        shadow_pos = (position[0] + 2, position[1] + 2)
        cv2.putText(frame, text, shadow_pos, font, scale, (0, 0, 0), thickness)
        
        # 실제 텍스트
        cv2.putText(frame, text, position, font, scale, color, thickness)
    
    def update_handover_status(self, progress: float, message: str = "", elapsed_time: float = 0.0):
        """핸드오버 상태 업데이트"""
        self.handover_progress = progress
        self.handover_message = message
        self.handover_time = elapsed_time
    
    def handle_click(self, click_x: int, click_y: int) -> Dict:
        """클릭 이벤트 처리"""
        if self.current_mode == UIMode.SINGLE:
            return {
                "success": True,
                "mode": "single",
                "camera": self.primary_camera,
                "click_coords": (click_x, click_y),
                "relative_coords": (click_x / self.display_width, click_y / self.display_height)
            }
        else:  # DUAL
            half_width = self.display_width // 2
            
            if click_x < half_width:
                # 왼쪽 (Primary) 클릭
                return {
                    "success": True,
                    "mode": "dual",
                    "region": "primary",
                    "camera": self.primary_camera,
                    "click_coords": (click_x, click_y),
                    "local_coords": (click_x, click_y),
                    "relative_coords": (click_x / half_width, click_y / self.display_height)
                }
            else:
                # 오른쪽 (Secondary) 클릭
                local_x = click_x - half_width
                return {
                    "success": True,
                    "mode": "dual",
                    "region": "secondary", 
                    "camera": self.secondary_camera,
                    "click_coords": (click_x, click_y),
                    "local_coords": (local_x, click_y),
                    "relative_coords": (local_x / half_width, click_y / self.display_height)
                }
    
    def get_current_cameras(self) -> List[str]:
        """현재 표시 중인 카메라 목록"""
        cameras = []
        if self.primary_camera:
            cameras.append(self.primary_camera)
        if self.secondary_camera:
            cameras.append(self.secondary_camera)
        return cameras

# 간단한 사용 예시
if __name__ == "__main__":
    # UI 초기화
    ui = SimpleHandoverUI()
    
    # 테스트 프레임들
    def create_test_frame(width=640, height=480, color=(100, 100, 100), text=""):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        if text:
            cv2.putText(frame, text, (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                       2, (255, 255, 255), 3)
        return frame
    
    frames = {
        "[남해선] 죽평": create_test_frame(color=(50, 100, 50), text="CAMERA 1"),
        "[남해선] 선평교": create_test_frame(color=(100, 50, 50), text="CAMERA 2")
    }
    
    # 1. 단일 모드 테스트
    print("=== 단일 모드 ===")
    ui.set_single_mode("[남해선] 죽평")
    single_frame, single_info = ui.create_display_frame(frames)
    
    click_result = ui.handle_click(960, 540)
    print("클릭 결과:", click_result)
    
    # 2. 듀얼 모드 테스트
    print("\n=== 듀얼 모드 ===")
    ui.set_dual_mode("[남해선] 죽평", "[남해선] 선평교")
    ui.update_handover_status(0.6, "매칭 중...", 3.2)
    
    dual_frame, dual_info = ui.create_display_frame(frames)
    
    # 왼쪽 클릭
    left_click = ui.handle_click(480, 540)
    print("왼쪽 클릭:", left_click)
    
    # 오른쪽 클릭  
    right_click = ui.handle_click(1440, 540)
    print("오른쪽 클릭:", right_click)
    
    # 결과 저장
    cv2.imwrite("simple_single.jpg", single_frame)
    cv2.imwrite("simple_dual.jpg", dual_frame)
    print("\n테스트 이미지 저장 완료!")
    print("현재 카메라들:", ui.get_current_cameras())