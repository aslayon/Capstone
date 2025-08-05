import numpy as np
from typing import Tuple, Dict, Optional, List

class CoordinateTransformer:
    """
    다단계 좌표 변환을 처리하는 클래스
    사용자 클릭 좌표 → 표시 좌표 → YOLO 좌표 → 원본 카메라 좌표
    """
    
    def __init__(self, display_width=1920, display_height=1080, 
                 yolo_width=640, yolo_height=640):
        """
        Args:
            display_width: UI 표시 화면 너비
            display_height: UI 표시 화면 높이  
            yolo_width: YOLO 처리용 프레임 너비
            yolo_height: YOLO 처리용 프레임 높이
        """
        self.display_width = display_width
        self.display_height = display_height
        self.yolo_width = yolo_width
        self.yolo_height = yolo_height
        
        # 듀얼 모드시 각 영역 크기
        self.dual_display_width = display_width // 2
        self.dual_yolo_width = yolo_width // 2
    
    def click_to_original_coords(self, click_x: float, click_y: float, 
                               ui_mode: str, camera_info: Dict, 
                               region_info: Dict = None) -> Optional[Dict]:
        """
        사용자 클릭 좌표를 원본 카메라 좌표로 변환
        
        Args:
            click_x, click_y: 사용자 클릭 좌표 (UI 화면 기준)
            ui_mode: "single" 또는 "dual"
            camera_info: 카메라 정보 (원본 해상도 등)
            region_info: 듀얼 모드시 영역 변환 정보
            
        Returns:
            변환된 좌표 정보 또는 None
        """
        if ui_mode == "single":
            return self._single_mode_transform(click_x, click_y, camera_info)
        elif ui_mode == "dual":
            return self._dual_mode_transform(click_x, click_y, camera_info, region_info)
        else:
            return None
    
    def _single_mode_transform(self, click_x: float, click_y: float, 
                             camera_info: Dict) -> Dict:
        """단일 모드에서의 좌표 변환"""
        # 1단계: UI 클릭 좌표 → YOLO 좌표
        yolo_x = (click_x / self.display_width) * self.yolo_width
        yolo_y = (click_y / self.display_height) * self.yolo_height
        
        # 2단계: YOLO 좌표 → 원본 카메라 좌표
        original_coords = self._yolo_to_original_coords(
            yolo_x, yolo_y, camera_info.get("transform", {})
        )
        
        return {
            "camera": camera_info.get("name", "unknown"),
            "click_coords": (click_x, click_y),
            "yolo_coords": (yolo_x, yolo_y),
            "original_coords": original_coords,
            "mode": "single"
        }
    
    def _dual_mode_transform(self, click_x: float, click_y: float, 
                           camera_info: Dict, region_info: Dict) -> Optional[Dict]:
        """듀얼 모드에서의 좌표 변환"""
        # 어느 영역(primary/secondary)을 클릭했는지 판단
        if click_x < self.dual_display_width:
            # Primary 영역 클릭
            region_type = "primary"
            local_click_x = click_x
            target_camera = camera_info.get("primary_camera", "unknown")
            transform_info = region_info.get("primary_region", {}).get("transform", {})
        else:
            # Secondary 영역 클릭
            region_type = "secondary"
            local_click_x = click_x - self.dual_display_width
            target_camera = camera_info.get("secondary_camera", "unknown")
            transform_info = region_info.get("secondary_region", {}).get("transform", {})
        
        # 영역 내 상대 좌표로 변환
        local_click_y = click_y
        
        # 1단계: 영역 내 클릭 좌표 → 영역 내 YOLO 좌표
        region_yolo_x = (local_click_x / self.dual_display_width) * self.dual_yolo_width
        region_yolo_y = (local_click_y / self.display_height) * self.yolo_height
        
        # 2단계: 영역 내 YOLO 좌표 → 원본 카메라 좌표
        original_coords = self._yolo_to_original_coords(
            region_yolo_x, region_yolo_y, transform_info
        )
        
        # 결합된 프레임에서의 전체 YOLO 좌표도 계산
        if region_type == "primary":
            full_yolo_x = region_yolo_x
        else:
            full_yolo_x = region_yolo_x + self.dual_yolo_width
        full_yolo_y = region_yolo_y
        
        return {
            "camera": target_camera,
            "region": region_type,
            "click_coords": (click_x, click_y),
            "local_click_coords": (local_click_x, local_click_y),
            "region_yolo_coords": (region_yolo_x, region_yolo_y),
            "full_yolo_coords": (full_yolo_x, full_yolo_y),
            "original_coords": original_coords,
            "mode": "dual"
        }
    
    def _yolo_to_original_coords(self, yolo_x: float, yolo_y: float, 
                               transform_info: Dict) -> Optional[Tuple[float, float]]:
        """YOLO 좌표를 원본 카메라 좌표로 변환"""
        if not transform_info:
            return None
        
        scale = transform_info.get("scale", 1.0)
        x_offset = transform_info.get("x_offset", 0)
        y_offset = transform_info.get("y_offset", 0)
        original_size = transform_info.get("original_size", (0, 0))
        
        if scale <= 0 or original_size[0] <= 0 or original_size[1] <= 0:
            return None
        
        # 패딩 제거
        unpadded_x = yolo_x - x_offset
        unpadded_y = yolo_y - y_offset
        
        # 스케일 해제
        original_x = unpadded_x / scale
        original_y = unpadded_y / scale
        
        # 경계 확인
        original_x = max(0, min(original_x, original_size[0]))
        original_y = max(0, min(original_y, original_size[1]))
        
        return (original_x, original_y)
    
    def original_to_display_coords(self, orig_x: float, orig_y: float,
                                 ui_mode: str, camera_info: Dict,
                                 region_info: Dict = None) -> Optional[Tuple[float, float]]:
        """
        원본 카메라 좌표를 UI 표시 좌표로 변환 (역변환)
        bbox 표시할 때 사용
        """
        if ui_mode == "single":
            return self._original_to_single_display(orig_x, orig_y, camera_info)
        elif ui_mode == "dual":
            return self._original_to_dual_display(orig_x, orig_y, camera_info, region_info)
        else:
            return None
    
    def _original_to_single_display(self, orig_x: float, orig_y: float,
                                  camera_info: Dict) -> Optional[Tuple[float, float]]:
        """단일 모드: 원본 → 표시 좌표"""
        transform_info = camera_info.get("transform", {})
        
        # 원본 → YOLO 좌표
        yolo_coords = self._original_to_yolo_coords(orig_x, orig_y, transform_info)
        if yolo_coords is None:
            return None
        
        yolo_x, yolo_y = yolo_coords
        
        # YOLO → 표시 좌표
        display_x = (yolo_x / self.yolo_width) * self.display_width
        display_y = (yolo_y / self.yolo_height) * self.display_height
        
        return (display_x, display_y)
    
    def _original_to_dual_display(self, orig_x: float, orig_y: float,
                                camera_info: Dict, region_info: Dict) -> Optional[Tuple[float, float]]:
        """듀얼 모드: 원본 → 표시 좌표"""
        camera_name = camera_info.get("camera_name", "")
        
        # 어느 영역(primary/secondary)인지 판단
        primary_camera = camera_info.get("primary_camera", "")
        secondary_camera = camera_info.get("secondary_camera", "")
        
        if camera_name == primary_camera:
            region_type = "primary"
            transform_info = region_info.get("primary_region", {}).get("transform", {})
            display_offset_x = 0
        elif camera_name == secondary_camera:
            region_type = "secondary"
            transform_info = region_info.get("secondary_region", {}).get("transform", {})
            display_offset_x = self.dual_display_width
        else:
            return None
        
        # 원본 → 영역 내 YOLO 좌표
        region_yolo_coords = self._original_to_yolo_coords(orig_x, orig_y, transform_info)
        if region_yolo_coords is None:
            return None
        
        region_yolo_x, region_yolo_y = region_yolo_coords
        
        # 영역 내 YOLO → 영역 내 표시 좌표
        local_display_x = (region_yolo_x / self.dual_yolo_width) * self.dual_display_width
        local_display_y = (region_yolo_y / self.yolo_height) * self.display_height
        
        # 전체 화면에서의 좌표
        display_x = local_display_x + display_offset_x
        display_y = local_display_y
        
        return (display_x, display_y)
    
    def _original_to_yolo_coords(self, orig_x: float, orig_y: float,
                               transform_info: Dict) -> Optional[Tuple[float, float]]:
        """원본 좌표를 YOLO 좌표로 변환"""
        if not transform_info:
            return None
        
        scale = transform_info.get("scale", 1.0)
        x_offset = transform_info.get("x_offset", 0)
        y_offset = transform_info.get("y_offset", 0)
        
        if scale <= 0:
            return None
        
        # 스케일 적용
        scaled_x = orig_x * scale
        scaled_y = orig_y * scale
        
        # 패딩 추가
        yolo_x = scaled_x + x_offset
        yolo_y = scaled_y + y_offset
        
        return (yolo_x, yolo_y)
    
    def convert_bbox_coordinates(self, bbox: List[float], source_coords: str,
                               target_coords: str, camera_info: Dict,
                               region_info: Dict = None) -> Optional[List[float]]:
        """
        bbox 좌표를 다른 좌표계로 변환
        
        Args:
            bbox: [x1, y1, x2, y2] 형태
            source_coords: "original", "yolo", "display" 중 하나
            target_coords: "original", "yolo", "display" 중 하나
            camera_info: 카메라 정보
            region_info: 영역 정보 (듀얼 모드시)
        """
        x1, y1, x2, y2 = bbox
        
        # 좌상단과 우하단 점을 각각 변환
        if source_coords == "original" and target_coords == "display":
            ui_mode = camera_info.get("ui_mode", "single")
            
            p1 = self.original_to_display_coords(x1, y1, ui_mode, camera_info, region_info)
            p2 = self.original_to_display_coords(x2, y2, ui_mode, camera_info, region_info)
            
            if p1 and p2:
                return [p1[0], p1[1], p2[0], p2[1]]
        
        # 다른 변환들도 필요시 추가 구현
        return None

# 사용 예시 및 테스트
if __name__ == "__main__":
    transformer = CoordinateTransformer()
    
    # 테스트 카메라 정보
    camera_info_single = {
        "name": "[남해선] 죽평",
        "transform": {
            "scale": 0.5,
            "x_offset": 50,
            "y_offset": 100,
            "original_size": (1920, 1080)
        }
    }
    
    # 단일 모드 테스트
    print("=== 단일 모드 테스트 ===")
    click_result = transformer.click_to_original_coords(
        960, 540, "single", camera_info_single
    )
    print("클릭 변환 결과:", click_result)
    
    # 역변환 테스트
    if click_result and click_result["original_coords"]:
        orig_x, orig_y = click_result["original_coords"]
        display_coords = transformer.original_to_display_coords(
            orig_x, orig_y, "single", camera_info_single
        )
        print("역변환 결과:", display_coords)
    
    # 듀얼 모드 테스트
    print("\n=== 듀얼 모드 테스트 ===")
    camera_info_dual = {
        "primary_camera": "[남해선] 죽평",
        "secondary_camera": "[남해선] 선평교"
    }
    
    region_info_dual = {
        "primary_region": {
            "transform": {
                "scale": 0.6,
                "x_offset": 30,
                "y_offset": 50,
                "original_size": (1920, 1080)
            }
        },
        "secondary_region": {
            "transform": {
                "scale": 0.6,
                "x_offset": 30,
                "y_offset": 50,
                "original_size": (1920, 1080)
            }
        }
    }
    
    # Primary 영역 클릭 (왼쪽)
    primary_click = transformer.click_to_original_coords(
        480, 540, "dual", camera_info_dual, region_info_dual
    )
    print("Primary 영역 클릭:", primary_click)
    
    # Secondary 영역 클릭 (오른쪽)
    secondary_click = transformer.click_to_original_coords(
        1440, 540, "dual", camera_info_dual, region_info_dual
    )
    print("Secondary 영역 클릭:", secondary_click)
    
    # bbox 변환 테스트
    test_bbox = [100, 100, 200, 200]  # 원본 좌표계
    camera_info_single["ui_mode"] = "single"
    
    converted_bbox = transformer.convert_bbox_coordinates(
        test_bbox, "original", "display", camera_info_single
    )
    print("bbox 변환 결과:", converted_bbox)