import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional, Any

class FrameConcatenator:
    def __init__(self, target_width=640, target_height=640):
        """
        프레임 결합 시스템
        
        Args:
            target_width: YOLO 입력용 최종 프레임 너비
            target_height: YOLO 입력용 최종 프레임 높이
        """
        self.target_width = target_width
        self.target_height = target_height
        self.half_width = target_width // 2
        self.concat_mode = "horizontal"  # 고정 방식: 좌우 분할
        
    def prepare_single_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """단일 프레임을 YOLO 입력 크기로 준비"""
        if frame is None:
            empty_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
            return empty_frame, self._get_default_transform()
        
        prepared_frame, transform_info = self._resize_and_pad(
            frame, self.target_width, self.target_height
        )
        
        return prepared_frame, {
            "mode": "single",
            "transform": transform_info,
            "original_shape": frame.shape[:2] if frame is not None else (0, 0)
        }
    
    def concatenate_frames(self, primary_frame: np.ndarray, 
                         secondary_frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        두 프레임을 좌우로 결합
        
        Args:
            primary_frame: 현재 카메라 프레임 (왼쪽에 배치)
            secondary_frame: 예상 카메라 프레임 (오른쪽에 배치)
            
        Returns:
            결합된 프레임, 변환 정보
        """
        # 각 프레임을 절반 크기로 준비
        primary_prepared, primary_transform = self._resize_and_pad(
            primary_frame, self.half_width, self.target_height
        )
        
        secondary_prepared, secondary_transform = self._resize_and_pad(
            secondary_frame, self.half_width, self.target_height
        )
        
        # 좌우 결합
        concatenated = np.hstack([primary_prepared, secondary_prepared])
        
        # 구분선 추가 (선택사항)
        self._draw_separator_line(concatenated)
        
        # 변환 정보 구성
        region_info = {
            "mode": "dual",
            "primary_region": {
                "bbox": (0, 0, self.half_width, self.target_height),
                "transform": primary_transform,
                "original_shape": primary_frame.shape[:2] if primary_frame is not None else (0, 0)
            },
            "secondary_region": {
                "bbox": (self.half_width, 0, self.half_width, self.target_height),
                "transform": secondary_transform,
                "original_shape": secondary_frame.shape[:2] if secondary_frame is not None else (0, 0)
            }
        }
        
        return concatenated, region_info
    
    def _resize_and_pad(self, frame: np.ndarray, target_w: int, 
                       target_h: int) -> Tuple[np.ndarray, Dict]:
        """프레임을 목표 크기로 리사이즈하고 패딩 추가"""
        if frame is None:
            # 빈 프레임 생성 (검은색)
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            return padded, self._get_default_transform()
        
        h, w = frame.shape[:2]
        
        # 비율 유지하면서 스케일 계산
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 리사이즈
        if scale != 1.0:
            resized = cv2.resize(frame, (new_w, new_h))
        else:
            resized = frame.copy()
        
        # 패딩 추가해서 정확한 크기 맞추기
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        transform_info = {
            "scale": scale,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "resized_size": (new_w, new_h),
            "original_size": (w, h)
        }
        
        return padded, transform_info
    
    def _draw_separator_line(self, concatenated_frame: np.ndarray):
        """결합된 프레임에 구분선 그리기"""
        cv2.line(
            concatenated_frame, 
            (self.half_width, 0), 
            (self.half_width, self.target_height), 
            (0, 255, 0), 2  # 초록색 선
        )
        
        # 라벨 추가 (선택사항)
        cv2.putText(
            concatenated_frame, "PRIMARY", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            concatenated_frame, "SECONDARY", (self.half_width + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )
    
    def _get_default_transform(self) -> Dict:
        """기본 변환 정보 반환"""
        return {
            "scale": 1.0,
            "x_offset": 0,
            "y_offset": 0,
            "resized_size": (0, 0),
            "original_size": (0, 0)
        }

class BBoxSeparator:
    """YOLO 결과를 각 카메라별로 분리하는 클래스"""
    
    def __init__(self):
        pass
    
    def separate_detections(self, yolo_results: List, 
                          region_info: Dict) -> Dict[str, List]:
        """
        YOLO 결과를 각 카메라별로 분리
        
        Args:
            yolo_results: YOLO detection 결과 [[x1,y1,x2,y2,conf,cls], ...]
            region_info: 프레임 결합 시 생성된 변환 정보
            
        Returns:
            각 카메라별로 분리된 detection 결과
        """
        if region_info["mode"] == "single":
            return {"primary": yolo_results, "secondary": []}
        
        separated = {"primary": [], "secondary": []}
        
        primary_region = region_info["primary_region"]
        secondary_region = region_info["secondary_region"]
        
        for detection in yolo_results:
            x1, y1, x2, y2, conf, cls = detection
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            
            # bbox 중심점이 어느 영역에 속하는지 판단
            if self._point_in_region(bbox_center_x, bbox_center_y, primary_region["bbox"]):
                # Primary 카메라 좌표로 변환
                converted = self._convert_to_original_coords(
                    detection, primary_region
                )
                if converted is not None:
                    separated["primary"].append(converted)
                    
            elif self._point_in_region(bbox_center_x, bbox_center_y, secondary_region["bbox"]):
                # Secondary 카메라 좌표로 변환
                converted = self._convert_to_original_coords(
                    detection, secondary_region
                )
                if converted is not None:
                    separated["secondary"].append(converted)
        
        return separated
    
    def _point_in_region(self, x: float, y: float, region_bbox: Tuple) -> bool:
        """점이 특정 영역에 있는지 확인"""
        rx, ry, rw, rh = region_bbox
        return rx <= x <= rx + rw and ry <= y <= ry + rh
    
    def _convert_to_original_coords(self, detection: List, region_info: Dict) -> Optional[List]:
        """
        결합된 프레임의 detection을 원본 카메라 좌표로 변환
        
        Args:
            detection: [x1, y1, x2, y2, conf, cls]
            region_info: 해당 영역의 변환 정보
            
        Returns:
            원본 좌표계로 변환된 detection
        """
        x1, y1, x2, y2, conf, cls = detection
        
        region_bbox = region_info["bbox"]
        transform = region_info["transform"]
        
        # 영역 좌표계로 변환 (결합된 프레임 → 개별 영역)
        local_x1 = x1 - region_bbox[0]
        local_y1 = y1 - region_bbox[1]
        local_x2 = x2 - region_bbox[0]
        local_y2 = y2 - region_bbox[1]
        
        # 패딩 제거 (개별 영역 → 리사이즈된 프레임)
        unpadded_x1 = local_x1 - transform["x_offset"]
        unpadded_y1 = local_y1 - transform["y_offset"]
        unpadded_x2 = local_x2 - transform["x_offset"]
        unpadded_y2 = local_y2 - transform["y_offset"]
        
        # 스케일 해제 (리사이즈된 프레임 → 원본 프레임)
        if transform["scale"] > 0:
            orig_x1 = unpadded_x1 / transform["scale"]
            orig_y1 = unpadded_y1 / transform["scale"]
            orig_x2 = unpadded_x2 / transform["scale"]
            orig_y2 = unpadded_y2 / transform["scale"]
        else:
            return None
        
        # 유효성 검사
        orig_w, orig_h = transform["original_size"]
        if orig_w <= 0 or orig_h <= 0:
            return None
            
        # 경계 확인
        orig_x1 = max(0, min(orig_x1, orig_w))
        orig_y1 = max(0, min(orig_y1, orig_h))
        orig_x2 = max(0, min(orig_x2, orig_w))
        orig_y2 = max(0, min(orig_y2, orig_h))
        
        # bbox 크기 확인
        if orig_x2 <= orig_x1 or orig_y2 <= orig_y1:
            return None
        
        return [orig_x1, orig_y1, orig_x2, orig_y2, conf, cls]

# 사용 예시 및 테스트
if __name__ == "__main__":
    # 테스트용 프레임 생성
    def create_test_frame(width=1920, height=1080, color=(100, 100, 100)):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        # 테스트용 사각형 추가
        cv2.rectangle(frame, (width//4, height//4), (width//2, height//2), (255, 255, 255), -1)
        return frame
    
    # 프레임 결합 테스트
    concatenator = FrameConcatenator()
    separator = BBoxSeparator()
    
    # 테스트 프레임들
    primary_frame = create_test_frame(color=(50, 50, 150))   # 빨간색 계열
    secondary_frame = create_test_frame(color=(50, 150, 50)) # 초록색 계열
    
    # 단일 프레임 테스트
    single_result, single_info = concatenator.prepare_single_frame(primary_frame)
    print("단일 프레임 크기:", single_result.shape)
    print("단일 프레임 변환 정보:", single_info)
    
    # 듀얼 프레임 테스트
    concat_result, region_info = concatenator.concatenate_frames(primary_frame, secondary_frame)
    print("\n결합된 프레임 크기:", concat_result.shape)
    print("영역 정보:", region_info["mode"])
    
    # 가짜 YOLO 결과로 분리 테스트
    fake_yolo_results = [
        [100, 100, 200, 200, 0.8, 0],  # Primary 영역에 있을 것
        [400, 100, 500, 200, 0.9, 0],  # Secondary 영역에 있을 것
        [300, 300, 350, 350, 0.7, 0],  # 경계선 근처
    ]
    
    separated_results = separator.separate_detections(fake_yolo_results, region_info)
    print("\n분리된 결과:")
    print("Primary 카메라:", len(separated_results["primary"]), "개")
    print("Secondary 카메라:", len(separated_results["secondary"]), "개")
    
    # 결합된 프레임 저장 (테스트용)
    cv2.imwrite("test_concatenated_frame.jpg", concat_result)
    print("\n테스트 결과가 'test_concatenated_frame.jpg'로 저장되었습니다.")