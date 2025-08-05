import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# 다른 모듈들 import (실제 사용시)
# from core.data_manager import DataManager
# from handover.frame_concatenator import FrameConcatenator, BBoxSeparator
# from handover.coordinate_transformer import CoordinateTransformer

class HandoverState(Enum):
    """핸드오버 상태"""
    IDLE = "idle"                    # 대기 상태
    PREPARING = "preparing"          # 핸드오버 준비 중
    ACTIVE = "active"               # 핸드오버 진행 중
    MATCHING = "matching"           # 매칭 시도 중
    SUCCESS = "success"             # 매칭 성공
    TIMEOUT = "timeout"             # 시간 초과
    FAILED = "failed"               # 매칭 실패

class VehicleDirection(Enum):
    """차량 진행 방향"""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    UNKNOWN = "unknown"

class HandoverManager:
    """핸드오버 전체 프로세스를 관리하는 클래스"""
    
    def __init__(self, data_manager=None, yolo_model=None):
        """
        Args:
            data_manager: DataManager 인스턴스
            yolo_model: YOLO 모델 인스턴스
        """
        # 의존성 주입
        self.data_manager = data_manager
        self.yolo_model = yolo_model
        
        # 핵심 모듈들
        self.frame_concatenator = None  # FrameConcatenator()
        self.bbox_separator = None      # BBoxSeparator()
        self.coord_transformer = None   # CoordinateTransformer()
        
        # 상태 관리
        self.current_state = HandoverState.IDLE
        self.active_session_id = None
        self.handover_start_time = None
        
        # 설정값
        self.handover_timeout = 5.0     # 5초 타임아웃
        self.boundary_threshold = 0.1   # 화면 경계 10% 영역
        self.matching_confidence = 0.7  # 매칭 신뢰도 임계값
        
        # 캐시
        self.last_frames = {}           # 각 카메라별 마지막 프레임
        self.vehicle_history = {}       # 차량별 추적 히스토리
        
        print("HandoverManager 초기화 완료")
    
    def initialize_modules(self):
        """의존 모듈들 초기화"""
        # 실제 사용시 주석 해제
        # self.frame_concatenator = FrameConcatenator()
        # self.bbox_separator = BBoxSeparator()
        # self.coord_transformer = CoordinateTransformer()
        pass
    
    def update_frame(self, camera_name: str, frame: np.ndarray):
        """프레임 업데이트"""
        self.last_frames[camera_name] = {
            "frame": frame,
            "timestamp": time.time()
        }
    
    def check_handover_trigger(self, vehicle_id: str, bbox: List[float], 
                             camera_name: str) -> bool:
        """
        핸드오버 트리거 조건 확인
        
        Args:
            vehicle_id: 추적 중인 차량 ID
            bbox: [x1, y1, x2, y2] 형태의 bounding box
            camera_name: 현재 카메라명
            
        Returns:
            핸드오버 시작 여부
        """
        if self.current_state != HandoverState.IDLE:
            return False
        
        # 차량이 화면 경계 근처에 있는지 확인
        if not self._is_near_boundary(bbox):
            return False
        
        # 차량 진행 방향 추정
        direction = self._estimate_vehicle_direction(vehicle_id, bbox, camera_name)
        if direction == VehicleDirection.UNKNOWN:
            return False
        
        # 다음 카메라 확인
        next_camera = self._get_next_camera(camera_name, direction.value)
        if not next_camera:
            return False
        
        print(f"핸드오버 트리거: {vehicle_id} @ {camera_name} → {next_camera}")
        return True
    
    def start_handover(self, vehicle_id: str, source_camera: str, 
                      target_camera: str) -> bool:
        """
        핸드오버 시작
        
        Args:
            vehicle_id: 대상 차량 ID
            source_camera: 현재 카메라
            target_camera: 다음 카메라
            
        Returns:
            시작 성공 여부
        """
        if self.current_state != HandoverState.IDLE:
            print(f"이미 핸드오버 진행 중: {self.current_state}")
            return False
        
        # 상태 변경
        self.current_state = HandoverState.PREPARING
        self.handover_start_time = time.time()
        
        # 데이터 매니저에 세션 생성
        if self.data_manager:
            self.active_session_id = self.data_manager.create_handover_session(
                vehicle_id, source_camera, target_camera
            )
            
            # 차량 상태 업데이트
            self.data_manager.set_vehicle_handover(vehicle_id, source_camera, target_camera)
            
            # UI 모드 변경
            self.data_manager.set_ui_mode("dual", source_camera, target_camera)
        
        print(f"핸드오버 시작: {vehicle_id} ({source_camera} → {target_camera})")
        
        # 활성 상태로 전환
        self.current_state = HandoverState.ACTIVE
        return True
    
    def process_handover_frame(self, primary_camera: str, secondary_camera: str) -> Dict:
        """
        핸드오버 중 프레임 처리
        
        Args:
            primary_camera: 현재 카메라명
            secondary_camera: 다음 카메라명
            
        Returns:
            처리 결과
        """
        if self.current_state not in [HandoverState.ACTIVE, HandoverState.MATCHING]:
            return {"success": False, "reason": "not_in_handover_state"}
        
        # 프레임 가져오기
        primary_frame = self._get_latest_frame(primary_camera)
        secondary_frame = self._get_latest_frame(secondary_camera)
        
        if primary_frame is None:
            return {"success": False, "reason": "no_primary_frame"}
        
        # 프레임 결합
        if self.frame_concatenator:
            concat_frame, region_info = self.frame_concatenator.concatenate_frames(
                primary_frame, secondary_frame
            )
        else:
            # 테스트용 더미 처리
            concat_frame = primary_frame
            region_info = {"mode": "dual"}
        
        # YOLO 실행
        if self.yolo_model:
            yolo_results = self.yolo_model(concat_frame)
        else:
            # 테스트용 더미 결과
            yolo_results = []
        
        # 결과 분리
        if self.bbox_separator and region_info.get("mode") == "dual":
            separated_results = self.bbox_separator.separate_detections(
                yolo_results, region_info
            )
        else:
            separated_results = {"primary": yolo_results, "secondary": []}
        
        # 매칭 시도
        matching_result = self._attempt_vehicle_matching(separated_results)
        
        return {
            "success": True,
            "concat_frame": concat_frame,
            "region_info": region_info,
            "detections": separated_results,
            "matching_result": matching_result
        }
    
    def update_handover_state(self, matching_result: Dict = None) -> Dict:
        """
        핸드오버 상태 업데이트
        
        Args:
            matching_result: 매칭 결과
            
        Returns:
            상태 업데이트 결과
        """
        if self.current_state == HandoverState.IDLE:
            return {"state": self.current_state.value}
        
        elapsed_time = time.time() - self.handover_start_time
        
        # 매칭 성공
        if matching_result and matching_result.get("success"):
            return self._handle_matching_success(matching_result)
        
        # 타임아웃 확인
        if elapsed_time > self.handover_timeout:
            return self._handle_timeout()
        
        # 진행 중 상태 업데이트
        if self.current_state == HandoverState.ACTIVE:
            self.current_state = HandoverState.MATCHING
            
            if self.data_manager and self.active_session_id:
                self.data_manager.update_handover_session(
                    self.active_session_id, "searching"
                )
        
        return {
            "state": self.current_state.value,
            "elapsed_time": elapsed_time,
            "timeout_remaining": self.handover_timeout - elapsed_time,
            "progress": min(elapsed_time / self.handover_timeout, 1.0)
        }
    
    def _handle_matching_success(self, matching_result: Dict) -> Dict:
        """매칭 성공 처리"""
        self.current_state = HandoverState.SUCCESS
        
        vehicle_id = matching_result.get("vehicle_id")
        target_camera = matching_result.get("target_camera")
        
        if self.data_manager:
            # 세션 완료
            if self.active_session_id:
                self.data_manager.update_handover_session(
                    self.active_session_id, "matched"
                )
                self.data_manager.remove_handover_session(self.active_session_id)
            
            # 차량 정보 업데이트
            self.data_manager.clear_vehicle_handover(vehicle_id)
            
            # UI 모드 변경
            self.data_manager.set_ui_mode("single", target_camera)
        
        # 상태 초기화
        self._reset_handover_state()
        
        print(f"핸드오버 성공: {vehicle_id} → {target_camera}")
        
        return {
            "state": HandoverState.SUCCESS.value,
            "vehicle_id": vehicle_id,
            "target_camera": target_camera,
            "message": "매칭 성공"
        }
    
    def _handle_timeout(self) -> Dict:
        """타임아웃 처리"""
        self.current_state = HandoverState.TIMEOUT
        
        if self.data_manager:
            # 세션 타임아웃 처리
            if self.active_session_id:
                self.data_manager.update_handover_session(
                    self.active_session_id, "timeout"
                )
                self.data_manager.remove_handover_session(self.active_session_id)
            
            # UI 모드 복귀
            system_state = self.data_manager.get_system_state()
            primary_camera = system_state.get("active_camera")
            self.data_manager.set_ui_mode("single", primary_camera)
        
        # 상태 초기화
        self._reset_handover_state()
        
        print("핸드오버 타임아웃")
        
        return {
            "state": HandoverState.TIMEOUT.value,
            "message": "추적 실패 - 시간 초과"
        }
    
    def _reset_handover_state(self):
        """핸드오버 상태 초기화"""
        self.current_state = HandoverState.IDLE
        self.active_session_id = None
        self.handover_start_time = None
    
    def _is_near_boundary(self, bbox: List[float], frame_width: int = 640, 
                         frame_height: int = 640) -> bool:
        """차량이 화면 경계 근처에 있는지 확인"""
        x1, y1, x2, y2 = bbox
        
        # 경계 임계값 계산
        x_threshold = frame_width * self.boundary_threshold
        y_threshold = frame_height * self.boundary_threshold
        
        # 경계 근처 확인
        near_left = x1 < x_threshold
        near_right = x2 > (frame_width - x_threshold)
        near_top = y1 < y_threshold
        near_bottom = y2 > (frame_height - y_threshold)
        
        return near_left or near_right or near_top or near_bottom
    
    def _estimate_vehicle_direction(self, vehicle_id: str, current_bbox: List[float], 
                                  camera_name: str) -> VehicleDirection:
        """차량 진행 방향 추정"""
        # 이전 위치와 비교해서 방향 추정
        if vehicle_id not in self.vehicle_history:
            self.vehicle_history[vehicle_id] = []
        
        history = self.vehicle_history[vehicle_id]
        history.append({
            "bbox": current_bbox,
            "camera": camera_name,
            "timestamp": time.time()
        })
        
        # 최근 몇 개 프레임만 유지
        if len(history) > 10:
            history.pop(0)
        
        # 방향 추정 로직 (간단한 예시)
        if len(history) >= 3:
            recent = history[-3:]
            dx = recent[-1]["bbox"][0] - recent[0]["bbox"][0]  # x 변화량
            dy = recent[-1]["bbox"][1] - recent[0]["bbox"][1]  # y 변화량
            
            if abs(dx) > abs(dy):
                return VehicleDirection.EAST if dx > 0 else VehicleDirection.WEST
            else:
                return VehicleDirection.SOUTH if dy > 0 else VehicleDirection.NORTH
        
        return VehicleDirection.UNKNOWN
    
    def _get_next_camera(self, current_camera: str, direction: str) -> Optional[str]:
        """다음 카메라 조회"""
        if self.data_manager:
            return self.data_manager.get_next_camera(current_camera, direction)
        return None
    
    def _get_latest_frame(self, camera_name: str) -> Optional[np.ndarray]:
        """최신 프레임 가져오기"""
        if camera_name in self.last_frames:
            return self.last_frames[camera_name]["frame"]
        return None
    
    def _attempt_vehicle_matching(self, separated_results: Dict) -> Dict:
        """차량 매칭 시도"""
        # 실제 ReID 로직 구현 필요
        # 여기서는 간단한 더미 로직
        
        primary_detections = separated_results.get("primary", [])
        secondary_detections = separated_results.get("secondary", [])
        
        # 단순 매칭: secondary에 detection이 있으면 성공으로 간주
        if secondary_detections:
            return {
                "success": True,
                "vehicle_id": "test_vehicle",
                "target_camera": "target_camera",
                "confidence": 0.8
            }
        
        return {"success": False}
    
    def get_status(self) -> Dict:
        """현재 상태 조회"""
        status = {
            "state": self.current_state.value,
            "active_session": self.active_session_id,
            "elapsed_time": 0
        }
        
        if self.handover_start_time:
            status["elapsed_time"] = time.time() - self.handover_start_time
        
        return status

# 사용 예시
if __name__ == "__main__":
    # 핸드오버 매니저 초기화
    manager = HandoverManager()
    
    # 테스트 시나리오
    print("=== 핸드오버 테스트 ===")
    
    # 1. 핸드오버 트리거 확인
    test_bbox = [580, 300, 640, 400]  # 화면 오른쪽 경계 근처
    trigger_result = manager.check_handover_trigger(
        "A123", test_bbox, "[남해선] 죽평"
    )
    print(f"트리거 결과: {trigger_result}")
    
    # 2. 핸드오버 시작
    if trigger_result:
        start_result = manager.start_handover(
            "A123", "[남해선] 죽평", "[남해선] 선평교"
        )
        print(f"시작 결과: {start_result}")
        
        # 3. 상태 업데이트
        for i in range(5):
            time.sleep(0.5)
            status = manager.update_handover_state()
            print(f"상태 {i+1}: {status}")
            
            if status.get("state") in ["success", "timeout", "failed"]:
                break
    
    print("테스트 완료")