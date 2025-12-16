import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import threading

class DataManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.lock = threading.Lock()
        
        # JSON 파일 경로들
        self.files = {
            "vehicle_tracking": os.path.join(data_dir, "vehicle_tracking.json"),
            "handover_sessions": os.path.join(data_dir, "handover_sessions.json"),
            "system_state": os.path.join(data_dir, "system_state.json"),
            "camera_connections": os.path.join(data_dir, "cctv_graph_connections.json")
        }
        
        self._initialize_data_directory()
        self._initialize_json_files()
    
    def _initialize_data_directory(self):
        """데이터 디렉토리 생성"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _initialize_json_files(self):
        """JSON 파일들 초기화"""
        default_data = {
            "vehicle_tracking": {
                "current_vehicles": {}
            },
            "handover_sessions": {
                "active_handovers": {}
            },
            "system_state": {
                "current_ui_mode": "single",
                "active_camera": None,
                "secondary_camera": None,
                "selected_vehicle": None,
                "last_update": self._get_timestamp()
            }
        }
        
        for file_key, file_path in self.files.items():
            if file_key == "camera_connections":
                continue  # 이미 존재하는 파일
                
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_data[file_key], f, indent=2, ensure_ascii=False)
                print(f"초기화됨: {file_path}")
    
    def _get_timestamp(self):
        """현재 타임스탬프 반환"""
        return datetime.now().isoformat()
    
    def load_json(self, file_key: str) -> Dict[str, Any]:
        """JSON 파일 로드"""
        if file_key not in self.files:
            raise ValueError(f"알 수 없는 파일 키: {file_key}")
        
        try:
            with open(self.files[file_key], 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"파일을 찾을 수 없음: {self.files[file_key]}")
            return {}
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return {}
    
    def save_json(self, file_key: str, data: Dict[str, Any]):
        """JSON 파일 저장"""
        if file_key not in self.files:
            raise ValueError(f"알 수 없는 파일 키: {file_key}")
        
        with self.lock:
            try:
                with open(self.files[file_key], 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"파일 저장 오류: {e}")
    
    # === 차량 추적 관련 메서드 ===
    def add_vehicle(self, vehicle_id: str, camera: str, position: Dict, 
                   vehicle_class: str = "unknown"):
        """새 차량 추가"""
        data = self.load_json("vehicle_tracking")
        
        data["current_vehicles"][vehicle_id] = {
            "last_camera": camera,
            "last_position": position,
            "last_timestamp": self._get_timestamp(),
            "vehicle_class": vehicle_class,
            "status": "tracking",
            "handover_info": {
                "active": False,
                "source_camera": None,
                "target_camera": None,
                "start_time": None
            }
        }
        
        self.save_json("vehicle_tracking", data)
        print(f"차량 추가됨: {vehicle_id} at {camera}")
    
    def update_vehicle_position(self, vehicle_id: str, camera: str, position: Dict):
        """차량 위치 업데이트"""
        data = self.load_json("vehicle_tracking")
        
        if vehicle_id in data["current_vehicles"]:
            vehicle = data["current_vehicles"][vehicle_id]
            vehicle["last_camera"] = camera
            vehicle["last_position"] = position
            vehicle["last_timestamp"] = self._get_timestamp()
            vehicle["status"] = "tracking"
            
            self.save_json("vehicle_tracking", data)
    
    def get_vehicle_info(self, vehicle_id: str) -> Optional[Dict]:
        """차량 정보 조회"""
        data = self.load_json("vehicle_tracking")
        return data["current_vehicles"].get(vehicle_id)
    
    def remove_vehicle(self, vehicle_id: str):
        """차량 제거"""
        data = self.load_json("vehicle_tracking")
        if vehicle_id in data["current_vehicles"]:
            del data["current_vehicles"][vehicle_id]
            self.save_json("vehicle_tracking", data)
            print(f"차량 제거됨: {vehicle_id}")
    
    def set_vehicle_handover(self, vehicle_id: str, source_camera: str, 
                           target_camera: str):
        """차량 핸드오버 상태 설정"""
        data = self.load_json("vehicle_tracking")
        
        if vehicle_id in data["current_vehicles"]:
            handover_info = data["current_vehicles"][vehicle_id]["handover_info"]
            handover_info["active"] = True
            handover_info["source_camera"] = source_camera
            handover_info["target_camera"] = target_camera
            handover_info["start_time"] = self._get_timestamp()
            
            data["current_vehicles"][vehicle_id]["status"] = "handover"
            self.save_json("vehicle_tracking", data)
    
    def clear_vehicle_handover(self, vehicle_id: str):
        """차량 핸드오버 상태 해제"""
        data = self.load_json("vehicle_tracking")
        
        if vehicle_id in data["current_vehicles"]:
            handover_info = data["current_vehicles"][vehicle_id]["handover_info"]
            handover_info["active"] = False
            handover_info["source_camera"] = None
            handover_info["target_camera"] = None
            handover_info["start_time"] = None
            
            data["current_vehicles"][vehicle_id]["status"] = "tracking"
            self.save_json("vehicle_tracking", data)
    
    # === 핸드오버 세션 관리 ===
    def create_handover_session(self, vehicle_id: str, source_camera: str, 
                              target_camera: str) -> str:
        """핸드오버 세션 생성"""
        data = self.load_json("handover_sessions")
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{vehicle_id}"
        
        data["active_handovers"][session_id] = {
            "vehicle_id": vehicle_id,
            "source_camera": source_camera,
            "target_camera": target_camera,
            "start_time": self._get_timestamp(),
            "status": "searching",
            "ui_mode": "dual"
        }
        
        self.save_json("handover_sessions", data)
        return session_id
    
    def update_handover_session(self, session_id: str, status: str):
        """핸드오버 세션 상태 업데이트"""
        data = self.load_json("handover_sessions")
        
        if session_id in data["active_handovers"]:
            data["active_handovers"][session_id]["status"] = status
            self.save_json("handover_sessions", data)
    
    def remove_handover_session(self, session_id: str):
        """핸드오버 세션 제거"""
        data = self.load_json("handover_sessions")
        
        if session_id in data["active_handovers"]:
            del data["active_handovers"][session_id]
            self.save_json("handover_sessions", data)
    
    def get_active_handovers(self) -> Dict:
        """활성 핸드오버 세션들 조회"""
        data = self.load_json("handover_sessions")
        return data["active_handovers"]
    
    # === 시스템 상태 관리 ===
    def set_ui_mode(self, mode: str, active_camera: str = None, 
                   secondary_camera: str = None):
        """UI 모드 설정"""
        data = self.load_json("system_state")
        
        data["current_ui_mode"] = mode
        data["active_camera"] = active_camera
        data["secondary_camera"] = secondary_camera
        data["last_update"] = self._get_timestamp()
        
        self.save_json("system_state", data)
    
    def set_selected_vehicle(self, vehicle_id: str):
        """선택된 차량 설정"""
        data = self.load_json("system_state")
        data["selected_vehicle"] = vehicle_id
        data["last_update"] = self._get_timestamp()
        self.save_json("system_state", data)
    
    def get_system_state(self) -> Dict:
        """시스템 상태 조회"""
        return self.load_json("system_state")
    
    # === 카메라 연결 정보 ===
    def get_camera_connections(self) -> Dict:
        """카메라 연결 정보 조회"""
        return self.load_json("camera_connections")
    
    def get_next_camera(self, current_camera: str, direction: str) -> Optional[str]:
        """다음 카메라 조회"""
        connections = self.get_camera_connections()
        
        for camera_info in connections:
            if camera_info["cctvname"] == current_camera:
                for conn in camera_info["connections"]:
                    if conn["direction"] == direction:
                        return conn["target"]
        return None

# 사용 예시 및 테스트
if __name__ == "__main__":
    # 데이터 매니저 초기화
    dm = DataManager()
    
    # 테스트 차량 추가
    dm.add_vehicle(
        vehicle_id="A123",
        camera="[남해선] 죽평",
        position={"x": 450, "y": 300, "w": 80, "h": 60},
        vehicle_class="sedan"
    )
    
    # 차량 정보 조회
    vehicle_info = dm.get_vehicle_info("A123")
    print("차량 정보:", vehicle_info)
    
    # 핸드오버 세션 생성
    session_id = dm.create_handover_session(
        vehicle_id="A123",
        source_camera="[남해선] 죽평", 
        target_camera="[남해선] 선평교"
    )
    print("핸드오버 세션 생성:", session_id)
    
    # UI 모드 변경
    dm.set_ui_mode("dual", "[남해선] 죽평", "[남해선] 선평교")
    
    # 시스템 상태 확인
    system_state = dm.get_system_state()
    print("시스템 상태:", system_state)