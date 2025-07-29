import json
import time
import threading
from utils.stream import VideoStreamThread

class CameraSwitcher:
    """CCTV 연결 관계 기반 카메라 전환 시스템"""
    
    def __init__(self, connections_file="cctv_graph_connections.json", cctv_list_file="cctv_list_4.json"):
        # 연결 관계 및 CCTV 정보 로드
        self.connections = self._load_connections(connections_file)
        self.cctv_list = self._load_cctv_list(cctv_list_file)
        
        # 현재 활성 카메라
        self.current_cctv = None
        self.current_stream = None
        
        # 전환 상태 관리
        self.switching_in_progress = False
        self.switch_timeout = 10.0  # 전환 타임아웃 (초)
        
        # 연결 관계 인덱스 (빠른 조회용)
        self.connection_index = self._build_connection_index()
        
        print(f"📡 카메라 전환 시스템 초기화 완료")
        print(f"   연결된 CCTV: {len(self.connections)}개")
        print(f"   사용 가능 CCTV: {len(self.cctv_list)}개")
    
    def _load_connections(self, file_path):
        """연결 관계 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                connections = json.load(f)
            print(f"✅ 연결 관계 로드: {file_path}")
            return connections
        except Exception as e:
            print(f"❌ 연결 관계 로드 실패: {e}")
            return []
    
    def _load_cctv_list(self, file_path):
        """CCTV 목록 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cctv_list = json.load(f)
            print(f"✅ CCTV 목록 로드: {file_path}")
            return cctv_list
        except Exception as e:
            print(f"❌ CCTV 목록 로드 실패: {e}")
            return []
    
    def _build_connection_index(self):
        """빠른 조회를 위한 연결 관계 인덱스 구축"""
        index = {}
        for cctv in self.connections:
            name = cctv["cctvname"]
            index[name] = {}
            for conn in cctv["connections"]:
                direction = conn["direction"]
                target = conn["target"]
                index[name][direction] = target
        
        print(f"🔗 연결 관계 인덱스 구축 완료")
        return index
    
    def get_cctv_info(self, cctv_name):
        """CCTV 이름으로 상세 정보 조회"""
        for cctv in self.cctv_list:
            if cctv_name in cctv["cctvname"]:
                return cctv
        return None
    
    def get_connected_camera(self, current_cctv_name, direction):
        """연결된 다음 카메라 조회"""
        if current_cctv_name not in self.connection_index:
            print(f"❌ 연결 정보 없음: {current_cctv_name}")
            return None
        
        connections = self.connection_index[current_cctv_name]
        if direction not in connections:
            print(f"❌ {direction} 방향 연결 없음: {current_cctv_name}")
            return None
        
        target_name = connections[direction]
        target_info = self.get_cctv_info(target_name)
        
        if target_info:
            print(f"🎯 다음 카메라 찾음: {current_cctv_name} → {target_name} ({direction})")
            return target_info
        else:
            print(f"❌ 타겟 카메라 정보 없음: {target_name}")
            return None
    
    def get_available_directions(self, cctv_name):
        """해당 CCTV에서 이동 가능한 방향들 반환"""
        if cctv_name not in self.connection_index:
            return []
        
        return list(self.connection_index[cctv_name].keys())
    
    def map_handover_direction_to_connection(self, handover_direction):
        """핸드오버 방향을 연결 관계 방향으로 매핑"""
        direction_mapping = {
            # 핸드오버 감지 방향 → 연결 관계 방향
            'left': 'south',    # 왼쪽으로 나가면 남쪽 방향
            'right': 'north',   # 오른쪽으로 나가면 북쪽 방향
            'top': 'north',     # 위로 나가면 북쪽 방향
            'bottom': 'south',  # 아래로 나가면 남쪽 방향
            
            # 직접 방향 지정도 허용
            'north': 'north',
            'south': 'south'
        }
        
        mapped = direction_mapping.get(handover_direction)
        if mapped:
            print(f"🧭 방향 매핑: {handover_direction} → {mapped}")
        else:
            print(f"⚠️ 알 수 없는 방향: {handover_direction}")
        
        return mapped
    
    def start_camera(self, cctv_name):
        """카메라 스트림 시작"""
        cctv_info = self.get_cctv_info(cctv_name)
        if not cctv_info:
            print(f"❌ CCTV 정보 없음: {cctv_name}")
            return False
        
        try:
            # 기존 스트림 정리
            if self.current_stream:
                print(f"📺 기존 스트림 종료: {self.current_cctv['cctvname'] if self.current_cctv else 'Unknown'}")
                self.current_stream.stop()
                self.current_stream = None
            
            # 새 스트림 시작
            print(f"🚀 새 스트림 시작: {cctv_info['cctvname']}")
            self.current_stream = VideoStreamThread(cctv_info["cctvname"], cctv_info["cctvurl"])
            self.current_cctv = cctv_info
            
            # 스트림 안정화 대기
            time.sleep(1.0)
            
            # 첫 프레임 확인
            frame = self.current_stream.read()
            if frame is not None:
                print(f"✅ 스트림 연결 성공: {cctv_info['cctvname']}")
                return True
            else:
                print(f"⚠️ 스트림 연결 불안정: {cctv_info['cctvname']}")
                return False
                
        except Exception as e:
            print(f"❌ 스트림 시작 실패: {e}")
            return False
    
    def switch_camera(self, handover_direction, timeout=None):
        """핸드오버 방향에 따라 카메라 전환"""
        if self.switching_in_progress:
            print("⚠️ 이미 카메라 전환 진행 중")
            return False
        
        if not self.current_cctv:
            print("❌ 현재 활성 카메라 없음")
            return False
        
        # 방향 매핑
        connection_direction = self.map_handover_direction_to_connection(handover_direction)
        if not connection_direction:
            print(f"❌ 잘못된 핸드오버 방향: {handover_direction}")
            return False
        
        # 다음 카메라 찾기
        next_cctv = self.get_connected_camera(self.current_cctv["cctvname"], connection_direction)
        if not next_cctv:
            print(f"❌ 다음 카메라를 찾을 수 없음: {handover_direction}")
            return False
        
        # 전환 시작
        self.switching_in_progress = True
        switch_start_time = time.time()
        
        try:
            print(f"🔄 카메라 전환 시작:")
            print(f"   {self.current_cctv['cctvname']} → {next_cctv['cctvname']}")
            print(f"   방향: {handover_direction} → {connection_direction}")
            
            # 새 카메라로 전환
            success = self.start_camera(next_cctv["cctvname"])
            
            if success:
                switch_time = time.time() - switch_start_time
                print(f"✅ 카메라 전환 완료 ({switch_time:.2f}초)")
                return True
            else:
                print(f"❌ 카메라 전환 실패")
                return False
                
        except Exception as e:
            print(f"💥 카메라 전환 중 오류: {e}")
            return False
        finally:
            self.switching_in_progress = False
    
    def get_current_frame(self):
        """현재 활성 카메라에서 프레임 읽기"""
        if not self.current_stream:
            return None
        
        return self.current_stream.read()
    
    def get_current_camera_info(self):
        """현재 활성 카메라 정보 반환"""
        return {
            'cctv_info': self.current_cctv,
            'is_active': self.current_stream is not None,
            'available_directions': self.get_available_directions(
                self.current_cctv["cctvname"] if self.current_cctv else ""
            ),
            'switching_in_progress': self.switching_in_progress
        }
    
    def stop_all_streams(self):
        """모든 스트림 정리"""
        if self.current_stream:
            print(f"🛑 스트림 종료: {self.current_cctv['cctvname'] if self.current_cctv else 'Unknown'}")
            self.current_stream.stop()
            self.current_stream = None
            self.current_cctv = None
    
    def get_network_status(self):
        """네트워크 연결 상태 반환"""
        if not self.current_cctv:
            return "NO_CAMERA"
        
        current_name = self.current_cctv["cctvname"]
        available_directions = self.get_available_directions(current_name)
        
        return {
            'current_camera': current_name,
            'available_directions': available_directions,
            'can_switch': len(available_directions) > 0,
            'connection_count': len(available_directions)
        }


# 핸드오버 감지와 연동하는 통합 클래스
class IntegratedHandoverSystem:
    """핸드오버 감지 + 카메라 전환 통합 시스템"""
    
    def __init__(self, connections_file="cctv_graph_connections.json", cctv_list_file="cctv_list_4.json"):
        self.camera_switcher = CameraSwitcher(connections_file, cctv_list_file)
        self.auto_switch_enabled = True
        self.handover_history = []
        
    def start_with_camera(self, cctv_name):
        """특정 카메라로 시스템 시작"""
        return self.camera_switcher.start_camera(cctv_name)
    
    def process_handover_event(self, handover_event):
        """핸드오버 이벤트 처리"""
        if handover_event['type'] != 'CONFIRMED':
            return False
        
        if not self.auto_switch_enabled:
            print("🔒 자동 전환 비활성화됨")
            return False
        
        track_id = handover_event['track_id']
        direction = handover_event['direction']
        
        print(f"🎯 핸드오버 이벤트 처리: ID{track_id} → {direction}")
        
        # 카메라 전환 실행
        success = self.camera_switcher.switch_camera(direction)
        
        # 이력 기록
        handover_record = {
            'timestamp': time.time(),
            'track_id': track_id,
            'direction': direction,
            'success': success,
            'from_camera': handover_event.get('candidate', {}).get('track_info', {}).get('bbox'),
            'to_camera': self.camera_switcher.current_cctv['cctvname'] if success else None
        }
        
        self.handover_history.append(handover_record)
        
        if success:
            print(f"✅ 핸드오버 성공: ID{track_id}")
        else:
            print(f"❌ 핸드오버 실패: ID{track_id}")
        
        return success
    
    def get_current_frame(self):
        """현재 프레임 반환"""
        return self.camera_switcher.get_current_frame()
    
    def get_status(self):
        """시스템 전체 상태 반환"""
        camera_status = self.camera_switcher.get_current_camera_info()
        network_status = self.camera_switcher.get_network_status()
        
        return {
            'camera': camera_status,
            'network': network_status,
            'auto_switch': self.auto_switch_enabled,
            'handover_count': len(self.handover_history),
            'recent_handovers': [h for h in self.handover_history 
                               if time.time() - h['timestamp'] < 60]
        }
    
    def toggle_auto_switch(self):
        """자동 전환 토글"""
        self.auto_switch_enabled = not self.auto_switch_enabled
        status = "활성화" if self.auto_switch_enabled else "비활성화"
        print(f"🔄 자동 카메라 전환 {status}")
        return self.auto_switch_enabled
    
    def manual_switch(self, direction):
        """수동 카메라 전환"""
        print(f"🎮 수동 카메라 전환: {direction}")
        return self.camera_switcher.switch_camera(direction)
    
    def shutdown(self):
        """시스템 종료"""
        print("🛑 통합 핸드오버 시스템 종료")
        self.camera_switcher.stop_all_streams()


# 사용 예시
if __name__ == "__main__":
    # 시스템 초기화
    handover_system = IntegratedHandoverSystem()
    
    # 시작 카메라 설정
    if handover_system.start_with_camera("[남해선] 죽평"):
        print("🚀 시스템 시작 성공")
        
        # 상태 확인
        status = handover_system.get_status()
        print(f"📊 현재 상태: {status}")
        
        # 수동 전환 테스트
        print("\n🎮 수동 전환 테스트:")
        handover_system.manual_switch("north")  # 죽평 → 선평교
        
        time.sleep(2)
        
        handover_system.manual_switch("south")  # 선평교 → 지본교 (잘못된 연결 테스트)
        
        # 종료
        handover_system.shutdown()
    else:
        print("❌ 시스템 시작 실패")