import cv2
import threading
import time
import os
from queue import Queue

class VideoStreamThread:
    """멀티스레드 비디오 스트림 처리"""
    
    def __init__(self, camera_name, stream_url, buffer_size=2):
        self.camera_name = camera_name
        self.stream_url = stream_url
        self.buffer_size = buffer_size
        
        # 스레드 제어
        self.stopped = False
        self.thread = None
        
        # 프레임 버퍼
        self.frame_queue = Queue(maxsize=buffer_size)
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # 연결 상태
        self.cap = None
        self.is_connected = False
        self.last_frame_time = 0
        self.fps = 0
        self.frame_count = 0
        
        # 자동 시작
        self._connect()
    
    def _connect(self):
        """스트림 연결"""
        try:
            print(f"📡 스트림 연결 중: {self.camera_name}")
            self.cap = cv2.VideoCapture(self.stream_url)
            
            # OpenCV 설정
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
            self.cap.set(cv2.CAP_PROP_FPS, 30)        # FPS 설정
            
            if self.cap.isOpened():
                self.is_connected = True
                print(f"✅ 스트림 연결 성공: {self.camera_name}")
                
                # 첫 프레임 테스트
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                    print(f"📷 첫 프레임 수신: {frame.shape}")
                    
                    # 스레드 시작
                    self.thread = threading.Thread(target=self._update)
                    self.thread.daemon = True
                    self.thread.start()
                    return True
                else:
                    print(f"⚠️ 첫 프레임 읽기 실패: {self.camera_name}")
                    return False
            else:
                print(f"❌ 스트림 연결 실패: {self.camera_name}")
                return False
                
        except Exception as e:
            print(f"💥 스트림 연결 오류: {e}")
            return False
    
    def _update(self):
        """프레임 업데이트 스레드"""
        fps_start = time.time()
        fps_count = 0
        
        while not self.stopped:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            try:
                ret, frame = self.cap.read()
                
                if ret:
                    # FPS 계산
                    fps_count += 1
                    if fps_count >= 30:
                        elapsed = time.time() - fps_start
                        self.fps = fps_count / elapsed
                        fps_start = time.time()
                        fps_count = 0
                    
                    # 프레임 저장
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                        self.last_frame_time = time.time()
                        self.frame_count += 1
                    
                    # 큐가 가득 차면 오래된 프레임 제거
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                    else:
                        try:
                            self.frame_queue.get_nowait()  # 오래된 프레임 제거
                            self.frame_queue.put(frame.copy())
                        except:
                            pass
                else:
                    print(f"⚠️ 프레임 읽기 실패: {self.camera_name}")
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"💥 스트림 업데이트 오류: {e}")
                time.sleep(0.1)
    
    def read(self):
        """최신 프레임 반환"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def read_queue(self):
        """큐에서 프레임 반환 (버퍼링)"""
        try:
            return self.frame_queue.get_nowait()
        except:
            return self.read()  # 큐가 비어있으면 최신 프레임
    
    def get_status(self):
        """스트림 상태 반환"""
        return {
            'camera_name': self.camera_name,
            'is_connected': self.is_connected,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'queue_size': self.frame_queue.qsize(),
            'last_frame_age': time.time() - self.last_frame_time if self.last_frame_time > 0 else float('inf')
        }
    
    def stop(self):
        """스트림 정지"""
        print(f"🛑 스트림 정지: {self.camera_name}")
        self.stopped = True
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.is_connected = False
    
    def restart(self):
        """스트림 재시작"""
        print(f"🔄 스트림 재시작: {self.camera_name}")
        self.stop()
        time.sleep(1.0)
        return self._connect()


class DualCameraManager:
    """듀얼 카메라 관리자"""
    
    def __init__(self):
        self.current_stream = None
        self.next_stream = None
        self.dual_mode = False
        self.dual_mode_start_time = 0
        self.dual_mode_timeout = 15.0  # 15초 후 자동 종료
        
    def set_current_camera(self, camera_name, stream_url):
        """현재 카메라 설정"""
        if self.current_stream:
            self.current_stream.stop()
        
        self.current_stream = VideoStreamThread(camera_name, stream_url)
        return self.current_stream.is_connected
    
    def activate_dual_mode(self, next_camera_name, next_stream_url):
        """듀얼 카메라 모드 활성화"""
        if not self.current_stream or not self.current_stream.is_connected:
            print("❌ 현재 카메라가 연결되지 않음")
            return False
        
        print(f"🔄 듀얼 카메라 모드 활성화")
        print(f"   현재: {self.current_stream.camera_name}")
        print(f"   다음: {next_camera_name}")
        
        # 다음 카메라 스트림 시작
        self.next_stream = VideoStreamThread(next_camera_name, next_stream_url)
        
        if self.next_stream.is_connected:
            self.dual_mode = True
            self.dual_mode_start_time = time.time()
            print("✅ 듀얼 카메라 모드 시작")
            return True
        else:
            print("❌ 다음 카메라 연결 실패")
            self.next_stream = None
            return False
    
    def get_frames(self):
        """현재 프레임(들) 반환"""
        if self.dual_mode:
            # 듀얼 모드: 두 프레임 반환
            current_frame = self.current_stream.read() if self.current_stream else None
            next_frame = self.next_stream.read() if self.next_stream else None
            return current_frame, next_frame
        else:
            # 싱글 모드: 현재 프레임만
            current_frame = self.current_stream.read() if self.current_stream else None
            return current_frame, None
    
    def switch_to_next(self):
        """다음 카메라로 전환"""
        if not self.dual_mode or not self.next_stream:
            print("❌ 듀얼 모드가 아니거나 다음 카메라 없음")
            return False
        
        print(f"🔄 카메라 전환: {self.current_stream.camera_name} → {self.next_stream.camera_name}")
        
        # 현재 스트림 정지
        if self.current_stream:
            self.current_stream.stop()
        
        # 다음 스트림을 현재로 이동
        self.current_stream = self.next_stream
        self.next_stream = None
        self.dual_mode = False
        
        print("✅ 카메라 전환 완료")
        return True
    
    def check_timeout(self):
        """듀얼 모드 타임아웃 확인"""
        if self.dual_mode and self.dual_mode_start_time > 0:
            elapsed = time.time() - self.dual_mode_start_time
            if elapsed > self.dual_mode_timeout:
                print(f"⏰ 듀얼 모드 타임아웃 ({elapsed:.1f}초) - 강제 종료")
                self.deactivate_dual_mode()
                return True
        return False
    
    def deactivate_dual_mode(self):
        """듀얼 모드 비활성화 (현재 카메라 유지)"""
        if self.dual_mode:
            print("🛑 듀얼 모드 비활성화")
            
            if self.next_stream:
                self.next_stream.stop()
                self.next_stream = None
            
            self.dual_mode = False
            self.dual_mode_start_time = 0
    
    def get_status(self):
        """듀얼 카메라 상태"""
        current_status = self.current_stream.get_status() if self.current_stream else None
        next_status = self.next_stream.get_status() if self.next_stream else None
        
        return {
            'dual_mode': self.dual_mode,
            'dual_mode_time': time.time() - self.dual_mode_start_time if self.dual_mode else 0,
            'current_camera': current_status,
            'next_camera': next_status
        }
    
    def shutdown(self):
        """전체 시스템 종료"""
        print("🛑 듀얼 카메라 시스템 종료")
        
        if self.current_stream:
            self.current_stream.stop()
        
        if self.next_stream:
            self.next_stream.stop()
        
        self.dual_mode = False


# 사용 예시
if __name__ == "__main__":
    # 듀얼 카메라 매니저 테스트
    manager = DualCameraManager()
    
    # 현재 카메라 설정
    current_url = os.getenv("CURRENT_CCTV_NAME")
    
    if manager.set_current_camera("[남해선] 죽평", current_url):
        print("✅ 현재 카메라 설정 완료")
        
        # 몇 초 대기
        time.sleep(3)
        
        # 듀얼 모드 활성화 테스트 (같은 URL로)
        if manager.activate_dual_mode("[남해선] 선평교", current_url):
            print("✅ 듀얼 모드 활성화")
            
            # 프레임 읽기 테스트
            for i in range(10):
                current_frame, next_frame = manager.get_frames()
                print(f"프레임 {i}: 현재={current_frame is not None}, 다음={next_frame is not None}")
                time.sleep(0.5)
            
            # 전환 테스트
            manager.switch_to_next()
        
        # 상태 확인
        status = manager.get_status()
        print(f"최종 상태: {status}")
    
    manager.shutdown()