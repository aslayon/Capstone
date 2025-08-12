"""
HLS 스트림 매니저 - 끊김 없는 재생을 위한 개선 버전
파일명: stream_manager.py
위치: core/stream_manager.py
"""

import cv2
import numpy as np
import requests
import time
import threading
import queue
from collections import deque
from typing import Dict, Optional, Tuple
import os
from dotenv import load_dotenv

class HLSStreamManager:
    """HLS 스트림 끊김 문제를 해결하는 스트림 매니저"""
    
    def __init__(self, api_key: str, update_interval: int = 20):
        """
        Args:
            api_key: ITS API 키
            update_interval: URL 갱신 주기 (초) - HLS 세그먼트 길이 고려
        """
        self.api_key = api_key
        self.update_interval = update_interval
        
        # 스트림 관리
        self.current_url = None
        self.current_cap = None
        self.backup_cap = None
        
        # 프레임 버퍼 (끊김 방지)
        self.frame_buffer = queue.Queue(maxsize=60)  # 2초 분량
        self.last_valid_frame = None
        
        # 스레드 관리
        self.running = False
        self.url_updater_thread = None
        self.frame_reader_thread = None
        
        # 통계
        self.stats = {
            'url_updates': 0,
            'stream_reconnects': 0,
            'frames_read': 0,
            'buffer_underruns': 0,
            'last_update_time': 0
        }
        
        # HLS 점프 감지
        self.jump_detector = HLSJumpDetector()
        
        print("🚀 HLS 스트림 매니저 초기화")
    
    def start(self, cctv_name: str, initial_url: str) -> bool:
        """스트림 시작"""
        print(f"📡 스트림 시작: {cctv_name}")
        
        self.cctv_name = cctv_name
        self.current_url = initial_url
        self.running = True
        
        # 초기 연결
        if not self._connect_stream(initial_url):
            print("❌ 초기 스트림 연결 실패")
            return False
        
        # URL 갱신 스레드 시작
        self.url_updater_thread = threading.Thread(
            target=self._url_update_loop,
            daemon=True
        )
        self.url_updater_thread.start()
        
        # 프레임 읽기 스레드 시작
        self.frame_reader_thread = threading.Thread(
            target=self._frame_reader_loop,
            daemon=True
        )
        self.frame_reader_thread.start()
        
        print("✅ 스트림 매니저 시작 완료")
        return True
    
    def _connect_stream(self, url: str) -> bool:
        """스트림 연결 (최적화된 설정)"""
        try:
            cap = cv2.VideoCapture(url)
            
            if not cap.isOpened():
                return False
            
            # HLS 최적화 설정
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 최소 버퍼
            
            # 첫 프레임 테스트
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return False
            
            # 백업으로 현재 cap 저장
            if self.current_cap:
                self.backup_cap = self.current_cap
            
            self.current_cap = cap
            self.last_valid_frame = frame
            
            print(f"✅ 스트림 연결 성공")
            return True
            
        except Exception as e:
            print(f"❌ 스트림 연결 오류: {e}")
            return False
    
    def _url_update_loop(self):
        """URL 주기적 갱신 스레드"""
        print("🔄 URL 갱신 스레드 시작")
        
        while self.running:
            try:
                # 대기 (첫 실행은 즉시)
                if self.stats['url_updates'] > 0:
                    time.sleep(self.update_interval)
                
                # 새 URL 가져오기
                new_url = self._fetch_new_url()
                
                if new_url and new_url != self.current_url:
                    print(f"🔄 새 URL 획득 (갱신 #{self.stats['url_updates'] + 1})")
                    
                    # 새 스트림 연결 시도
                    if self._connect_stream(new_url):
                        self.current_url = new_url
                        self.stats['url_updates'] += 1
                        self.stats['last_update_time'] = time.time()
                        
                        # 이전 백업 정리
                        if self.backup_cap:
                            self.backup_cap.release()
                            self.backup_cap = None
                    else:
                        print("⚠️ 새 URL 연결 실패, 현재 스트림 유지")
                
            except Exception as e:
                print(f"❌ URL 갱신 오류: {e}")
                time.sleep(5)  # 오류 시 잠시 대기
    
    def _fetch_new_url(self) -> Optional[str]:
        """API에서 새 URL 가져오기"""
        try:
            # 실제 API 호출 (좌표 기반)
            url = "https://openapi.its.go.kr:9443/cctvInfo"
            
            # cctv_name으로 좌표 찾기 (실제로는 DB나 캐시에서)
            # 여기서는 예시 좌표 사용
            params = {
                "apiKey": self.api_key,
                "type": "all", 
                "cctvType": "1",
                "minX": "127.0",
                "maxX": "127.1",
                "minY": "34.9",
                "maxY": "35.0",
                "getType": "json"
            }
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if "response" in data and "data" in data["response"]:
                for cctv in data["response"]["data"]:
                    if self.cctv_name in cctv.get("cctvname", ""):
                        return cctv.get("cctvurl")
            
            return None
            
        except Exception as e:
            print(f"⚠️ API 호출 실패: {e}")
            return None
    
    def _frame_reader_loop(self):
        """프레임 읽기 스레드 (버퍼링)"""
        print("📹 프레임 읽기 스레드 시작")
        
        consecutive_failures = 0
        max_failures = 30  # 1초 정도
        
        while self.running:
            try:
                if not self.current_cap:
                    time.sleep(0.1)
                    continue
                
                # 프레임 읽기
                ret, frame = self.current_cap.read()
                
                if ret:
                    # 성공: 버퍼에 추가
                    consecutive_failures = 0
                    self.last_valid_frame = frame
                    self.stats['frames_read'] += 1
                    
                    # 버퍼가 가득 찬 경우 오래된 프레임 제거
                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()
                        except:
                            pass
                    
                    self.frame_buffer.put(frame)
                    
                    # HLS 점프 감지
                    if self.jump_detector.detect_jump(frame):
                        print("🔥 HLS 점프 감지! 스트림 연속성 문제")
                        self.stats['stream_reconnects'] += 1
                    
                else:
                    # 실패: 카운트 증가
                    consecutive_failures += 1
                    
                    # 너무 많이 실패하면 재연결 필요
                    if consecutive_failures >= max_failures:
                        print("⚠️ 프레임 읽기 실패 지속, 스트림 상태 확인")
                        consecutive_failures = 0
                        
                        # 백업 스트림 시도
                        if self.backup_cap:
                            print("🔄 백업 스트림으로 전환")
                            self.current_cap, self.backup_cap = self.backup_cap, self.current_cap
                        
                        time.sleep(0.5)
                
                # CPU 부하 방지
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ 프레임 읽기 오류: {e}")
                time.sleep(0.1)
    
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        버퍼에서 프레임 가져오기
        버퍼가 비어있으면 마지막 유효 프레임 반환 (끊김 방지)
        """
        try:
            # 버퍼에서 프레임 가져오기
            frame = self.frame_buffer.get(timeout=timeout)
            return frame
            
        except queue.Empty:
            # 버퍼가 비어있으면 마지막 프레임 반환
            self.stats['buffer_underruns'] += 1
            
            if self.last_valid_frame is not None:
                # 마지막 프레임 복사본 반환
                return self.last_valid_frame.copy()
            
            return None
    
    def get_stream_health(self) -> Dict:
        """스트림 상태 정보"""
        buffer_size = self.frame_buffer.qsize()
        
        health = {
            'is_connected': self.current_cap is not None and self.current_cap.isOpened(),
            'buffer_size': buffer_size,
            'buffer_health': 'good' if buffer_size > 20 else 'low' if buffer_size > 5 else 'critical',
            'url_updates': self.stats['url_updates'],
            'reconnects': self.stats['stream_reconnects'],
            'total_frames': self.stats['frames_read'],
            'buffer_underruns': self.stats['buffer_underruns'],
            'time_since_update': time.time() - self.stats['last_update_time'] if self.stats['last_update_time'] > 0 else 0
        }
        
        return health
    
    def stop(self):
        """스트림 매니저 중지"""
        print("🛑 스트림 매니저 중지 중...")
        
        self.running = False
        
        # 스레드 종료 대기
        if self.url_updater_thread:
            self.url_updater_thread.join(timeout=2)
        
        if self.frame_reader_thread:
            self.frame_reader_thread.join(timeout=2)
        
        # 리소스 정리
        if self.current_cap:
            self.current_cap.release()
        
        if self.backup_cap:
            self.backup_cap.release()
        
        # 버퍼 비우기
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except:
                break
        
        print("✅ 스트림 매니저 중지 완료")


class HLSJumpDetector:
    """HLS 세그먼트 점프 감지기"""
    
    def __init__(self, window_size: int = 30):
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = None
        self.jump_count = 0
    
    def detect_jump(self, frame: np.ndarray) -> bool:
        """프레임 간 시간 간격으로 점프 감지"""
        current_time = time.time()
        
        if self.last_frame_time is not None:
            interval = current_time - self.last_frame_time
            self.frame_times.append(interval)
            
            if len(self.frame_times) > 10:
                avg_interval = sum(self.frame_times) / len(self.frame_times)
                
                # 평균의 3배 이상 간격이면 점프
                if interval > avg_interval * 3 and interval > 0.5:
                    self.jump_count += 1
                    print(f"⚡ 점프 감지 #{self.jump_count}: {interval:.2f}초 (평균: {avg_interval:.3f}초)")
                    self.last_frame_time = current_time
                    return True
        
        self.last_frame_time = current_time
        return False


# 통합 함수: 기존 시스템과 연동
def integrate_with_existing_system(system):
    """기존 IntegratedHandoverSystem과 통합"""
    
    # 환경 변수 로드
    load_dotenv()
    api_key = os.getenv("ITS_API_KEY")
    
    if not api_key:
        print("❌ ITS_API_KEY가 설정되지 않았습니다")
        return False
    
    # HLS 스트림 매니저 생성
    stream_manager = HLSStreamManager(api_key, update_interval=20)
    
    # 기존 시스템에 통합
    system.stream_manager = stream_manager
    
    # 기존 get_frames 메서드 오버라이드
    original_get_frames = system.get_frames
    
    def new_get_frames():
        """개선된 프레임 가져오기"""
        if hasattr(system, 'stream_manager'):
            frame = system.stream_manager.get_frame()
            if frame is not None:
                return frame, None
        
        # 폴백: 기존 방식
        return original_get_frames()
    
    system.get_frames = new_get_frames
    
    # 기존 start_camera 메서드 오버라이드
    original_start_camera = system.start_camera
    
    def new_start_camera(cctv_name: str, stream_url: str) -> bool:
        """개선된 카메라 시작"""
        # 스트림 매니저로 시작
        if hasattr(system, 'stream_manager'):
            success = system.stream_manager.start(cctv_name, stream_url)
            if success:
                # 기존 시스템 설정도 수행
                return original_start_camera(cctv_name, stream_url)
        
        return False
    
    system.start_camera = new_start_camera
    
    print("✅ HLS 스트림 매니저 통합 완료")
    return True


# 사용 예시
if __name__ == "__main__":
    load_dotenv()
    
    # 단독 테스트
    api_key = os.getenv("ITS_API_KEY")
    stream_url = os.getenv("CURRENT_CCTV_URL")
    cctv_name = os.getenv("CURRENT_CCTV_NAME", "죽평")
    
    if not api_key or not stream_url:
        print("❌ 환경 변수를 설정하세요")
        exit(1)
    
    # 스트림 매니저 테스트
    manager = HLSStreamManager(api_key)
    
    if manager.start(cctv_name, stream_url):
        print("📹 스트림 테스트 시작 (30초간)")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 30:
            frame = manager.get_frame()
            
            if frame is not None:
                frame_count += 1
                
                # 10프레임마다 상태 출력
                if frame_count % 10 == 0:
                    health = manager.get_stream_health()
                    print(f"프레임 {frame_count}: 버퍼={health['buffer_size']}, "
                          f"상태={health['buffer_health']}, "
                          f"갱신={health['url_updates']}회")
                
                # 화면 표시 (선택사항)
                # cv2.imshow("Stream Test", frame)
                # if cv2.waitKey(1) & 0xFF == 27:  # ESC
                #     break
            
            time.sleep(0.03)  # ~30 FPS
        
        manager.stop()
        
        # 최종 통계
        print(f"\n📊 테스트 완료:")
        print(f"  총 프레임: {frame_count}")
        print(f"  평균 FPS: {frame_count / 30:.1f}")
        
        final_health = manager.get_stream_health()
        print(f"  URL 갱신: {final_health['url_updates']}회")
        print(f"  재연결: {final_health['reconnects']}회")
        print(f"  버퍼 언더런: {final_health['buffer_underruns']}회")