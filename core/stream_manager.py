"""
범용 스트림 매니저 - HLS 스트림과 MP4 파일 모두 지원
파일명: universal_stream_manager.py
위치: core/universal_stream_manager.py

사용법:
    # HLS 스트림
    manager = UniversalStreamManager(api_key="your_key")
    manager.start("카메라명", "http://example.com/stream.m3u8")
    
    # MP4 파일
    manager = UniversalStreamManager()
    manager.start("비디오1", "/path/to/video.mp4")
    
    # RTSP 스트림
    manager = UniversalStreamManager()
    manager.start("RTSP카메라", "rtsp://192.168.1.100:554/stream")
"""

import cv2
import numpy as np
import requests
import time
import threading
import queue
from collections import deque
from typing import Dict, Optional
from pathlib import Path
import os
from dotenv import load_dotenv


class UniversalStreamManager:
    """HLS, MP4, RTSP 등 모든 스트림 소스를 지원하는 범용 매니저"""
    
    def __init__(self, api_key: Optional[str] = None, update_interval: int = 20):
        """
        Args:
            api_key: ITS API 키 (HLS 스트림용, None이면 API 미사용)
            update_interval: HLS URL 갱신 주기 (초)
        """
        self.api_key = api_key
        self.update_interval = update_interval
        
        # 스트림 타입 감지
        self.source_type = None  # "hls", "mp4", "rtsp", "webcam"
        self.is_file = False
        self.file_loop = True  # MP4 반복 재생 여부
        
        # 스트림 관리
        self.current_url = None
        self.current_cap = None
        self.backup_cap = None
        
        # 프레임 버퍼
        self.frame_buffer = queue.Queue(maxsize=60)
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
            'last_update_time': 0,
            'source_type': 'unknown'
        }
        
        print("🚀 범용 스트림 매니저 초기화")
    
    def _detect_source_type(self, url: str) -> str:
        """URL/경로로부터 소스 타입 자동 감지"""
        url_lower = url.lower()
        
        # 파일 경로 체크
        if os.path.isfile(url):
            ext = Path(url).suffix.lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']:
                return 'mp4'
        
        # URL 프로토콜 체크
        if url_lower.startswith('rtsp://'):
            return 'rtsp'
        elif url_lower.startswith('rtmp://'):
            return 'rtmp'
        elif url_lower.endswith('.m3u8') or 'playlist.m3u8' in url_lower:
            return 'hls'
        elif any(ext in url_lower for ext in ['.mp4', '.avi', '.mov']):
            return 'mp4'
        elif url.isdigit():
            return 'webcam'
        
        # 기본값: HLS로 가정
        return 'hls'
    
    def start(self, cctv_name: str, source: str, loop: bool = True) -> bool:
        """
        스트림/파일 시작
        
        Args:
            cctv_name: 카메라/소스 이름
            source: URL 또는 파일 경로
            loop: MP4 파일 반복 재생 여부
        """
        self.cctv_name = cctv_name
        self.current_url = source
        self.file_loop = loop
        self.running = True
        
        # 소스 타입 감지
        self.source_type = self._detect_source_type(source)
        self.is_file = self.source_type == 'mp4'
        self.stats['source_type'] = self.source_type
        
        print(f"📡 스트림 시작: {cctv_name}")
        print(f"   타입: {self.source_type}")
        print(f"   소스: {source}")
        
        # 초기 연결
        if not self._connect_stream(source):
            print("❌ 초기 스트림 연결 실패")
            return False
        
        # HLS만 URL 갱신 스레드 사용
        if self.source_type == 'hls' and self.api_key:
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
    
    def _connect_stream(self, source: str) -> bool:
        """스트림/파일 연결"""
        try:
            # 파일 경로면 정수로 변환 시도 (웹캠)
            if source.isdigit():
                source = int(source)
            
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                print(f"❌ 소스 열기 실패: {source}")
                return False
            
            # 설정 최적화
            if self.source_type == 'hls':
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # HLS: 최소 버퍼
            elif self.source_type == 'mp4':
                # MP4: 기본 설정 사용
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                print(f"   파일 정보: {total_frames} 프레임, {fps:.1f} FPS, {duration:.1f}초")
            
            # 첫 프레임 테스트
            ret, frame = cap.read()
            if not ret:
                cap.release()
                print("❌ 첫 프레임 읽기 실패")
                return False
            
            # 백업으로 현재 cap 저장
            if self.current_cap:
                self.backup_cap = self.current_cap
            
            self.current_cap = cap
            self.last_valid_frame = frame
            
            print(f"✅ 스트림 연결 성공 ({frame.shape[1]}x{frame.shape[0]})")
            return True
            
        except Exception as e:
            print(f"❌ 스트림 연결 오류: {e}")
            return False
    
    def _url_update_loop(self):
        """HLS URL 주기적 갱신 스레드"""
        if self.source_type != 'hls' or not self.api_key:
            return
        
        print("🔄 URL 갱신 스레드 시작")
        
        while self.running:
            try:
                # 대기
                if self.stats['url_updates'] > 0:
                    time.sleep(self.update_interval)
                
                # 새 URL 가져오기
                new_url = self._fetch_new_url()
                
                if new_url and new_url != self.current_url:
                    print(f"🔄 새 URL 획득 (갱신 #{self.stats['url_updates'] + 1})")
                    
                    if self._connect_stream(new_url):
                        self.current_url = new_url
                        self.stats['url_updates'] += 1
                        self.stats['last_update_time'] = time.time()
                        
                        if self.backup_cap:
                            self.backup_cap.release()
                            self.backup_cap = None
                
            except Exception as e:
                print(f"❌ URL 갱신 오류: {e}")
                time.sleep(5)
    
    def _fetch_new_url(self) -> Optional[str]:
        """HLS API에서 새 URL 가져오기"""
        if not self.api_key:
            return None
        
        try:
            url = "https://openapi.its.go.kr:9443/cctvInfo"
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
    
    def switch_to(self, cctv_name: str, new_source: str) -> bool:
        """
        실행 중 부드러운 전환
        
        Args:
            cctv_name: 새 카메라/소스 이름
            new_source: 새 URL/파일 경로
        """
        print(f"🔄 전환 중: {self.cctv_name} -> {cctv_name}")
        
        self.cctv_name = cctv_name
        self.source_type = self._detect_source_type(new_source)
        self.is_file = self.source_type == 'mp4'
        self.stats['source_type'] = self.source_type
        
        ok = self._connect_stream(new_source)
        
        if ok:
            self.current_url = new_source
            self.stats['url_updates'] += 1
            self.stats['last_update_time'] = time.time()
            print(f"✅ 전환 완료: {self.source_type}")
        else:
            print("[UniversalStreamManager] switch_to() 실패")
        
        return ok
    
    def _frame_reader_loop(self):
        """프레임 읽기 스레드"""
        print("📹 프레임 읽기 스레드 시작")
        
        consecutive_failures = 0
        max_failures = 30
        
        while self.running:
            try:
                if not self.current_cap:
                    time.sleep(0.1)
                    continue
                
                # 프레임 읽기
                ret, frame = self.current_cap.read()
                
                if ret:
                    consecutive_failures = 0
                    self.last_valid_frame = frame
                    self.stats['frames_read'] += 1
                    
                    # 버퍼 관리
                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()
                        except:
                            pass
                    
                    self.frame_buffer.put(frame)
                    
                else:
                    # MP4 파일 끝 처리
                    if self.is_file:
                        if self.file_loop:
                            # 반복 재생
                            print("🔁 파일 끝, 처음부터 재생")
                            self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            print("⏹️ 파일 재생 완료")
                            break
                    
                    # 스트림 실패 처리
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_failures:
                        print("⚠️ 프레임 읽기 실패 지속")
                        consecutive_failures = 0
                        
                        # 백업 스트림 시도
                        if self.backup_cap:
                            print("🔄 백업 스트림으로 전환")
                            self.current_cap, self.backup_cap = self.backup_cap, self.current_cap
                        
                        time.sleep(0.5)
                
                # CPU 부하 방지
                time.sleep(0.03)
                
            except Exception as e:
                print(f"❌ 프레임 읽기 오류: {e}")
                time.sleep(0.1)
    
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        버퍼에서 프레임 가져오기
        """
        try:
            frame = self.frame_buffer.get(timeout=timeout)
            return frame
            
        except queue.Empty:
            self.stats['buffer_underruns'] += 1
            
            if self.last_valid_frame is not None:
                return self.last_valid_frame.copy()
            
            return None
    
    def get_stream_health(self) -> Dict:
        """스트림 상태 정보"""
        buffer_size = self.frame_buffer.qsize()
        
        health = {
            'source_type': self.source_type,
            'is_connected': self.current_cap is not None and self.current_cap.isOpened(),
            'buffer_size': buffer_size,
            'buffer_health': 'good' if buffer_size > 20 else 'low' if buffer_size > 5 else 'critical',
            'url_updates': self.stats['url_updates'],
            'reconnects': self.stats['stream_reconnects'],
            'total_frames': self.stats['frames_read'],
            'buffer_underruns': self.stats['buffer_underruns'],
            'time_since_update': time.time() - self.stats['last_update_time'] if self.stats['last_update_time'] > 0 else 0
        }
        
        # MP4 파일 진행률 추가
        if self.is_file and self.current_cap:
            current_frame = self.current_cap.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = self.current_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames > 0:
                health['progress'] = f"{current_frame:.0f}/{total_frames:.0f} ({current_frame/total_frames*100:.1f}%)"
        
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


# ============================================================
# 기존 HLSStreamManager와 호환성 유지를 위한 별칭
# ============================================================
HLSStreamManager = UniversalStreamManager


# ============================================================
# 사용 예시
# ============================================================
if __name__ == "__main__":
    load_dotenv()
    
    # 예시 1: HLS 스트림
    print("\n" + "="*60)
    print("예시 1: HLS 스트림")
    print("="*60)
    
    api_key = os.getenv("ITS_API_KEY")
    stream_url = os.getenv("CURRENT_CCTV_URL")
    
    if api_key and stream_url:
        manager = UniversalStreamManager(api_key=api_key)
        if manager.start("죽평", stream_url):
            print("\n📹 5초간 스트림 테스트...")
            
            for i in range(150):  # 5초 (30fps)
                frame = manager.get_frame()
                if frame is not None and i % 30 == 0:
                    health = manager.get_stream_health()
                    print(f"  {i//30 + 1}초: 버퍼={health['buffer_size']}, 타입={health['source_type']}")
                time.sleep(0.03)
            
            manager.stop()
    
    # 예시 2: MP4 파일
    print("\n" + "="*60)
    print("예시 2: MP4 파일")
    print("="*60)
    
    # 테스트용 MP4 파일이 있다면
    test_video = "test_video.mp4"
    if os.path.exists(test_video):
        manager = UniversalStreamManager()
        if manager.start("테스트비디오", test_video, loop=True):
            print("\n📹 3초간 파일 재생 테스트...")
            
            for i in range(90):
                frame = manager.get_frame()
                if frame is not None and i % 30 == 0:
                    health = manager.get_stream_health()
                    print(f"  {i//30 + 1}초: {health.get('progress', 'N/A')}")
                time.sleep(0.03)
            
            manager.stop()
    else:
        print(f"⚠️  테스트 파일 없음: {test_video}")
    
    # 예시 3: 웹캠
    print("\n" + "="*60)
    print("예시 3: 웹캠 (선택사항)")
    print("="*60)
    print("웹캠 테스트를 원하시면 주석을 해제하세요")
    
    # manager = UniversalStreamManager()
    # if manager.start("웹캠", "0"):  # "0"은 기본 웹캠
    #     print("\n📹 3초간 웹캠 테스트...")
    #     for i in range(90):
    #         frame = manager.get_frame()
    #         if frame is not None and i % 30 == 0:
    #             print(f"  {i//30 + 1}초: 프레임 {frame.shape}")
    #         time.sleep(0.03)
    #     manager.stop()