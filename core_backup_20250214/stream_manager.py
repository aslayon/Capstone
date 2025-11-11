"""
범용 스트림 매니저 - 스마트 URL 갱신 전략 포함
"""

import cv2
import numpy as np
import requests
import time
import threading
import queue
from typing import Dict, Optional
from pathlib import Path
import os


class UniversalStreamManager:
    """HLS, MP4, RTSP 등 모든 스트림 소스를 지원하는 범용 매니저"""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 update_interval: int = 300,
                 smart_update: bool = True):
        """
        Args:
            api_key: ITS API 키 (HLS 스트림용, None이면 API 미사용)
            update_interval: HLS URL 갱신 주기 (초) 
                - 권장: 300 (5분) - 안정적
                - 최소: 240 (4분) - URL 만료 직전
                - 최대: 600 (10분) - 연결 끊김 위험
            smart_update: 스마트 갱신 활성화
                - True: 스트림 품질 모니터링, 문제 시에만 즉시 갱신
                - False: 고정 주기로만 갱신
        """
        self.api_key = api_key
        self.update_interval = update_interval
        self.smart_update = smart_update
        
        # 스트림 타입 감지
        self.source_type = None
        self.is_file = False
        self.file_loop = True
        
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
        
        # 스마트 갱신용 품질 모니터링
        self.quality_monitor = {
            'consecutive_failures': 0,
            'last_success_time': time.time(),
            'health_check_interval': 30,  # 30초마다 건강 체크
        }
        
        # 통계
        self.stats = {
            'url_updates': 0,
            'stream_reconnects': 0,
            'frames_read': 0,
            'buffer_underruns': 0,
            'last_update_time': 0,
            'source_type': 'unknown',
            'api_calls': 0,
            'api_failures': 0,
            'emergency_updates': 0  # 긴급 갱신 횟수
        }
        
        print("🚀 범용 스트림 매니저 초기화")
        print(f"   갱신 주기: {update_interval}초 ({update_interval/60:.1f}분)")
        print(f"   스마트 갱신: {'활성화' if smart_update else '비활성화'}")
    
    def _detect_source_type(self, url: str) -> str:
        """URL/경로로부터 소스 타입 자동 감지"""
        url_lower = url.lower()
        
        if os.path.isfile(url):
            ext = Path(url).suffix.lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']:
                return 'mp4'
        
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
        
        return 'hls'
    
    def start(self, cctv_name: str, source: str, loop: bool = True) -> bool:
        """스트림/파일 시작"""
        self.cctv_name = cctv_name
        self.current_url = source
        self.file_loop = loop
        self.running = True
        
        self.source_type = self._detect_source_type(source)
        self.is_file = self.source_type == 'mp4'
        self.stats['source_type'] = self.source_type
        
        print(f"📡 스트림 시작: {cctv_name}")
        print(f"   타입: {self.source_type}")
        print(f"   소스: {source[:80]}..." if len(source) > 80 else f"   소스: {source}")
        
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
            if source.isdigit():
                source = int(source)
            
            if self.source_type in ['hls', 'rtsp', 'rtmp']:
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                print(f"❌ 소스 열기 실패: {source[:80]}...")
                
                if self.source_type == 'hls':
                    print("   💡 HLS 문제 해결 팁:")
                    print("      1. URL 만료 (5분 유효) - API로 갱신 필요")
                    print("      2. API 사용량 초과 - 내일 다시 시도")
                    print("      3. OpenCV FFmpeg 미설치")
                
                return False
            
            if self.source_type == 'hls':
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                try:
                    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                except:
                    pass
                    
            elif self.source_type == 'mp4':
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                print(f"   📹 파일: {total_frames}프레임, {fps:.1f}FPS, {duration:.1f}초")
            
            # 첫 프레임 테스트
            ret, frame = False, None
            for attempt in range(3):
                ret, frame = cap.read()
                if ret:
                    break
                time.sleep(0.1)
            
            if not ret or frame is None:
                cap.release()
                print("❌ 첫 프레임 읽기 실패")
                return False
            
            if self.current_cap:
                self.backup_cap = self.current_cap
            
            self.current_cap = cap
            self.last_valid_frame = frame
            
            # 품질 모니터 리셋
            self.quality_monitor['consecutive_failures'] = 0
            self.quality_monitor['last_success_time'] = time.time()
            
            print(f"✅ 스트림 연결 성공 ({frame.shape[1]}x{frame.shape[0]})")
            return True
            
        except Exception as e:
            print(f"❌ 스트림 연결 오류: {e}")
            return False
    
    def _should_emergency_update(self) -> bool:
        """
        긴급 갱신이 필요한지 판단 (스마트 갱신)
        
        다음 상황에서 즉시 갱신:
        1. 연속 실패 10회 이상
        2. 마지막 성공 후 1분 이상 실패
        3. 버퍼 완전 고갈
        """
        if not self.smart_update:
            return False
        
        # 1. 연속 실패 체크
        if self.quality_monitor['consecutive_failures'] >= 10:
            print("   ⚠️  연속 실패 10회 - 긴급 갱신 필요")
            return True
        
        # 2. 장시간 실패 체크
        time_since_success = time.time() - self.quality_monitor['last_success_time']
        if time_since_success > 60:  # 1분
            print(f"   ⚠️  {time_since_success:.0f}초간 실패 - 긴급 갱신 필요")
            return True
        
        # 3. 버퍼 고갈 체크
        if self.frame_buffer.qsize() == 0 and self.last_valid_frame is None:
            print("   ⚠️  버퍼 완전 고갈 - 긴급 갱신 필요")
            return True
        
        return False
    
    def _url_update_loop(self):
        """HLS URL 주기적 갱신 스레드 (스마트 갱신 포함)"""
        if self.source_type != 'hls' or not self.api_key:
            return
        
        print(f"🔄 URL 갱신 스레드 시작")
        
        # 첫 갱신은 4분 후 (URL 만료 직전)
        next_update_time = time.time() + 240
        
        while self.running:
            try:
                now = time.time()
                
                # 정기 갱신 시간 체크
                time_until_update = next_update_time - now
                
                # 긴급 갱신 체크 (30초마다)
                if self.smart_update and time_until_update > 30:
                    time.sleep(30)
                    
                    if self._should_emergency_update():
                        print("🚨 긴급 URL 갱신 시도")
                        self.stats['emergency_updates'] += 1
                        
                        new_url = self._fetch_new_url()
                        if new_url and new_url != self.current_url:
                            if self._connect_stream(new_url):
                                self.current_url = new_url
                                self.stats['url_updates'] += 1
                                self.stats['last_update_time'] = time.time()
                                next_update_time = time.time() + self.update_interval
                        
                        continue
                
                # 정기 갱신 대기
                if time_until_update > 0:
                    time.sleep(min(time_until_update, 30))
                    continue
                
                # 정기 갱신 실행
                print(f"⏰ 정기 URL 갱신 (#{self.stats['url_updates'] + 1})")
                
                new_url = self._fetch_new_url()
                
                if new_url and new_url != self.current_url:
                    print(f"🔄 새 URL 획득")
                    
                    if self._connect_stream(new_url):
                        self.current_url = new_url
                        self.stats['url_updates'] += 1
                        self.stats['last_update_time'] = time.time()
                        
                        if self.backup_cap:
                            self.backup_cap.release()
                            self.backup_cap = None
                
                # 다음 갱신 시간 설정
                next_update_time = time.time() + self.update_interval
                
            except Exception as e:
                print(f"❌ URL 갱신 오류: {e}")
                time.sleep(5)
    
    def _fetch_new_url(self) -> Optional[str]:
        """HLS API에서 새 URL 가져오기 (전국 범위)"""
        if not self.api_key:
            return None
        
        self.stats['api_calls'] += 1
        
        try:
            url = "https://openapi.its.go.kr:9443/cctvInfo"
            
            # 전국 범위
            params = {
                "apiKey": self.api_key,
                "type": "all",
                "cctvType": "1",
                "minX": "124",
                "maxX": "132",
                "minY": "33",
                "maxY": "43",
                "getType": "json"
            }
            
            print(f"   📡 API 호출 중... (#{self.stats['api_calls']})")
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 429:
                print(f"   ⚠️  API 사용량 초과 (429)")
                print(f"   💡 오늘은 더 이상 갱신 불가 - 기존 URL 유지")
                self.stats['api_failures'] += 1
                return None
            
            if response.status_code != 200:
                print(f"   ⚠️  API 오류: HTTP {response.status_code}")
                self.stats['api_failures'] += 1
                return None
            
            data = response.json()
            
            if "response" not in data or "data" not in data["response"]:
                if "response" in data and "errMsg" in data["response"]:
                    err_msg = data["response"]["errMsg"]
                    print(f"   ⚠️  API 오류: {err_msg}")
                self.stats['api_failures'] += 1
                return None
            
            cctv_list = data["response"]["data"]
            print(f"   ✅ {len(cctv_list)}개 CCTV 조회 성공")
            
            for cctv in cctv_list:
                cctv_name = cctv.get("cctvname", "")
                if self.cctv_name in cctv_name:
                    new_url = cctv.get("cctvurl")
                    if new_url:
                        print(f"   🎯 '{self.cctv_name}' 매칭 성공")
                        return new_url
            
            print(f"   ⚠️  '{self.cctv_name}' 매칭 실패")
            return None
            
        except requests.exceptions.Timeout:
            print(f"   ⚠️  API 타임아웃")
            self.stats['api_failures'] += 1
            return None
        except Exception as e:
            print(f"   ⚠️  API 호출 실패: {e}")
            self.stats['api_failures'] += 1
            return None
    
    def switch_to(self, cctv_name: str, new_source: str) -> bool:
        """실행 중 부드러운 전환"""
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
                
                ret, frame = self.current_cap.read()
                
                if ret and frame is not None:
                    consecutive_failures = 0
                    self.last_valid_frame = frame
                    self.stats['frames_read'] += 1
                    
                    # 품질 모니터 업데이트
                    self.quality_monitor['consecutive_failures'] = 0
                    self.quality_monitor['last_success_time'] = time.time()
                    
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
                            print("🔁 파일 끝, 처음부터 재생")
                            self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            print("⏹️ 파일 재생 완료")
                            break
                    
                    # 스트림 실패 처리
                    consecutive_failures += 1
                    self.quality_monitor['consecutive_failures'] += 1
                    
                    if consecutive_failures >= max_failures:
                        print("⚠️ 프레임 읽기 실패 지속")
                        consecutive_failures = 0
                        
                        if self.backup_cap:
                            print("🔄 백업 스트림으로 전환")
                            self.current_cap, self.backup_cap = self.backup_cap, self.current_cap
                        
                        time.sleep(0.5)
                
                time.sleep(0.03)
                
            except Exception as e:
                print(f"❌ 프레임 읽기 오류: {e}")
                time.sleep(0.1)
    
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """버퍼에서 프레임 가져오기"""
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
            'emergency_updates': self.stats['emergency_updates'],
            'reconnects': self.stats['stream_reconnects'],
            'total_frames': self.stats['frames_read'],
            'buffer_underruns': self.stats['buffer_underruns'],
            'api_calls': self.stats['api_calls'],
            'api_failures': self.stats['api_failures'],
            'consecutive_failures': self.quality_monitor['consecutive_failures'],
            'time_since_update': time.time() - self.stats['last_update_time'] if self.stats['last_update_time'] > 0 else 0
        }
        
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
        
        if self.url_updater_thread:
            self.url_updater_thread.join(timeout=2)
        
        if self.frame_reader_thread:
            self.frame_reader_thread.join(timeout=2)
        
        if self.current_cap:
            self.current_cap.release()
        
        if self.backup_cap:
            self.backup_cap.release()
        
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except:
                break
        
        print("✅ 스트림 매니저 중지 완료")
        
        # 통계 출력
        if self.stats['api_calls'] > 0:
            print(f"\n📊 최종 통계:")
            print(f"   총 프레임: {self.stats['frames_read']:,}개")
            print(f"   API 호출: {self.stats['api_calls']}회")
            print(f"   - 정기 갱신: {self.stats['url_updates']}회")
            print(f"   - 긴급 갱신: {self.stats['emergency_updates']}회")
            print(f"   - 실패: {self.stats['api_failures']}회")
            if self.stats['api_calls'] > 0:
                print(f"   성공률: {(1 - self.stats['api_failures']/self.stats['api_calls'])*100:.1f}%")


# 기존 HLSStreamManager와 호환성 유지
HLSStreamManager = UniversalStreamManager