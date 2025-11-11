# pipeline.py 상단 (import 아래)
from collections import deque
import numpy as np

class TrackHistory:
    """트랙별로 최근 프레임의 crop 저장 + 히스토리 기반 검색"""
    def __init__(self, maxlen=30):
        self.history = {}  # {tid: deque([(crop, bbox, frame_idx), ...])}
        self.maxlen = maxlen
    
    def add(self, tid, frame, bbox, frame_idx):
        """트랙에 프레임 추가"""
        if tid not in self.history:
            self.history[tid] = deque(maxlen=self.maxlen)
        
        crop = self._crop_safe(frame, bbox)
        if crop is not None:
            self.history[tid].append((crop.copy(), bbox, frame_idx))
    
    def get_history(self, tid, count=None):
        """트랙의 히스토리 반환 (최근부터)"""
        if tid not in self.history:
            return []
        
        history = list(self.history[tid])
        if count is not None:
            history = history[-count:]
        
        return history
    
    def clear(self, tid=None):
        """히스토리 삭제"""
        if tid is None:
            self.history.clear()
        elif tid in self.history:
            del self.history[tid]
    
    def cleanup_old_tracks(self, current_tids):
        """현재 없는 트랙 정리 (메모리 절약)"""
        dead_tids = [tid for tid in self.history.keys() if tid not in current_tids]
        for tid in dead_tids:
            del self.history[tid]
        return len(dead_tids)
    
    # ✅ 새로 추가: 클릭 위치 기반 트랙 검색
    def find_closest_track(self, click_x, click_y, max_frames_back=15, max_distance=150, verbose=True):
        """
        최근 N 프레임의 히스토리에서 클릭 위치에 가장 가까운 트랙 찾기
        
        Args:
            click_x, click_y: 클릭 좌표
            max_frames_back: 최대 몇 프레임 전까지 검색 (기본 15)
            max_distance: 최대 허용 거리 픽셀 (기본 150)
            verbose: 로그 출력 여부
        
        Returns:
            dict: {
                'tid': 트랙 ID,
                'distance': 거리,
                'frame_idx': 프레임 인덱스,
                'bbox': 해당 프레임의 bbox,
                'checked_frames': 검색한 총 프레임 수
            } or None
        """
        if not self.history:
            if verbose:
                print("[HISTORY_SEARCH] 히스토리가 비어있음")
            return None
        
        best_result = None
        best_distance = max_distance
        total_checked = 0
        
        # 모든 트랙의 최근 프레임 검색
        for tid, frames in self.history.items():
            if not frames or len(frames) == 0:
                continue
            
            # 최근 max_frames_back 프레임만 검색
            recent_count = min(len(frames), max_frames_back)
            recent_frames = list(frames)[-recent_count:]
            
            for crop, bbox, fidx in recent_frames:
                total_checked += 1
                x1, y1, x2, y2 = bbox
                
                # bbox 중심점 계산
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # 유클리드 거리 계산
                distance = np.sqrt((cx - click_x)**2 + (cy - click_y)**2)
                
                # 더 가까운 트랙 발견
                if distance < best_distance:
                    best_distance = distance
                    best_result = {
                        'tid': tid,
                        'distance': distance,
                        'frame_idx': fidx,
                        'bbox': bbox,
                        'checked_frames': total_checked
                    }
        
        if best_result and verbose:
            print(f"[HISTORY_SEARCH] ✅ {total_checked}개 프레임 검색 → "
                  f"ID {best_result['tid']} 발견 "
                  f"(거리: {best_result['distance']:.1f}px, 프레임: {best_result['frame_idx']})")
        elif verbose:
            print(f"[HISTORY_SEARCH] ❌ {total_checked}개 프레임 검색했으나 {max_distance}px 이내 트랙 없음")
        
        return best_result
    
    # ✅ 새로 추가: bbox 영역 기반 트랙 검색 (클릭 대신 영역으로)
    def find_tracks_in_region(self, x1, y1, x2, y2, max_frames_back=10):
        """
        특정 영역 내에 있는 모든 트랙 찾기 (최근 N 프레임)
        
        Returns:
            list of dict: [{'tid': ..., 'bbox': ..., 'frame_idx': ...}, ...]
        """
        results = []
        checked = set()  # 중복 방지
        
        for tid, frames in self.history.items():
            if not frames:
                continue
            
            recent_count = min(len(frames), max_frames_back)
            recent_frames = list(frames)[-recent_count:]
            
            for crop, bbox, fidx in recent_frames:
                if tid in checked:
                    continue
                
                bx1, by1, bx2, by2 = bbox
                # bbox 중심이 영역 내에 있는지 확인
                cx = (bx1 + bx2) / 2
                cy = (by1 + by2) / 2
                
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    results.append({
                        'tid': tid,
                        'bbox': bbox,
                        'frame_idx': fidx
                    })
                    checked.add(tid)
                    break
        
        return results
    
    # ✅ 새로 추가: 통계 정보
    def get_stats(self):
        """히스토리 통계 반환"""
        total_tracks = len(self.history)
        total_frames = sum(len(frames) for frames in self.history.values())
        avg_frames = total_frames / total_tracks if total_tracks > 0 else 0
        
        return {
            'total_tracks': total_tracks,
            'total_frames': total_frames,
            'avg_frames_per_track': avg_frames,
            'maxlen': self.maxlen
        }
    
    @staticmethod
    def _crop_safe(img, bbox, pad=2):
        """안전하게 crop"""
        if img is None:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return img[y1:y2, x1:x2]