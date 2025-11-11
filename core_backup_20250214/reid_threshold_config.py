# reid_threshold_config.py
"""
실제 로그 분석 기반 최적화된 ReID 임계값 설정
- 분석 날짜: 2025-11-02
- 테스트 차량: 2대
- 결과: 임계값 0.70으로 두 차량 모두 매칭 성공
"""

# ===== 기본 임계값 =====
REID_THRESH_SAME = 0.55   # 동일 카메라 (같은 세그먼트)
REID_THRESH_OTHER = 0.70  # 다른 카메라 (카메라 전환)

# ===== 카메라 전환별 임계값 (선택적) =====
# 로그 분석 결과 중앙(C) 카메라에서 d_mean이 약간 높은 경향
# 필요시 카메라 조합별로 다른 임계값 사용 가능
THRESH_BY_TRANSITION = {
    # (from_seg, to_seg): threshold
    ("L", "L"): 0.55,  # 같은 카메라
    ("C", "C"): 0.55,
    ("R", "R"): 0.55,
    
    ("L", "C"): 0.70,  # 좌 → 중앙
    ("C", "R"): 0.70,  # 중앙 → 우
    ("L", "R"): 0.75,  # 좌 → 우 (직접 전환, 드물지만 여유있게)
    
    ("R", "C"): 0.70,  # 역방향
    ("C", "L"): 0.70,
    ("R", "L"): 0.75,
}

# ===== 야간 보정 (선택적) =====
NIGHT_THRESH_BONUS = 0.05  # 야간엔 임계값에 +0.05 여유

# ===== 매칭 신뢰도 레벨 =====
CONFIDENCE_HIGH = 0.60    # d_mean < 0.60: 매우 확실
CONFIDENCE_MEDIUM = 0.70  # d_mean < 0.70: 확실
CONFIDENCE_LOW = 0.80     # d_mean < 0.80: 불확실 (추가 검증 필요)


# ===== 사용 예시 =====
def get_threshold(from_seg, to_seg, is_night=False):
    """
    상황에 맞는 임계값 반환
    
    Args:
        from_seg: 이전 세그먼트 ("L", "C", "R")
        to_seg: 현재 세그먼트 ("L", "C", "R")
        is_night: 야간 여부
    
    Returns:
        float: 적용할 임계값
    """
    # 같은 세그먼트
    if from_seg == to_seg:
        thresh = REID_THRESH_SAME
    else:
        # 전환별 임계값 사용 (있으면)
        thresh = THRESH_BY_TRANSITION.get(
            (from_seg, to_seg), 
            REID_THRESH_OTHER  # 없으면 기본값
        )
    
    # 야간 보정
    if is_night:
        thresh += NIGHT_THRESH_BONUS
    
    return thresh


def evaluate_match_confidence(d_mean):
    """
    매칭 신뢰도 평가
    
    Args:
        d_mean: 바타차야 거리
    
    Returns:
        str: "HIGH", "MEDIUM", "LOW", "REJECT"
    """
    if d_mean < CONFIDENCE_HIGH:
        return "HIGH"
    elif d_mean < CONFIDENCE_MEDIUM:
        return "MEDIUM"
    elif d_mean < CONFIDENCE_LOW:
        return "LOW"
    else:
        return "REJECT"


# ===== 통계 기반 동적 임계값 (고급) =====
class AdaptiveThreshold:
    """
    실시간 통계로 임계값 자동 조정
    """
    def __init__(self):
        self.positive_samples = []  # 같은 차량의 d_mean 기록
        self.negative_samples = []  # 다른 차량의 d_mean 기록
    
    def add_positive(self, d_mean):
        """같은 차량 매칭 성공 시"""
        self.positive_samples.append(d_mean)
        if len(self.positive_samples) > 100:
            self.positive_samples.pop(0)
    
    def add_negative(self, d_mean):
        """다른 차량 오탐 시"""
        self.negative_samples.append(d_mean)
        if len(self.negative_samples) > 100:
            self.negative_samples.pop(0)
    
    def get_adaptive_threshold(self):
        """
        통계 기반 최적 임계값 계산
        
        Returns:
            float: 동적 임계값 (또는 None)
        """
        if len(self.positive_samples) < 10 or len(self.negative_samples) < 10:
            return None  # 데이터 부족
        
        import numpy as np
        
        # 같은 차량: 95th percentile
        pos_95 = np.percentile(self.positive_samples, 95)
        
        # 다른 차량: 5th percentile
        neg_5 = np.percentile(self.negative_samples, 5)
        
        # 중간값 사용 (안전 마진 포함)
        adaptive = (pos_95 + neg_5) / 2
        
        # 범위 제한
        return max(0.50, min(0.85, adaptive))


# ===== 실전 사용 예시 =====
if __name__ == "__main__":
    # 예시 1: 기본 사용
    from_seg = "L"
    to_seg = "C"
    thresh = get_threshold(from_seg, to_seg, is_night=False)
    print(f"임계값 ({from_seg} → {to_seg}): {thresh:.2f}")
    
    # 예시 2: 신뢰도 평가
    d_mean = 0.65
    confidence = evaluate_match_confidence(d_mean)
    print(f"d_mean={d_mean:.2f} → 신뢰도: {confidence}")
    
    # 예시 3: 동적 임계값
    adaptive = AdaptiveThreshold()
    # ... 학습 ...
    # adaptive_thresh = adaptive.get_adaptive_threshold()