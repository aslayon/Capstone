# core/roi_utils.py

def shift_roi(roi, offset_x=0, offset_y=0):
    if not roi: return None
    x1,y1,x2,y2 = roi
    return (x1+offset_x, y1+offset_y, x2+offset_x, y2+offset_y)

def bbox_center_in_any_roi(det, rois):
    x1,y1,x2,y2,conf,cls = det
    cx,cy = (x1+x2)/2, (y1+y2)/2
    return any(r and (r[0] <= cx <= r[2] and r[1] <= cy <= r[3]) for r in rois)


def tri_rois(center_roi, left_roi=None, right_roi=None, seg_w=720):
    """
    2160x480(= 720*3 x 480) 기준으로,
    좌/중/우 카메라 ROI를 결합 프레임 좌표계로 변환해 반환.
    - center_roi: 필수 (기본 ROI)
    - left/right_roi: 없으면 center_roi를 그대로 사용(간편 모드)
    """
    base = center_roi
    l = shift_roi(left_roi or base,   offset_x=0)
    c = shift_roi(center_roi,         offset_x=seg_w)
    r = shift_roi(right_roi or base,  offset_x=seg_w*2)
    return l, c, r



