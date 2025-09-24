# core/roi_utils.py
def shift_roi(roi, offset_x=0, offset_y=0):
    if roi is None: return None
    x1,y1,x2,y2 = roi
    return (x1+offset_x, y1+offset_y, x2+offset_x, y2+offset_y)

def bbox_center_in_any_roi(det, rois):
    x1,y1,x2,y2,conf,cls = det
    cx,cy = (x1+x2)/2, (y1+y2)/2
    for r in rois:
        if r and r[0] <= cx <= r[2] and r[1] <= cy <= r[3]:
            return True
    return False
