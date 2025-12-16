# tri_concat.py
# 3캠 결합: 2160×480
import cv2
import numpy as np

def resize_pad(frame, w, h):
    if frame is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    H, W = frame.shape[:2]
    scale = min(w / W, h / H)
    newW, newH = int(W * scale), int(H * scale)
    resized = cv2.resize(frame, (newW, newH)) if scale != 1.0 else frame
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x_off = (w - newW) // 2
    y_off = (h - newH) // 2
    canvas[y_off:y_off+newH, x_off:x_off+newW] = resized
    return canvas

def concat_three(left, center, right, target=(2160, 480)):
    W, H = target
    each_w = W // 3  # 720*3=2160
    lh = rh = ch = H
    lw = cw = rw = each_w
    l = resize_pad(left, lw, lh)
    c = resize_pad(center, cw, ch)
    r = resize_pad(right, rw, rh)
    return np.hstack([l, c, r])
