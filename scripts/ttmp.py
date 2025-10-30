import cv2

img_path = "assets/samples/frame.jpg"   # CCTV에서 캡처한 샘플 이미지
img = cv2.imread(img_path)
clone = img.copy()
roi_coords = []

# 마우스 콜백
def select_roi(event, x, y, flags, param):
    global roi_coords, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_coords = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        roi_coords.append((x, y))
        cv2.rectangle(clone, roi_coords[0], roi_coords[1], (255,0,0), 2)
        cv2.imshow("ROI Selector", clone)

cv2.namedWindow("ROI Selector")
cv2.setMouseCallback("ROI Selector", select_roi)

while True:
    cv2.imshow("ROI Selector", clone)
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter 키 누르면 종료
        break
    elif key == 27:  # Esc 누르면 취소
        roi_coords = []
        break

cv2.destroyAllWindows()

if roi_coords and len(roi_coords) == 2:
    (x1,y1), (x2,y2) = roi_coords
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    print(f"ROI 좌표: ({x1},{y1},{x2},{y2})")
