import cv2
import matplotlib.pyplot as plt

stream_url = "http://cctvsec.ktict.co.kr/138/pQahsqagIvXoxtKYMYuTVxSWQPyEx4a/DycV69i2ghScblbPnSTRLT9ttd6K1vxfdzVH2B2WDjzDDFu8a5pSZocJ9jNGE5Bx51hdStrzVl0="  
cap = cv2.VideoCapture(stream_url)

ret, frame = cap.read()
if not ret:
    print("❌ 프레임 읽기 실패")
else:
    # BGR -> RGB 변환 (matplotlib은 RGB 기준)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.imshow(frame_rgb)
    plt.title("CCTV Frame")
    plt.axis("off")
    plt.show()

cap.release()
