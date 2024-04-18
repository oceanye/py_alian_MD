import cv2
from skimage.feature import match_template
import numpy as np

# 初始化矩形坐标
rect = None


def select_region(image):
    global rect
    # 标记是否开始画图
    drawing = False
    ix, iy = -1, -1

    def mouse_callback(event, x, y, flags, param):
        nonlocal ix, iy, drawing
        global rect
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img = image.copy()
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("Select Region", img)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Region", image)

    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", mouse_callback)
    cv2.imshow("Select Region", image)
    cv2.waitKey(0)
    cv2.destroyWindow("Select Region")


cap = cv2.VideoCapture('Video_20240311134300390.avi')
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

ret, frame = cap.read()
if not ret:
    print("无法读取视频的第一帧")
    exit()

frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
select_region(frame)
template = frame[rect[1]:rect[3], rect[0]:rect[2]]

while True:
    ret, frame = cap.read()
    if not ret:
        print("视频帧读取完毕或失败")
        break
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    result = match_template(frame, template)
    heatmap = np.uint8(255 * result)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

    cv2.imshow('Original Frame', frame)
    cv2.imshow('Match Template Result', heatmap)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()