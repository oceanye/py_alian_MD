import cv2
import numpy as np
import camera_utils

# 全局变量
points = []
trackers = []

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # 创建一个新的CSRT追踪器
        tracker = cv2.TrackerCSRT_create()
        # 初始化追踪器
        bbox = (x-20, y-20, 40, 40)  # 假设目标大小为40x40
        tracker.init(frame, bbox)
        trackers.append(tracker)
        # 在图像上画出选择的点
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.namedWindow('Frame', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Frame', frame)

camera = camera_utils.open_camera()
# 打开摄像头


# 读取第一帧
frame = camera_utils.get_frame(camera)


# 创建窗口并设置鼠标回调
cv2.namedWindow('Frame1',cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('Frame1', mouse_callback)

print("请在图像上点击要追踪的点，然后按'q'开始追踪")

# 等待用户选择点
while True:

    # 建立一个namedwindow
    cv2.namedWindow('Frame1',cv2.WINDOW_KEEPRATIO)

    cv2.imshow('Frame1', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"选择了 {len(points)} 个点进行追踪")

# 主循环
while True:
    frame = camera_utils.get_frame(camera)


    # 更新每个追踪器
    for i, tracker in enumerate(trackers):
        success, bbox = tracker.update(frame)
        if success:
            # 追踪成功，更新点的位置并画出
            x, y = int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)
            points[i] = (x, y)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                          (255, 0, 0), 2)
            # 在点旁边显示点的编号
            cv2.putText(frame, str(i), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            # 追踪失败，画出一个红色的圆
            cv2.circle(frame, points[i], 5, (0, 0, 255), -1)
            # 在点旁边显示点的编号
            cv2.putText(frame, str(i), (points[i][0]+10, points[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.namedWindow('Tracking',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()