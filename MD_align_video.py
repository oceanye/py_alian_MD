import cv2
import numpy as np

# 读取视频
cap = cv2.VideoCapture('Video_20240311134300390.avi')

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()

# 获取视频信息
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Video_20240311134300390-r.avi', fourcc, fps, (width, height))

# 检查视频写入对象是否成功创建
if not out.isOpened():
    print("Error: Unable to create the output video file.")
    cap.release()
    exit()

# 创建窗口并设置初始大小
cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Contours', width, height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 添加模糊操作
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 通过亮度筛选图像轮廓
    _, thresh = cv2.threshold(blurred, 100, 225, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 循环处理每个轮廓
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        if 4000 < area < 50000:


            # 计算轮廓的圆度
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # 判断圆度是否满足条件
            if circularity < 0.6:
                continue

            # 根据面积大小选择颜色
            color = (0, 255, 0) if area > 0 else (0, 0, 255)
            # 画出轮廓
            cv2.drawContours(frame, [contour], -1, color, 2)


            # 在轮廓中心点标注圆度q
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(frame, f'{circularity:.2f}'+"/"+f'{area}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)

                # 向内偏移10像素
                radius = int(np.sqrt(area / np.pi)) - 30

                # 截取圆内的区域
                circle_mask = np.zeros_like(gray)
                cv2.circle(circle_mask, (cX, cY), radius, 255, -1)
                circle_area = cv2.bitwise_and(gray, gray, mask=circle_mask)

                # 找到圆内亮度最高的点
                _, max_val, _, max_loc = cv2.minMaxLoc(circle_area)

                # 标记红点
                cv2.circle(frame, max_loc, 5, (0, 0, 255), -1)

    # 将处理后的帧写入到视频文件中
    out.write(frame)

    # 显示结果并调整窗口大小
    cv2.imshow('Contours', frame)
    cv2.resizeWindow('Contours', width, height)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
