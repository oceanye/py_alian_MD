import cv2
import numpy as np

# 读取图像
image = cv2.imread('frame_0030.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 添加模糊操作
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 通过亮度筛选图像轮廓
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个窗口并设置大小
cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Contours', 800, 600)

# 循环处理每个轮廓
for contour in contours:
    # 计算轮廓的面积
    area = cv2.contourArea(contour)
    if area > 1000 and area < 20000:
        # 根据面积大小选择颜色
        color = (0, 255, 0) if area > 0 else (0, 0, 255)
        # 画出轮廓
        #cv2.drawContours(image, [contour], -1, color, 2)

        # 计算轮廓的圆度
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # 判断圆度是否满足条件
        if circularity < 0.7:
            continue

        # 根据面积大小选择颜色
        color = (255, 0, 0) if area > 0 else (0, 0, 255)
        # 画出轮廓
        cv2.drawContours(image, [contour], -1, color, 2)

        # 在轮廓中心点标注圆度
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(image, f'{circularity:.2f}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)

            # 向内偏移10像素
            radius = int(np.sqrt(area / np.pi)) - 30

            # 截取圆内的区域
            circle_mask = np.zeros_like(gray)
            cv2.circle(circle_mask, (cX, cY), radius, 255, -1)
            circle_area = cv2.bitwise_and(gray, gray, mask=circle_mask)

            # 找到圆内亮度最高的点
            _, max_val, _, max_loc = cv2.minMaxLoc(circle_area)

            # 标记红点
            cv2.circle(image, max_loc, 5, (0, 0, 255), -1)

# 显示结果
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
