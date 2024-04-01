import cv2
import numpy as np

# 步骤1：读取PNG图像
image = cv2.imread('vlcsnap2.png')  # 请将'path_to_your_image.png'替换为你的图像文件路径

# 步骤2：将图像转换为灰度
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_image', gray_image)
#gray_image 进行mean 和 dyn_threshold
blur_image = cv2.GaussianBlur(gray_image, (31, 31), 0)
cv2.imshow('blur_image', blur_image)
difference = cv2.absdiff(gray_image, blur_image)

# 计算差异图像的均值，用作动态阈值
mean_diff = np.mean(difference)
print("mean_diff:",mean_diff)

# 应用动态阈值
_, dyn_thresholded = cv2.threshold(difference, 10, 255, cv2.THRESH_BINARY)


cv2.imshow('dyn_threshold', dyn_thresholded)

# 步骤3：创建一个垂直方向的结构元素

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))  # 宽度为1，高度为5的矩形
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))  # 宽度为1，高度为5的矩形
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,4))
#kernel5 为圆形，直径2
kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))

# 步骤4：应用闭运算
#closed = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

#erosion0 = cv2.erode(dyn_thresholded, kernel3, iterations=1)
erosion = cv2.erode(dyn_thresholded, kernel4, iterations=1)
erosion1 = cv2.erode(erosion, kernel1, iterations=3)
dialeted = cv2.dilate(erosion1, kernel5, iterations=1)
#erosion2 = cv2.dilate(dialeted, kernel2, iterations=1)
#针对dialeted做一次闭区间操作

# 闭运算填充圆中的缺口
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
closed_image = cv2.morphologyEx(dyn_thresholded, cv2.MORPH_CLOSE, kernel_close)

# 开运算移除直线
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
opened_image = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel_open)

#closed = cv2.morphologyEx(dialeted, cv2.MORPH_CLOSE, kernel2)
#对erosion2做一次开区间操作

#opened = cv2.morphologyEx(dialeted, cv2.MORPH_OPEN, kernel2)

# 查找dialeted中的边界
contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 查找contours中的圆形

#在opened_image中找houghcircle
circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=50, maxRadius=100)

#imshow circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(opened_image, (i[0], i[1]), i[2], (255, 0, 0), 2)
        # draw the center of the circle
        cv2.circle(opened_image, (i[0], i[1]), 2, (0, 255, 0), 5)
        print("center:",(i[0], i[1]),"radius:",i[2])
#print contours的个数
print(len(contours))


#contours画到dialeted上
#cv2.drawContours(erosion, contours, -1, (255, 0, 0), 2)
ci=0
for contour in contours:
    # 计算轮廓的周长
    perimeter = cv2.arcLength(contour, True)
    # 计算轮廓的面积
    area = cv2.contourArea(contour)

    if area > 50:  # 防止除以0的情况
        # 计算圆形度量
        circularity = (perimeter ** 2) / (4 * np.pi * area)

        # 设定圆形度量的阈值，例如0.8到1.2之间认为是圆形
        if 0.75 < circularity < 1.2:
            # 计算最小外接圆
            ci+=1
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            print("center:",center,"radius:",radius)
            # 绘制圆形
            cv2.circle(erosion, center, radius, (255, 255, 0), 1)


print("ci:",ci)




# 显示原始图像和处理后的图像
#cv2.imshow('Original Image', image)
cv2.imshow("erosion Image", erosion)
cv2.imshow('erosion1 Image', erosion1)
#cv2.imshow('dialeted Image', dialeted)
cv2.imshow('opened Image', opened_image)
#cv2.imshow('closed Image', closed)

# 等待按键然后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
