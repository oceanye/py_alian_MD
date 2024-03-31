import cv2
import numpy as np

# 定义全局变量
threshold_min = 0
threshold_max = 255
min_circularity = 0.1
max_circularity = 0.8
min_area = 10
max_area = 10000

# 回调函数，用于调整阈值范围滑块
def onThresholdChangeMin(value):
    global threshold_min
    threshold_min = value
    updateResult()

def onThresholdChangeMax(value):
    global threshold_max
    threshold_max = value
    updateResult()

# 回调函数，用于调整最小圆形度滑块
def onCircularityMinChange(value):
    global min_circularity
    min_circularity = 0.1 + value / 100.0
    updateResult()

# 回调函数，用于调整最大圆形度滑块
def onCircularityMaxChange(value):
    global max_circularity
    max_circularity = 1.1 + value / 100.0
    updateResult()

# 回调函数，用于调整面积范围滑块
def onAreaChange(value):
    global min_area
    min_area = value
    updateResult()

# 更新结果函数
def updateResult():
    global image, output, threshold_min, threshold_max, min_circularity, max_circularity, min_area, max_area
    output = image.copy()
    # 转换为灰度图
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, thresh = cv2.threshold(gray, threshold_min, threshold_max, cv2.THRESH_BINARY)
    print(threshold_min,threshold_max)
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选封闭曲线
    closed_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:  # 如果面积不在范围内
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = (perimeter ** 2) / (4 * np.pi * area)
        if min_circularity < circularity < max_circularity:  # 如果圆形度在范围内
            closed_contours.append(contour)

    # 绘制筛选后的封闭曲线
    cv2.drawContours(output, closed_contours, -1, (0, 255, 0), 3)

    # 显示结果图像
    cv2.imshow("Closed Contours", output)

# 读取图像
image = cv2.imread('frame_0030.jpg')

# 创建窗口
cv2.namedWindow('Closed Contours', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Closed Contours', 800, 600)  # 设置窗口大小

# 创建阈值范围滑块
cv2.createTrackbar('Min Threshold', 'Closed Contours', threshold_min, 255, onThresholdChangeMin)
cv2.createTrackbar('Max Threshold', 'Closed Contours', threshold_max, 255, onThresholdChangeMax)

# 创建圆形度范围滑块
# 创建圆形度范围滑块
cv2.createTrackbar('Min Circularity', 'Closed Contours', int((min_circularity - 0.2) * 100), 60, onCircularityMinChange)
cv2.createTrackbar('Max Circularity', 'Closed Contours', int((max_circularity - 1.2) * 100), 80, onCircularityMaxChange)

# 创建面积范围滑块
cv2.createTrackbar('Min Area', 'Closed Contours', min_area, 10000, onAreaChange)

# 初始化结果
output = image.copy()
updateResult()

# 等待键盘按键
cv2.waitKey(0)
cv2.destroyAllWindows()
