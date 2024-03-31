import random

import cv2
import numpy as np

# 加载视频
video_path = 'Video_20240311134300390.avi'  # 替换为你的视频路径
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Failed to open video file.")
    exit()


# 鼠标点击事件回调函数
def mouse_callback(event, x, y, flags, param):
    global x1, x2,y1,y2
    if event == cv2.EVENT_LBUTTONDOWN:
        if 'x1' not in globals():
            x1 = x
            y1 = y
        elif 'x2' not in globals():
            x2 = x
            y2 = y


def find_brightest_region_center_convolution(image, window_size):
    # 定义一个 10x10 的矩阵，其中所有元素都为 1
    kernel = np.ones(window_size, np.float32) / (window_size[0] * window_size[1])

    # 使用 cv2.filter2D 进行卷积操作
    convolved_image = cv2.filter2D(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), -1, kernel)

    # 找到最亮区域的中心点坐标
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(convolved_image)
    brightest_region_center = (max_loc[0] + window_size[0] // 2, max_loc[1] + window_size[1] // 2)

    return brightest_region_center


def find_point_region_center(image1, region_size):
    #在image中做binary，并检索轮廓，面积最接近region_size的轮廓即为所求，输该出轮廓中点

    cX= random.randint(0, 100)
    cY=random.randint(0, 100)

    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    #引入blur高斯模糊
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours.__len__()==0:
        return (cX,cY)




    for contour in contours:
        area = cv2.contourArea(contour)
        min_area_ratio = 10

        if area>region_size*10:
            continue

        if abs(1-area/region_size) < min_area_ratio:
            min_area_ratio = abs(1-area/region_size)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

            image2=image1.copy()
            # 将contour绘制在gray上,并保存到/tmp/gray*.jpg,其中gray*为顺序的编号
            cv2.drawContours(image2, contour, -1, (0, 255, 0), 1)
            #cv2 在image1的左上角写出 area的大小
            cv2.putText(image2, "a:{}".format(area), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # 通过imshow显示thresh
            cv2.imshow('image2', image2)
            # 要求加入随机数，避免覆盖 cv2.imwrite('thresh'+random.random(5)+'.jpg',thresh)
            cv2.imwrite('image2' + str(random.random()) + '.jpg', image2)

            #return (cX,cY)

    return (cX,cY)


# 在第一帧图像上设置鼠标点击事件
ret, first_frame = cap.read()
if not ret:
    print("Error: Failed to read first frame.")
    exit()

resized_scale=2

resized_first_frame = cv2.resize(first_frame, (first_frame.shape[1] // resized_scale, first_frame.shape[0] // resized_scale))

cv2.namedWindow('Select Points')
cv2.setMouseCallback('Select Points', mouse_callback)

while True:
    cv2.imshow('Select Points', resized_first_frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or ('x1' in globals() and 'x2' in globals()):
        break

cv2.destroyAllWindows()
cap.release()

x1 = int(resized_scale*x1)
x2 = int(resized_scale*x2)

angle0= np.arctan2(x2-x1,y2-y1)

print("Selected x1:", x1)
print("Selected x2:", x2)

# 重新打开视频
cap = cv2.VideoCapture(video_path)

# 处理每一帧图像
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 沿着 Y 方向划分为宽度为 100 的条状
    width = frame.shape[1]
    strip_height = 200
    num_strips = width // strip_height


    x_offset = 100
    for i in range(num_strips):
        # 计算当前条状的范围
        start_x1 = x1 - x_offset
        end_x1 = x1 + x_offset

        start_x2 = x2 - x_offset
        end_x2 = x2 + x_offset

        # 确保范围在图像内
        #start_x1 = max(0, start_x1)
        #end_x1 = min(width, end_x1)

        #start_x2 = max(0, start_x2)
        #end_x2 = min(width, end_x2)

        # 获取当前条状的图像
        strip1 = frame[start_x1:end_x1,i*strip_height:(i+1)*strip_height]
        strip2 = frame[start_x2:end_x2,i*strip_height:(i+1)*strip_height ]

        # 确保截取的区域不为空
        if strip1.size == 0 or strip2.size == 0:
            continue

        # 计算 strip1 的最亮点
        #max_position1 = cv2.minMaxLoc(cv2.cvtColor(strip1, cv2.COLOR_BGR2GRAY))[3]
        #max_position1 = find_brightest_region_center_convolution(strip1, (10, 10))
        max_position1=find_point_region_center(strip1, 100)
        xl1 = start_x1 + max_position1[0]
        yl1 = i*strip_height + max_position1[1]
        #print("start_x1:",start_x1)
        #print("xl1:",xl1)
        #print("max_position1:",max_position1)

        # 计算 strip2 的最亮点
        #max_position2 = cv2.minMaxLoc(cv2.cvtColor(strip2, cv2.COLOR_BGR2GRAY))[3]
        #max_position2 = find_brightest_region_center_convolution(strip2, (10, 10))
        max_position2=find_point_region_center(strip2, 100)
        xl2 = start_x2 + max_position2[0]
        yl2 = i*strip_height +max_position2[1]


        # 计算最亮点距离
        distance = abs(xl2 - xl1)
        angle = np.arctan2(yl2-yl1,xl2-xl1)

        cv2.line(frame,(start_x1,i*strip_height),(start_x1,(i+1)*strip_height),(255,0,255),1)
        cv2.line(frame,(end_x1,i*strip_height),(end_x1,(i+1)*strip_height),(255,0,255),1)

        cv2.line(frame,(start_x2,i*strip_height),(start_x2,(i+1)*strip_height),(255,0,255),1)
        cv2.line(frame,(end_x2,i*strip_height),(end_x2,(i+1)*strip_height),(255,0,255),1)

        cv2.line(frame,(start_x1,i*strip_height),(end_x1,i*strip_height),(255,0,255),1)
        cv2.line(frame,(start_x2,i*strip_height),(end_x2,i*strip_height),(255,0,255),1)


        # 输出每个 strip 的最亮点
        # 如果最亮点距离和原始选择点的距离偏差小于阈值，则标记为绿色
        if ((abs(1-distance /(x2 - x1))  < 0.1) and (abs(1-angle0/angle)<0.1)):
            color = (0, 255, 0)  # 绿色
            cv2.line(frame,(xl1,yl1),(xl2,yl2),(0,255,0),2)
            print("Strip {} - Brightest Point x1: {}, x2: {},dist:{},OK".format(i + 1, xl1, xl2, distance))
        else:
            color = (0, 0, 255)  # 红色
            print("Strip {} - Brightest Point x1: {}, x2: {},dist:{}".format(i + 1, xl1, xl2, distance))

        # 在原图上标记红色的中心点

        cv2.circle(frame, (xl1,yl1), 3, color, -1)


        cv2.circle(frame, (xl2,yl2), 3, color, -1)


    print("------")

    # 缩放图像以便在更小的窗口中显示
    resized_window = cv2.resize(frame, (frame.shape[1] // resized_scale, frame.shape[0] // resized_scale))

    # 显示处理后的图像
    cv2.imshow('Processed Frame', resized_window)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
