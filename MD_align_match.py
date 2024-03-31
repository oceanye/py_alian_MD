import cv2
import numpy as np
import time
# 初始化全局变量
drawing = False  # 如果按下鼠标，则为真
ix, iy = -1, -1
radius = 0  # 圆的半径
template = None  # 用于存储模板
template_mask = None  # 用于模板匹配的掩码


# 加载视频
video_path = 'Video_20240311134300390.avi'  # 替换为你的视频路径
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Failed to open video file.")
    exit()

# 创建鼠标回调函数
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, radius, template, template_mask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img2 = img.copy()

            radius = int(np.sqrt((x-ix)**2 + (y-iy)**2) / 2)
            center = (int((ix+x)/2), int((iy+y)/2))
            cv2.circle(img2, center, radius, (0, 255, 0), 2)
            cv2.imshow('image', img2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        center = (int((ix+x)/2), int((iy+y)/2))
        cv2.circle(img, center, radius, (0, 255, 0), 2)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        template = cv2.bitwise_and(img, img, mask=mask)[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]
        template_mask = mask[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]



# 创建鼠标回调函数

# 在第一帧图像上设置鼠标点击事件
ret, img = cap.read()
if not ret:
    print("Error: Failed to read first frame.")
    exit()

# 加载图像
img = cv2.imread('vlcsnap5.png')
#img图像缩小到20%
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

#img处理成gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (31, 31), 0)

difference = cv2.absdiff(img_gray, img_blur)
# 应用动态阈值
_, dyn_thresholded = cv2.threshold(difference, 10, 255, cv2.THRESH_BINARY)
#img=dyn_thresholded

#img处理成binary
#_, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

# 选择模板
f_i = 0
while(1):

    # 重新打开视频
    #cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # 按 'ESC' 退出
        break
    elif f_i>0 or (k == ord('m') and template is not None and template_mask is not None):  # 按 'm' 进行匹配
        #开始计时
        start = time.time()

        # 模板匹配
        # 使用TM_CCOEFF_NORMED方法进行模板匹配
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=template_mask)

        # 降低匹配阈值
        threshold = 0.2  # 假设原来是0.8，现在降低到0.6以放宽匹配容差

        # 根据阈值获取匹配结果的位置
        locations = np.where(result >= threshold)
        #将中心点位置距离小于50的result合并
        for i in range(len(locations[0])):
            for j in range(i+1,len(locations[0])):
                if (locations[0][i]-locations[0][j])**2+(locations[1][i]-locations[1][j])**2<50**2:
                    result[locations[0][i],locations[1][i]]=result[locations[0][j],locations[1][j]]
                    locations[0][i]=locations[0][j]
                    locations[1][i]=locations[1][j]

        #进一步筛选，只保留与鼠标框选时，Y坐标相差小于100的result
        for i in range(len(locations[0])):
            if abs(locations[0][i]-iy)>100:

                result[locations[0][i],locations[1][i]]=0




        # 在原图上标记所有匹配的区域
        for top_left in zip(*locations[::-1]):
            bottom_right = (top_left[0] + 2 * radius, top_left[1] + 2 * radius)
            cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
            #在rectangle左上角写出result值
            cv2.putText(img, str(result[top_left[1], top_left[0]]), top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        #结束计时
        end = time.time()

        #在这些locations中,放大区域20个像素形成ROI，在这些ROI中查找contours，img中绘制各个ROI中的contours
        #所有的ROI区域向外放大20个像素



        for top_left in zip(*locations[::-1]):
            c_i =0

            bottom_right = (top_left[0] + 2 * radius, top_left[1] + 2 * radius)
            roi = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #判断是否为圆形
            for contour in contours:
                area = cv2.contourArea(contour)
                if area<100:
                    continue
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2)

                if circularity < 0.7:
                    continue
                c_i+=1
                cv2.drawContours(roi, contour, -1, (0, 0, 255), 2)

                #建立mask，在contour中进行erosion，在进行threshold，找到面积最大的一个contour其中心点
                mask = np.zeros(roi_gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)

                #进行erosion
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
                mask = cv2.erode(mask, kernel, iterations=1)
                #进行threshold
                _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
                #找到mask中的contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #找到面积最大的contour
                max_area = 0
                max_area_contour = None
                for c in contours:
                    area = cv2.contourArea(c)
                    if area > max_area:
                        max_area = area
                        max_area_contour = c
                #找到面积最大的contour的中心点
                M = cv2.moments(max_area_contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                #在roi中画出中心点
                cv2.circle(roi, (cX, cY), 2, (0, 255, 0), -1)
                #在img中画出中心点
                cv2.circle(img, (top_left[0]+cX, top_left[1]+cY), 2, (0, 255, 0), -1)
                #在img中写出circularity和radius
                cv2.putText(img, "circularity:{:.2f}".format(circularity), (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img, "radius:{}".format(radius), (top_left[0], top_left[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                #
                #cv2.imshow('roi', roi)
                #cv2.waitKey(0)



                print("circularity:", circularity, "radius:", radius)
            if c_i>0:
                print("c_i:",c_i)


        cv2.imshow('Matched Result', img)
        #cv2.waitKey(0)

        #print result的个数
        print(len(locations[0]))
        #输出时间，单位s
        print("Time:",end-start)

        f_i = f_i+1
#cv2.destroyAllWindows()
