import cv2
import numpy as np
import time

# 初始化全局变量
drawing = False  # 如果按下鼠标，则为真
ix, iy = -1, -1  # 鼠标按下时的坐标
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

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img2 = frame.copy()
        radius = int(np.sqrt((x - ix)**2 + (y - iy)**2))
        cv2.circle(img2, (ix, iy), radius, (0, 255, 0), 2)
        cv2.imshow('First Frame', img2)

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        radius = int(np.sqrt((x - ix)**2 + (y - iy)**2))
        cv2.circle(frame, (ix, iy), radius, (0, 255, 0), 2)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (ix, iy), radius, 255, -1)

        # 使用掩码提取圆形模板
        template = cv2.bitwise_and(frame, frame, mask=mask)[iy-radius:iy+radius, ix-radius:ix+radius]

        # 检查模板是否有效（非全零）
        if cv2.countNonZero(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)) > 0:
            # 模板有效，可以继续处理
            print("Template extracted successfully.")
        else:
            print("Template is empty. Check the circle's radius and position.")


        # 创建与模板相同大小的掩码，其中圆形区域为白色
        template_mask = np.zeros((2*radius, 2*radius), dtype=np.uint8)
        cv2.circle(template_mask, (radius, radius), radius, 255, -1)
        cv2.imshow('First Frame', frame)
        cv2.destroyWindow('First Frame')  # 关闭第一帧的窗口
# 读取并显示第一帧，等待用户选择模板
ret, frame = cap.read()

#frame缩小到20%
frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
#frame 改为灰度图
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

if ret:
    cv2.namedWindow('First Frame')
    cv2.setMouseCallback('First Frame', draw_circle)
    cv2.imshow('First Frame', frame)
    cv2.waitKey(0)  # 等待用户完成模板选择
else:
    print("Error: Failed to read the first frame.")
    cap.release()
    exit()

# 主循环，对视频的每一帧进行处理
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
    # frame 改为灰度图
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        break



    #如果模板和掩码已定义，执行模板匹配和轮廓查找
    if template is not None:
        # 模板的高和宽
        h, w = template.shape[:2]

        # 初始化与模板大小相同的掩码
        template_mask = np.zeros((h, w), dtype=np.uint8)

        # 假设圆形模板位于掩码的中心，设置掩码中的圆形区域为白色
        cv2.circle(template_mask, (w // 2, h // 2), radius, 255, -1)

        # 使用模板和掩码进行模板匹配
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED, mask=template_mask)

        dist = radius
        #对result矩阵进行处理，距离10个像素以内只保留最大的数
        for i in range(len(result)):
            for j in range(len(result[0])):
                if i<dist or j<dist or i>len(result)-dist or j>len(result[0])-dist:
                    result[i][j]=0




        threshold = 0.3 # 根据需要调整阈值
        locations = np.where(result >= threshold)


        #遍历locations，绘制出各个矩形
        #for loc in locations:
        #    cv2.rectangle(frame, loc, (loc[0] + w, loc[1] + h), (0, 0, 255), 2)



        for pt in zip(*locations[::-1]):

            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)



        #输出locations的的个数
        print("ROI个数:",len(locations))

        # 遍历所有匹配的位置
        for loc in zip(*locations[::-1]):
            top_left = loc
            #如果locations只有一个元素，那么bottom_right为
            if len(locations)==1:
                bottom_right = (top_left[0] + w, top_left[1] + h)

            else:
                bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

            # 定义ROI
            roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # 在ROI中查找轮廓
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            #j计算roi的面积，并且print#计算frame的面积并print

            print("ROI的面积:",np.sum(roi_gray>0),"frame的面积:",np.sum(frame>0))

            #_, roi_thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY)
            roi_blur = cv2.GaussianBlur(roi_gray, (round(radius/10)*2+1,round(radius/10)*2+1), 0)
            diff = cv2.absdiff(roi_gray, roi_blur)
            _, roi_thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
            #roi_thresh = cv2.dilate(roi_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)), iterations=1)
            #roi_thresh 进行闭运算
            roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
            contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #在contours中筛选面积大于100，且圆度大于0.7的contours
            print("ROI中的contours个数:",len(contours))
            #frame顶部写出contours的个数
            cv2.putText(frame, "ROI_Contours_Count:{}".format(len(contours)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            c_i = 0
            # 在原视频帧上绘制找到的轮廓
            for contour in contours:

                area = cv2.contourArea(contour)
                if area<radius*radius/4:
                    continue
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2)

                if circularity < 0.3:
                    continue

                # 轮廓坐标需要调整为相对于原帧的位置
                adjusted_contour = contour + [top_left[0], top_left[1]]
                cv2.drawContours(frame, [adjusted_contour], -1, (0, 255, 0), 2)
                c_i += 1

            print("ROI中的圆形个数:",c_i)
            #frame顶部写出圆形的个数
            cv2.putText(frame, "ROI_Cricle:{}".format(c_i), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('Frame with Contours', frame)
    #清除frame
    frame = None
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'退出
        break

cap.release()
cv2.destroyAllWindows()
