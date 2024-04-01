import cv2
import numpy as np

# 全局变量
drawing = False  # 真如果鼠标被按下
mode = True  # 如果为真，绘制矩形。按'm'切换到曲线
ix, iy = -1, -1
circles = []  # 用于存储圆心和半径
circles_draw=[] #用于绘制的圆


# 鼠标回调函数
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, img_temp, circles,img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        img_temp = img.copy() # 复制图像以绘制预览圆

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img = img_temp.copy() # 恢复到未绘制圆的状态
            cv2.circle(img, (ix, iy), int(np.sqrt((x - ix)**2 + (y - iy)**2)), (0, 255, 0), 1) # 绘制预览圆

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        radius = int(np.sqrt((x - ix)**2 + (y - iy)**2))
        circles_draw.append((ix, iy, radius))
        cv2.circle(img, (ix, iy), radius, (0, 255, 0), 2) # 绘制最终圆


# 读取视频
cap = cv2.VideoCapture('Video_20240311134546312.avi')

# 读取第一帧
ret, img = cap.read()
#img 缩小到50%
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while (1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Esc键退出
        break
    elif k == 13:  # Enter键确认绘制的圆
        break

#cap.release()
cv2.destroyAllWindows()

# 重新打开视频
#cap = cv2.VideoCapture('your_video.avi')

while (cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(gray, (31, 31), 0)
        #difference = cv2.absdiff(gray, blur)
        #_, thresh = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)

        #建立gray_roi是draw_circle1的圆心的左右两倍半径区域，Y不做限制

        # 假设 circles_draw[i][0] 是 x 坐标，circles_draw[i][1] 是 y 坐标，circles_draw[i][2] 是半径
        gray_roi_1 = gray[:,
                     (circles_draw[0][0] - circles_draw[0][2] ):(circles_draw[0][0] + circles_draw[0][2] )]
        gray_roi_2 = gray[:,
                     (circles_draw[1][0] - circles_draw[1][2] ):(circles_draw[1][0] + circles_draw[1][2] )]

        # 应用 Canny 边缘检测

        median_val1 = np.median(gray_roi_1)  # 计算图像中位数

        # 设置高低阈值
        lower_threshold1 = int(max(0, (1.0 - 0.33) * median_val1))
        upper_threshold1 = int(min(255, (1.0 + 0.33) * median_val1))


        median_val2 = np.median(gray_roi_2)  # 计算图像中位数

        # 设置高低阈值
        lower_threshold2 = int(max(0, (1.0 - 0.33) * median_val2))
        upper_threshold2 = int(min(255, (1.0 + 0.33) * median_val2))


        edges_1 = cv2.Canny(gray_roi_1, lower_threshold1, upper_threshold1)
        edges_2 = cv2.Canny(gray_roi_2, lower_threshold2, upper_threshold2)


        #opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3)))


        r_min = int(min(circles_draw[0][2], circles_draw[1][2])*0.8)
        r_max = int(max(circles_draw[0][2], circles_draw[1][2])*1.2)
        circles_1 = cv2.HoughCircles(edges_1, cv2.HOUGH_GRADIENT, dp=1, minDist=int(r_min*0.5), param1=50, param2=20, minRadius=r_min, maxRadius=r_max)
        circles_2 = cv2.HoughCircles(edges_2, cv2.HOUGH_GRADIENT, dp=1, minDist=int(r_min*0.5), param1=50, param2=20, minRadius=r_min, maxRadius=r_max)

        #将circles_1映射回到frame位置

        #frame 的text写出 circles_1的个数与circles_2的个数
        if (circles_1 is not None) and (circles_2 is not None):
            cv2.putText(frame, "C1:{}".format(circles_1.shape[1])+"/C2:{}".format(circles_2.shape[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)


        #江circle_1 向右移动circles_draw[0][0]，circle_2 向右移动circles_draw[1][0]
        if circles_1 is not None:
            circles_1[:,:, 0] += (circles_draw[0][0]-circles_draw[0][2] )

        if circles_2 is not None:
            circles_2[:,:, 0] += (circles_draw[1][0] - circles_draw[1][2] )



        if (circles_1 is not None) and(circles_2 is not None):




            # 假设circles[0]为C1，circles[1]为C2

            c1_match, c2_match = None, None
            min_dist_c1, min_dist_c2 = float('inf'), float('inf')

            for i in circles_1[0, :]:
                # 计算圆心之间的距离
                dist_c1 = (i[0] - circles_draw[0][0]) ** 2 #  Ci_x = i[0], Ci_y = i[1]



                c1_match = i




            # 历边circles，计算其中的圆心与点P的距离dist_P,其中P_x = c1_match的x坐标,P_y = circles_draw[0][1]
            # 如果dist_P < 10,则认为该圆心为c2_match
                #if (c1_match is not None) and (c2_match is not None):
                for i in circles_2[0, :]:
                    dist_P = (i[0] - circles_draw[1][0]) ** 2 + (i[1] - c1_match[1]) ** 2
                    if dist_P < c1_match[2] ** 2:
                        c2_match = i
                        break

                # 如果c1_match和c2_match都不为空，则计算两个圆圆心的X距离，和draw_circle做比较，如果距离差小于draw_circle的半径的则绘制直线，否则 continue
                if (c1_match is not None) and (c2_match is not None):
                    distance = abs(c2_match[0] - c1_match[0])
                    distance_draw = abs(circles_draw[1][0] - circles_draw[0][0])

                    if abs(c1_match[1]-c2_match[1])>circles_draw[0][2]*0.1:
                        continue

                    l_match = c1_match[0] - c2_match[0]
                    l_draw = circles_draw[0][0] - circles_draw[1][0]
                    if abs(l_match-l_draw)>circles_draw[0][2]*0.05:
                        continue

                    print("x1,y1/ x2,y2/distance/distance_draw", c1_match[0], c1_match[1], c2_match[0], c2_match[1], distance, distance_draw)
                    if abs(distance-distance_draw) < int(c1_match[2]):
                        color = (0, 255, 0)
                        cv2.line(frame, (int(c1_match[0]), int(c1_match[1])), (int(c2_match[0]), int(c2_match[1])), (0, 255, 0), 2)

                    else:
                        continue

                # 绘制最匹配的圆形

                    cv2.circle(frame, (int(c1_match[0]), int(c1_match[1])), int(c1_match[2]), (0, 255, 0), 2)  # 绘制圆
                    cv2.putText(frame, "C1_R", (int(c1_match[0]), int(c1_match[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)  # 添加文本

                    cv2.circle(frame, (int(c2_match[0]), int(c2_match[1])), int(c2_match[2]), (0, 0, 255), 2)  # 绘制圆
                    cv2.putText(frame, "C2_R", (int(c2_match[0]), int(c2_match[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)  # 添加文本


            # 显示结果
            cv2.imshow('Frame with Detected Circles', frame)
            #cv2.imshow('thresh', thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'键退出
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
