import random
import cv2
import numpy as np
import msvcrt

import camera_utils

#2024.05.10 提交

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



#定义一个函数，检索图像中的最大的封闭曲线
def find_MD_contour(image1):
    # 转换为灰度图像
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # 使用高斯滤波器平滑图像
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #执行膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated = cv2.dilate(thresh, kernel)



    # 闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)

    edges_1 = cv2.Canny(closed, lower_threshold1, upper_threshold1)
    # 查找轮廓
    contours, _ = cv2.findContours(edges_1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   #cv2.RETR_LIST #cv2.RETR_EXTERNAL




    List_cX=[]
    List_cY=[]

    #计算contours的面积
    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 1000 and area >100:
            #cv2.drawContours(image, contour, -1, (0, 255, 0), 2)
            # 在中心标注面积
            M = cv2.moments(contour)
            # 计算轮廓的中心x,y坐标


            if M["m00"] != 0:
                List_cX.append(int(M["m10"] / M["m00"]/2))
                List_cY.append(int(M["m01"] / M["m00"]/2))
                #绘制轮廓
                cv2.drawContours(image1, contour, -1, (0, 255, 0), 2)
                #写上面积
                cv2.putText(image1, "A:{}".format(area), (List_cX[-1], List_cY[-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 输出单个ROI中MD的q调试结果
    fn_MD= str(random.randint(10000, 99999)) + '.png'

    #cv2.imwrite("edge_"+fn_MD,edges_1)
    #cv2.imwrite("image_" + fn_MD, image1)

    return List_cX,List_cY


cap_source = "camera"  # "camera" or "video"

if cap_source == "camera":
    cam = camera_utils.open_camera()
    data, width, height = camera_utils.get_frame(cam)
    img = np.frombuffer(data, dtype=np.uint8)
    img = img.reshape((height, width, 1))

else:
    # 读取视频
    cap = cv2.VideoCapture('Video_20240311134223551.avi')

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

frame_count = 0
line_count =0
while (cap.isOpened() or (cam is not None)):
    if cap_source == "camera":
        data, width, height = camera_utils.get_frame(cam)
        img = np.frombuffer(data, dtype=np.uint8)
        frame = img.reshape((height, width, 1))
        ret = True
    else:
        ret, frame = cap.read()

    frame2 = frame.__deepcopy__(frame)
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    #建立一个循环计数，每24次循环清空frame_count
    frame_count = frame_count + 1


    if frame_count%24==0:
        line_count = 0

    if ret == True:
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(gray, (31, 31), 0)
        #difference = cv2.absdiff(gray, blur)
        #_, thresh = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)

        #建立gray_roi是draw_circle1的圆心的左右两倍半径区域，Y不做限制

        # 假设 circles_draw[i][0] 是 x 坐标，circles_draw[i][1] 是 y 坐标，circles_draw[i][2] 是半径

        r_min = int(min(circles_draw[0][2], circles_draw[1][2])*0.8)
        r_max = int(max(circles_draw[0][2], circles_draw[1][2])*1.2)

        frame_roi_1 = frame[:,
                     (circles_draw[0][0] - r_max ):(circles_draw[0][0] + r_max )]
        frame_roi_2 = frame[:,
                     (circles_draw[1][0] - r_max ):(circles_draw[1][0] + r_max )]

        gray_roi_1 = cv2.cvtColor(frame_roi_1, cv2.COLOR_BGR2GRAY)
        gray_roi_2 = cv2.cvtColor(frame_roi_2, cv2.COLOR_BGR2GRAY)

        gray_roi_1 = cv2.GaussianBlur(gray_roi_1, (7, 7), 0)
        gray_roi_2 = cv2.GaussianBlur(gray_roi_2, (7, 7), 0)

        #gray_roi_1 = cv2.erode(gray_roi_1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3)))
        #gray_roi_2 = cv2.erode(gray_roi_2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3)))

        #gray_roi_1 =cv2.erode(gray_roi_1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3)))
        #gray_roi_2 =cv2.erode(gray_roi_2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3)))

        #在frame上标注gray_roi_1与gray_roi_2的4条Y向分割线
        cv2.line(frame, (circles_draw[0][0] - r_max, 0), (circles_draw[0][0] - r_max, frame.shape[0]), (255, 255, 0), 2)
        cv2.line(frame, (circles_draw[0][0] + r_max, 0), (circles_draw[0][0] + r_max, frame.shape[0]), (255, 255, 0), 2)
        cv2.line(frame, (circles_draw[1][0] - r_max, 0), (circles_draw[1][0] - r_max, frame.shape[0]), (255, 255, 0), 2)
        cv2.line(frame, (circles_draw[1][0] + r_max, 0), (circles_draw[1][0] + r_max, frame.shape[0]), (255, 255, 0), 2)

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

        #提取edges_1中的圆
        #contour_c1, _ = cv2.findContours(edges_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contour_c2, _ = cv2.findContours(edges_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        circles_1 = cv2.HoughCircles(edges_1, cv2.HOUGH_GRADIENT, dp=1, minDist=int(r_min*0.5), param1=50, param2=25, minRadius=r_min, maxRadius=r_max)
        circles_2 = cv2.HoughCircles(edges_2, cv2.HOUGH_GRADIENT, dp=1, minDist=int(r_min*0.5), param1=50, param2=25, minRadius=r_min, maxRadius=r_max)



        #将circles_1映射回到frame位置

        #frame 的text写出 circles_1的个数与circles_2的个数
        if (circles_1 is not None) and (circles_2 is not None):
            cv2.putText(frame, "C1:{}".format(circles_1.shape[1])+"/C2:{}".format(circles_2.shape[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)


        #江circle_1 向右移动circles_draw[0][0]，circle_2 向右移动circles_draw[1][0]
        if circles_1 is not None:
            circles_1[:,:, 0] += (circles_draw[0][0]-r_max )

        if circles_2 is not None:
            circles_2[:,:, 0] += (circles_draw[1][0] - r_max )


        #frame上画出circles_1与circles_2
        if circles_1 is not None:
            circles_1 = np.uint16(np.around(circles_1))
            for i in circles_1[0, :]:
                #frame中截取圆形区域i，并保存为png
                #cv2.imwrite('circle_1_'+str(random.randint(10000,99999))+'.png',frame2[i[1]-i[2]:i[1]+i[2],i[0]-i[2]:i[0]+i[2]])
                #打印输出C1的识别到的圆，不绘制
                print("C1_X:",i[0],"C1_Y:",i[1],"C1_R:",i[2])
                cv2.circle(frame, (i[0] , i[1]), i[2], (0, 255, 255), 1)




        if circles_2 is not None:
            circles_2 = np.uint16(np.around(circles_2))
            for i in circles_2[0, :]:
                cv2.circle(frame, (i[0] , i[1]), i[2], (0, 255, 255), 1)
                print("C2_X:", i[0], "C2_Y:", i[1], "C2_R:", i[2])


                # frame中截取圆形区域i，并保存为png
                #cv2.imwrite('circle_2_' + str(random.randint(10000,99999)) + '.png',frame2[i[1] - i[2]:i[1] + i[2], i[0] - i[2]:i[0] + i[2]])

        #在circles_1与circles_2中相互组合，找到最匹配的两个圆，满足直线L（C1_R,C2_R） 与draw_circle的圆心连线夹角小于10度，且距离差小于20




        if (circles_1 is not None) and(circles_2 is not None):

            # 将圆心坐标转换为整数类型
            circles_2 = np.uint16(np.around(circles_2))



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

                #在circles_2 中计算圆心到点P（circles_draw[1][0],c1_match[1])距离，距离最近的圆就是c2_match

                #circles_2中的圆心到（x0，y0）的距离排序，最小的为c2_match

                if circles_2 is not None:


                    # 提取圆心坐标和半径
                    circle_centers = circles_2[:, 0, :2]
                    circle_radii = circles_2[:, 0, 2]

                    x0=circles_draw[1][0]
                    y0=c1_match[1]
                    # 计算每个圆心到给定点的欧氏距离
                    distances = np.sqrt((circle_centers[:, 0] - x0) ** 2 + (circle_centers[:, 1] - y0) ** 2)

                    # 找到距离最小的圆的索引
                    min_distance_index = np.argmin(distances)

                    # 提取最近的圆的圆心坐标和半径
                    #c2_match = circles_2[min_distance_index, 0, :2].append(circles_2[min_distance_index, 0, 2])
                    c2_match = circles_2[min_distance_index, 0, :]



                # 如果c1_match和c2_match都不为空，则计算两个圆圆心的X距离，和draw_circle做比较，如果距离差小于draw_circle的半径的则绘制直线，否则 continue
                if (c1_match is not None) and (c2_match is not None):
                    distance = abs(c2_match[0] - c1_match[0])
                    distance_draw = abs(circles_draw[1][0] - circles_draw[0][0])

                    #识别得到的两个圆，在Y轴方向进行配对，Y轴差不会过大
                    if abs(c1_match[1]-c2_match[1])>circles_draw[0][2]*0.5:
                        print("倾角过大，不匹配")
                        continue

                    #l_match = c1_match[0] - c2_match[0]
                    #l_draw = circles_draw[0][0] - circles_draw[1][0]
                    #
                    #if abs(l_match-l_draw)>circles_draw[0][2]*0.2:
                    #    continue

                    #识别得到的标定线长度差
                    print("X1/Y1:", c1_match[0], c1_match[1],"X2/Y2:", c2_match[0], c2_match[1],"distance/distance_draw:", distance, distance_draw,"Radius:",circles_draw[0][2])
                    if abs(distance-distance_draw) > int(circles_draw[0][2]):

                        print("距离差过大，不匹配")
                        continue


                    #根据匹配圆形区域，进一步建立roi，在此中检索对齐点

                    # 绘制最匹配的圆形
                    cx1 = c1_match[0]*2
                    cy1 = c1_match[1]*2
                    cr1 = c1_match[2]*2
                    cf1 = "circle_1_" + str(frame_count) + ".png"
                    print("cx1:", cx1, "cy1:", cy1, "cr1:", cr1)
                    # print("sizeof frame2:",frame2.shape)
                    print("cf1", cf1)
                    # 分离识别区域
                    #cv2.imwrite(cf1, frame2[ cy1 - cr1:cy1 + cr1,cx1 - cr1:cx1 + cr1])



                    #寻找中心点并绘制
                    #i_range = int(r_max * 1.2)
                    L_cx1,L_cy1 = find_MD_contour(frame2[cy1-cr1:cy1+cr1,cx1-cr1:cx1+cr1])
                    ##将ix,iy映射到frame中
                    L_cX1 = L_cx1 + cx1/2 - cr1/2
                    L_cY1 = L_cy1 + cy1/2 - cr1/2
                    #历边L_cX,L_cY,绘制中心点，半径5
                    #for i in range(len(L_cX1)):
                    #    cv2.circle(frame, (int(L_cX1[i]), int(L_cY1[i])), 5, (255, 255, 0), 2)

                    #cv2.circle(frame, (iX, iY), 20, (255, 255, 0), 2)


                    cx2=c2_match[0]*2
                    cy2=c2_match[1]*2
                    cr2=c2_match[2]*2

                    cf2= "circle_2_"+str(frame_count)+".png"
                    print("cx2:",cx2,"cy2:",cy2,"cr2:",cr2)
                    #print("sizeof frame2:",frame2.shape)
                    print("cf2",cf2)
                    #分离识别区域
                    #cv2.imwrite(cf2,frame2[cy2-cr2:cy2+cr2,cx2-cr2:cx2+cr2])

                    L_cx2,L_cy2 = find_MD_contour(frame2[cy2-cr2:cy2+cr2,cx2-cr2:cx2+cr2])
                    L_cX2 = L_cx2 + cx2/2 - cr2/2
                    L_cY2 = L_cy2 + cy2/2 - cr2/2
                    # 历边L_cX,L_cY,绘制中心点，半径5
                    #for i in range(len(L_cX)):
                    #   cv2.circle(frame, (int(L_cX2[i]), int(L_cY2[i])), 5, (255, 255, 0), 2)


                    if L_cX2 is None or L_cX1 is None:
                        continue
                    #点1的系列（L_cx1,L_cy1)和点2的系列（L_cx2,L_cy2)进行交叉匹配，其连线长度与绘制的圆心线draw_circle夹角最小的为匹配的一对点
                    #计算两个点的连线长度，以及与draw_circle的夹角
                    #计算两个点的连线长度，以及与draw_circle的夹角
                    min_angle = float('inf')
                    min_dist = float('inf')
                    min_pair = None
                    for i in range(len(L_cX1)):
                        for j in range(len(L_cX2)):
                            dist = np.sqrt((L_cX1[i] - L_cX2[j]) ** 2 + (L_cY1[i] - L_cY2[j]) ** 2)
                            angle = np.abs(np.arctan((L_cY1[i] - L_cY2[j]) / (L_cX1[i] - L_cX2[j])) - np.arctan((circles_draw[0][1] - circles_draw[1][1]) / (circles_draw[0][0] - circles_draw[1][0])))
                            if angle < min_angle :## dist < min_dist:
                                min_angle = angle
                                #min_dist = dist
                                min_pair = (i, j)

                    if min_pair is not None:


                        MD1X,MD1Y = int(L_cX1[min_pair[0]]), int(L_cY1[min_pair[0]])
                        MD2X,MD2Y = int(L_cX2[min_pair[1]]), int(L_cY2[min_pair[1]])
                        print("MD1X:",MD1X,"MD1Y:",MD1Y,"MD2X:",MD2X,"MD2Y:",MD2Y)
                        cv2.line(frame, (MD1X,MD1Y),(MD2X,MD2Y), (255, 255, 244), 2)
                        cv2.circle(frame, (MD1X,MD1Y), 5, (0, 255, 0), 2)
                        cv2.circle(frame, (MD2X,MD2Y), 5, (0, 255, 0), 2)

                    else:
                        print("匹配失败")

                    #图面绘制出匹配圆形

                    color = (0, 255, 0)
                    #cv2.line(frame, (int(c1_match[0]), int(c1_match[1])), (int(c2_match[0]), int(c2_match[1])),(0, 255, 0), 2)

                    fps = line_count
                    cv2.putText(frame, "FPS:{}".format(fps), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.circle(frame, (int(c1_match[0]), int(c1_match[1])), int(c1_match[2]), (0, 255, 0), 2)  # 绘制圆
                    cv2.putText(frame, "C1_R", (int(c1_match[0]), int(c1_match[1]*1.2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,cv2.LINE_AA)  # 添加文本

                    cv2.circle(frame, (int(c2_match[0]), int(c2_match[1])), int(c2_match[2]), (0, 0, 255), 2)  # 绘制圆
                    cv2.putText(frame, "C2_R", (int(c2_match[0]), int(c2_match[1]*1.2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)  # 添加文本



            # 水平合并两张图片
            merged_image = cv2.hconcat((cv2.cvtColor(edges_1, cv2.COLOR_GRAY2BGR), frame,cv2.cvtColor(edges_2, cv2.COLOR_GRAY2BGR)))
            # 显示结果
            cv2.imshow('Frame with Detected Circles', merged_image)
            #cv2.imshow('thresh', thresh)

        line_count = line_count + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'键退出
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
