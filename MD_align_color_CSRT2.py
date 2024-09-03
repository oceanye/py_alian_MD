import math

import cv2
import numpy as np
import tkinter as tk
import threading
import camera_utils
from tkinter import ttk

from py_arduino_motor import MotorControl, MotorControlApp, run_gui

# 全局变量
colors = []
zoom_factor = 4
zoom_size = 150
avg_color_block_size = 100  # 平均颜色块的大小
avg_color = None
is_picking_color = True  # 标识是否在拾取颜色状态
is_draw_cut_line= True # 表示是否在绘制切割标记线

# 新增全局变量
picker_dist = 0
avg_color_display = None
lower_tolerance_display = None
upper_tolerance_display = None
picker_dist_label = None

avg_color = np.array([0, 0, 0])  # 初始化为黑色

# 全局变量用于滑块和复选框
area_min = 50
area_max = 200
color_tolerance = 20
coord_threshold = 50
use_x_coord = False
dilation_size = 5
color_regions_history = []


click_count = 0
first_click_pos = None
second_click_pos = None
cut_points = [None,None]

dist_offset = 50
mask_roi = []
boundary =[0,0,0,0]

cut_start_point=(0,0)
cut_end_point=(0,0)

extended_start=(0,0)
extended_end =(0,0)

motor_connect = False

dist_history = []
#min_distance = 0



def zoom_effect(event, x, y, flags, param):
    """鼠标事件的回调函数，用于显示放大效果和选择颜色"""
    # 调整zoom_effect 显示比例，基于 size_factor

    global colors, frame, zoom_img, avg_color_img, click_count, first_click_pos, second_click_pos, picker_dist

    if event == cv2.EVENT_MOUSEMOVE:
        zoom_img = frame.copy()
        h, w = frame.shape[:2]

        x_start = max(0, x - zoom_size // (2 * zoom_factor))
        x_end = min(w, x + zoom_size // (2 * zoom_factor))
        y_start = max(0, y - zoom_size // (2 * zoom_factor))
        y_end = min(h, y + zoom_size // (2 * zoom_factor))

        zoom_area = frame[y_start:y_end, x_start:x_end]
        zoomed = cv2.resize(zoom_area, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)

        cv2.rectangle(zoom_img, (x - zoom_size // 2, y - zoom_size // 2),
                      (x + zoom_size // 2, y + zoom_size // 2), (0, 255, 0), 4)
        zoom_img[y - zoom_size // 2:y + zoom_size // 2, x - zoom_size // 2:x + zoom_size // 2] = zoomed

    elif event == cv2.EVENT_LBUTTONDOWN:
        click_count += 1
        color = frame[y, x]
        colors.append(color)
        print(f"选中的颜色: {color}")

        if click_count == 1:
            first_click_pos = (x, y)
        elif click_count == 2:
            second_click_pos = (x, y)
            picker_dist = np.sqrt((second_click_pos[0] - first_click_pos[0])**2 +
                                  (second_click_pos[1] - first_click_pos[1])**2)
            print(f"两次点击之间的距离: {picker_dist}")
            click_count = 0
            h, w = frame.shape[:2]
            update_mask_roi(h, w )
            update_avg_color()

            first_click_pos = None
            is_picking_color = False



def update_avg_color_display():
    """更新GUI中显示平均颜色的色块"""
    global avg_color_display, avg_color
    if avg_color_display and avg_color is not None:
        color = f'#{int(avg_color[2]):02x}{int(avg_color[1]):02x}{int(avg_color[0]):02x}'
        avg_color_display.delete("all")  # 清除之前的内容
        avg_color_display.create_rectangle(0, 0, 100, 100, fill=color, outline="")
        avg_color_display.create_text(50, 50, text=f"RGB: {tuple(avg_color)}", fill="white" if sum(avg_color) < 384 else "black")


def refresh_parameters():
    """刷新参数"""
    global area_min, area_max, color_tolerance, coord_threshold, use_x_coord, dilation_size

    area_min = area_min_var.get()
    area_max = area_max_var.get()
    color_tolerance = color_tolerance_var.get()
    coord_threshold = coord_threshold_var.get()
    use_x_coord = bool(use_x_coord_var.get())
    dilation_size = 2 * dilation_size_var.get() + 3
    dist_offset = dist_offset_var.get()


def reselect_cut_callback():

    global is_draw_cut_line
    is_draw_cut_line=True
    print("重新标定切割投影")

def reselect_colors_callback():
    """重新拾取颜色的回调函数"""
    global colors, is_picking_color
    colors = []  # 清空已选择的颜色
    is_picking_color = True  # 重新进入颜色拾取模式
    print("重新进入绘制切割对齐线")


def create_gui():
    """创建 Tkinter GUI 界面"""
    global area_min_var, area_max_var, color_tolerance_var, coord_threshold_var, use_x_coord_var, dilation_size_var, dist_offset_var
    global avg_color_display, lower_tolerance_display, upper_tolerance_display, picker_dist_label

    root = tk.Tk()
    root.title("参数调整")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    row = 0

    # 色彩容差
    # 色彩容差
    ttk.Label(main_frame, text="色彩容差").grid(row=row, column=0, sticky=tk.W)
    color_tolerance_var = tk.IntVar(value=color_tolerance)
    color_tolerance_scale = ttk.Scale(main_frame, from_=0, to=50, orient=tk.HORIZONTAL,
                                      variable=color_tolerance_var, length=200)
    color_tolerance_scale.grid(row=row, column=1)
    color_tolerance_scale.bind("<Motion>", update_tolerance_displays)  # 添加这行
    ttk.Label(main_frame, textvariable=color_tolerance_var).grid(row=row, column=2)
    row += 1

    # 膨胀尺寸
    ttk.Label(main_frame, text="膨胀尺寸").grid(row=row, column=0, sticky=tk.W)
    dilation_size_var = tk.IntVar(value=(dilation_size - 3) // 2)
    ttk.Scale(main_frame, from_=0, to=7, orient=tk.HORIZONTAL, variable=dilation_size_var, length=200).grid(row=row, column=1)
    ttk.Label(main_frame, textvariable=dilation_size_var).grid(row=row, column=2)
    row += 1


    # 最小面积
    ttk.Label(main_frame, text="最小面积").grid(row=row, column=0, sticky=tk.W)
    area_min_var = tk.IntVar(value=area_min)
    ttk.Scale(main_frame, from_=0, to=500, orient=tk.HORIZONTAL, variable=area_min_var, length=200).grid(row=row, column=1)
    ttk.Label(main_frame, textvariable=area_min_var).grid(row=row, column=2)
    row += 1

    # 最大面积
    ttk.Label(main_frame, text="最大面积").grid(row=row, column=0, sticky=tk.W)
    area_max_var = tk.IntVar(value=area_max)
    ttk.Scale(main_frame, from_=0, to=500, orient=tk.HORIZONTAL, variable=area_max_var, length=200).grid(row=row, column=1)
    ttk.Label(main_frame, textvariable=area_max_var).grid(row=row, column=2)
    row += 1

    # 水平分割线
    ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=10)
    row += 1

    # 坐标容差
    ttk.Label(main_frame, text="坐标容差").grid(row=row, column=0, sticky=tk.W)
    coord_threshold_var = tk.IntVar(value=coord_threshold)
    ttk.Scale(main_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=coord_threshold_var, length=200).grid(row=row, column=1)
    ttk.Label(main_frame, textvariable=coord_threshold_var).grid(row=row, column=2)
    row += 1





    # 开启Mask遮罩
    #ttk.Label(main_frame, text="ROI范围").grid(row=row, column=0, sticky=tk.W)
    #dist_offset_var = tk.IntVar(value=dist_offset)
    #ttk.Scale(main_frame, from_=dist_offset, to=dist_offset*10, orient=tk.HORIZONTAL, variable=dist_offset_var, length=200).grid(row=row, column=1)
    #ttk.Label(main_frame, textvariable=dist_offset_var).grid(row=row, column=2)
    #row +=1

    # 切换X方向
    #use_x_coord_var = tk.IntVar(value=use_x_coord)
    #ttk.Checkbutton(main_frame, text="切换X方向", variable=use_x_coord_var).grid(row=row, column=0, columnspan=3, sticky=tk.W)
    #row += 1

    # 按钮
    ttk.Button(main_frame, text="刷新", command=refresh_parameters).grid(row=row, column=0, pady=10)
    ttk.Button(main_frame, text="重新拾取颜色", command=reselect_colors_callback).grid(row=row, column=1, pady=10)
    ttk.Button(main_frame, text = "绘制基准线", command = reselect_cut_callback).grid(row=row, column = 2, pady =10)
    #ttk.Button(main_frame, text = "标定切割位置",command =recut_line_callback).grid(row = row,column =2,pady=10)
    row += 1

    # 显示颜色的色块
    color_frame = ttk.Frame(main_frame)
    color_frame.grid(row=row, column=0, columnspan=3, pady=10)

    lower_tolerance_display = tk.Canvas(color_frame, width=50, height=50)
    lower_tolerance_display.grid(row=0, column=0, padx=5)

    avg_color_display = tk.Canvas(color_frame, width=100, height=100)
    avg_color_display.grid(row=0, column=1, padx=5)

    upper_tolerance_display = tk.Canvas(color_frame, width=50, height=50)
    upper_tolerance_display.grid(row=0, column=2, padx=5)

    update_avg_color_display()
    row += 1


    # 显示picker_dist
    ttk.Label(main_frame, text="Picker Distance:").grid(row=row, column=0, sticky=tk.W)
    picker_dist_label = ttk.Label(main_frame, text="0")
    picker_dist_label.grid(row=row, column=1, sticky=tk.W)
    row += 1

    # 更新picker_dist显示
    def update_picker_dist():
        picker_dist_label.config(text=f"{picker_dist:.2f}")
        root.after(100, update_picker_dist)

    update_picker_dist()


    root.mainloop()


def update_tolerance_displays(event=None):
    """更新显示色彩容差上下限的色块"""
    global lower_tolerance_display, upper_tolerance_display, avg_color, color_tolerance_var
    if lower_tolerance_display and upper_tolerance_display and avg_color is not None:
        tolerance = color_tolerance_var.get()

        lower_color = np.clip(avg_color - tolerance, 0, 255).astype(int)
        upper_color = np.clip(avg_color + tolerance, 0, 255).astype(int)

        lower_hex = f'#{int(lower_color[2]):02x}{int(lower_color[1]):02x}{int(lower_color[0]):02x}'
        upper_hex = f'#{int(upper_color[2]):02x}{int(upper_color[1]):02x}{int(upper_color[0]):02x}'

        lower_tolerance_display.delete("all")
        lower_tolerance_display.create_rectangle(0, 0, 50, 50, fill=lower_hex, outline="")

        upper_tolerance_display.delete("all")
        upper_tolerance_display.create_rectangle(0, 0, 50, 50, fill=upper_hex, outline="")


def update_avg_color_display():
    """更新GUI中显示平均颜色的色块"""
    global avg_color_display, avg_color
    if avg_color_display and avg_color is not None:
        color = f'#{int(avg_color[2]):02x}{int(avg_color[1]):02x}{int(avg_color[0]):02x}'
        avg_color_display.delete("all")
        avg_color_display.create_rectangle(0, 0, 100, 100, fill=color, outline="")
        avg_color_display.create_text(50, 50, text=f"RGB: {tuple(avg_color)}",
                                      fill="white" if sum(avg_color) < 384 else "black")
    update_tolerance_displays()

def update_avg_color():
    """更新平均颜色"""
    global avg_color, colors
    if colors:
        avg_color = np.mean(colors, axis=0).astype(int)
        print(f"平均颜色: {avg_color}")
        update_avg_color_display()


def update_mask_roi(h, w):
    global first_click_pos,second_click_pos,dist_offset,mask_roi,use_x_coord,boundary

    print("first_click_pos", first_click_pos)
    print("second_click_pos", second_click_pos)

    if True:#if first_click_pos is not None and second_click_pos is not None:
        [x1, y1] = first_click_pos
        [x2, y2] = second_click_pos

        print("x1",x1)
        print("first_click_pos",first_click_pos)
        # 重置 mask_roi
        mask_roi=np.zeros([h, w],dtype=frame.dtype)

        # Calculate the top-left and bottom-right corners of the ROI
        if use_x_coord:
            # 创建两个 Y 向条状区域
            y1_min = max(0, y1 - dist_offset)
            y1_max = min(mask_roi.shape[0] - 1, y1 + dist_offset)
            y2_min = max(0, y2 - dist_offset)
            y2_max = min(mask_roi.shape[0] - 1, y2 + dist_offset)

            boundary = [y1_min, y1_max, y2_min, y2_max]

            mask_roi[y1_min:y1_max + 1, :] = 1
            mask_roi[y2_min:y2_max + 1, :] = 1

        else:
            # 创建两个 X 向条状区域
            x1_min = max(0, x1 - dist_offset)
            x1_max = min(mask_roi.shape[1] - 1, x1 + dist_offset)
            x2_min = max(0, x2 - dist_offset)
            x2_max = min(mask_roi.shape[1] - 1, x2 + dist_offset)

            boundary = [x1_min,x1_max,x2_min,x2_max]

            mask_roi[:, x1_min:x1_max + 1] = 1
            mask_roi[:, x2_min:x2_max + 1] = 1


    print("Set Boundary ", boundary)

def draw_roi_boundaries(frame, boundary, use_x_coord=True, color=(0, 255, 255), thickness=2):
    result = frame.copy()
    height, width = frame.shape[:2]
    v1_min, v1_max, v2_min, v2_max = boundary

    if use_x_coord:
        # 绘制水平线
        cv2.line(result, (0, v1_min), (width, v1_min), color, thickness)
        cv2.line(result, (0, v1_max), (width, v1_max), color, thickness)
        cv2.line(result, (0, v2_min), (width, v2_min), color, thickness)
        cv2.line(result, (0, v2_max), (width, v2_max), color, thickness)

    else:

        # 绘制垂直线
        cv2.line(result, (v1_min, 0), (v1_min, height), color, thickness)
        cv2.line(result, (v1_max, 0), (v1_max, height), color, thickness)
        cv2.line(result, (v2_min, 0), (v2_min, height), color, thickness)
        cv2.line(result, (v2_max, 0), (v2_max, height), color, thickness)
    return result




def draw_histogram(history, frame):
    """在视频帧上绘制颜色区域计数的历史记录直方图，支持正负值，根据实际最大最小值调整比例"""
    hist_height = 400
    hist_width = 600
    hist = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)

    if len(history) > hist_width:
        history = history[-hist_width:]

    if not history:
        return frame

    min_val = min(history)
    max_val = max(history)
    value_range = max_val - min_val

    if value_range < 1e-6:  # 避免除以接近零的值
        value_range = 1

    zero_line = int(hist_height * (max_val / value_range))
    zero_line = max(10, min(hist_height - 10, zero_line))  # 确保零线不会太靠近边缘

    for i, val in enumerate(history):
        normalized_val = (val - min_val) / value_range
        height = int(hist_height * normalized_val)
        cv2.line(hist, (i, zero_line), (i, height), (0, 255, 0), 1)

    # 添加背景和边框
    cv2.rectangle(hist, (0, 0), (hist_width - 1, hist_height - 1), (255, 255, 255), 1)

    # 绘制零线
    cv2.line(hist, (0, zero_line), (hist_width, zero_line), (255, 255, 255), 1)

    # 在直方图上添加标签
    cv2.putText(hist, "Distance History", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(hist, f"Max: {max_val:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(hist, f"Min: {min_val:.2f}", (10, hist_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 计算直方图在帧中的位置
    x_offset = 100
    y_offset = frame.shape[0] - hist_height - 50

    # 使用 addWeighted 来叠加直方图到原始帧上
    roi = frame[y_offset:y_offset + hist_height, x_offset:x_offset + hist_width]
    dst = cv2.addWeighted(roi, 1, hist, 0.7, 0)
    frame[y_offset:y_offset + hist_height, x_offset:x_offset + hist_width] = dst

    return frame
def create_side_by_side_display(frame1, frame2, window_name='Side by Side Display', scale_factor=0.5):
    # 检查输入帧是否为空
    if frame1 is None or frame2 is None:
        print("Error: One or both input frames are None")
        return None

    # 打印帧的形状和类型，用于调试
    print(f"Frame1 shape: {frame1.shape}, dtype: {frame1.dtype}")
    print(f"Frame2 shape: {frame2.shape}, dtype: {frame2.dtype}")

    # 确保两个帧的高度相同
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    h = min(h1, h2)

    try:
        # 调整帧的大小以匹配高度
        frame1_resized = cv2.resize(frame1, (int(w1 * h / h1), h))
        frame2_resized = cv2.resize(frame2, (int(w2 * h / h2), h))

        print(f"Resized frame1 shape: {frame1_resized.shape}")
        print(f"Resized frame2 shape: {frame2_resized.shape}")

        # 水平连接两个帧
        combined_frame = np.hstack((frame1_resized, frame2_resized))

        print(f"Combined frame shape: {combined_frame.shape}")

        # 缩放combined_frame
        combined_frame_scaled = cv2.resize(combined_frame, None, fx=scale_factor, fy=scale_factor)

        print(f"Scaled combined frame shape: {combined_frame_scaled.shape}")

        # 显示结果
        cv2.imshow(window_name, combined_frame_scaled)

        return combined_frame_scaled
    except Exception as e:
        print(f"Error in create_side_by_side_display: {e}")
        import traceback
        traceback.print_exc()
        return None



def dist_group(group):
    global picker_dist
    p1=group[0]
    p2=group[1]

    d = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    if abs(d/picker_dist-1)<0.2:
        return True
    else:
        return False


def draw_cut_line(event,x,y,flags, param):
    global is_draw_cut_line
    global cut_start_point,cut_end_point

    is_draw_cut_line = False
    #click_count = 0
    if event == cv2.EVENT_LBUTTONDOWN :
        if cut_start_point[0]+cut_start_point[1]==0:

            cut_start_point = (x, y)
            #click_count =click_count +1

            #cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        else :
            cut_end_point = (x,y)
            #click_count =0
            #cv2.line(frame, cut_start_point, cut_end_point,(0, 255, 0), 2)
            #cv2.imshow(frame)



    print("Start:",cut_start_point,"End",cut_end_point)
    return


def get_extended_line(frame, cut_start_point, cut_end_point):
    height, width = frame.shape[:2]

    x1, y1 = map(float, cut_start_point)
    x2, y2 = map(float, cut_end_point)

    # 检查点是否相同
    if x1 == x2 and y1 == y2:
        raise ValueError("起点和终点相同，无法确定直线")

    if x2 - x1 == 0:  # 垂直线的情况
        return (int(x1), 0), (int(x1), height)

    m = (y2 - y1) / (x2 - x1)  # 斜率
    b = y1 - m * x1  # y轴截距

    # 计算延长线与图像边界的交点
    left_y = m * 0 + b
    right_y = m * width + b
    top_x = -b / m if m != 0 else 0
    bottom_x = (height - b) / m if m != 0 else width

    # 确定延长线的起点和终点
    candidates = [
        (0, left_y),
        (width, right_y),
        (top_x, 0),
        (bottom_x, height)
    ]

    valid_points = [
        (int(round(x)), int(round(y)))
        for x, y in candidates
        if 0 <= x <= width and 0 <= y <= height
    ]

    if len(valid_points) < 2:
        raise ValueError("无法确定有效的延长线端点")

    extended_start, extended_end = valid_points[0], valid_points[-1]

    return extended_start, extended_end


def nearest_group_movement(groups, cut_start_point, cut_end_point):
    def point_line_distance(point, line_point1, line_point2):
        x0, y0 = point
        x1, y1 = line_point1
        x2, y2 = line_point2

        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        distance = (A * x0 + B * y0 + C) / np.sqrt(A ** 2 + B ** 2)

        # 如果直线倾斜向下，我们需要反转符号以确保Y正向为正
        if y2 < y1:
            distance = -distance

        return distance

    nearest_group = None
    min_distance = 9999


    if len(groups)>0:
        for i, group in enumerate(groups):
            avg_point = ((group[0][0] + group[1][0]) / 2, (group[0][1] + group[1][1]) / 2)
            distance = point_line_distance(avg_point, cut_start_point, cut_end_point)

            if abs(distance) < abs(min_distance):
                min_distance = distance
                nearest_group = groups[i]
    else:
        nearest_group=([0,0],[0,0])
        min_distance=0

    #print("groups",groups)

    if min_distance>0:
        direction = "UP"+"^"
    else:
        direction = "Down"+"v"

    return direction,nearest_group, round(min_distance,1)


def initialize_tracker(point1, point2, width_offset):
    global tracker
    x1, y1 = point1
    x2, y2 = point2

    # 计算矩形的左上角和右下角坐标
    left = min(x1, x2) - width_offset
    top = min(y1, y2)
    right = max(x1, x2) + width_offset
    bottom = max(y1, y2)

    # 创建跟踪器
    tracker = cv2.TrackerCSRT_create()
    bbox = (left, top, right - left, bottom - top)
    success = tracker.init(frame, bbox)

    if success:
        print("Tracker initialized successfully")
    else:
        print("Failed to initialize tracker")
        tracker = None

def main():
    """主函数，处理视频并应用颜色过滤和标记"""
    global frame, zoom_img, avg_color_img, colors, avg_color, is_picking_color, click_count ,mask_roi ,boundary,is_draw_cut_line
    global cut_start_point,cut_end_point
    global motor_connect,motor_control
    global dist_history

    # 启动 Tkinter GUI 线程
    gui_thread = threading.Thread(target=create_gui, daemon=True)
    gui_thread.start()

    camera = camera_utils.open_camera()

    try:
        motor_control = MotorControl()
        motor_control.init_serial("COM3")  # 请替换为正确的串口名称
        motor_connect = True
    except Exception as e:
        print("motor not connect with COM3")



    # 打开视频文件
    # video = cv2.VideoCapture(video_path)

    if camera is None:
        print("无法打开摄像头")
        return

    while True:

        frame = camera_utils.get_frame(camera)
        frame_org = frame.copy()


        if is_draw_cut_line:
            #frame = camera_utils.get_frame(camera)

            is_draw_cut_line = False

            cut_start_point=(0,0)
            cut_end_point=(0,0)
            cv2.namedWindow('Cut Line', cv2.WINDOW_KEEPRATIO)
            cv2.setMouseCallback('Cut Line', draw_cut_line)


            print("请在当前帧上绘制切割对齐线，按Enter键结束选择")

            #cv2.namedWindow('Cut Line', cv2.WINDOW_AUTOSIZE)
            #cv2.resizeWindow('Cut Line',800,600)
            while True:

                cv2.imshow('Cut Line', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 13 or cut_start_point[0]*cut_end_point[0] !=0:  # Enter key
                    break


            print(f"切割点坐标: {cut_points}")

            cv2.destroyWindow('Cut Line')

            extended_start,extended_end = get_extended_line(frame_org,cut_start_point,cut_end_point)



        # 绘制切割位置线
        cv2.line(frame_org,extended_start,extended_end ,(0,0,255),2)



        if is_picking_color:
            # ret, frame = video.read()



            #frame = camera_utils.get_frame(camera)


            if 0 == 1:
                print("无法读取视频或视频读取完毕")
                return

            zoom_img = frame.copy()
            avg_color_img = np.zeros((frame.shape[0], avg_color_block_size, 3), dtype=np.uint8)

            cv2.namedWindow('Pick Color',cv2.WINDOW_KEEPRATIO) #cv2.WINDOW_NORMAL |
            cv2.setMouseCallback('Pick Color', zoom_effect)

            print("请在当前帧上选择颜色点，按Enter键结束选择")

            print("zoom_img", zoom_img.shape)
            print("avg_color", avg_color_img.shape)

            while True:
                display_img = np.hstack((zoom_img, avg_color_img))
                cv2.imshow('Pick Color', display_img)
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter key
                    break

            if not colors:
                print("未选择颜色，程序退出")
                return

            avg_color = np.mean(colors, axis=0).astype(int)
            print(f"平均颜色: {avg_color}")

            cv2.destroyWindow('Pick Color')
            is_picking_color = False






        # 使用 bitwise_and 操作
        frame_temp = cv2.bitwise_and(frame, frame,mask = mask_roi)

        frame = cv2.resize(frame_temp, (0, 0), fx=1.0, fy=1.0)

        if frame is None:
            print("无法获取帧")
            continue

        # 创建颜色掩码
        lower = np.array([max(0, c - color_tolerance) for c in avg_color])
        upper = np.array([min(255, c + color_tolerance) for c in avg_color])
        mask = cv2.inRange(frame, lower, upper)

        # 创建膨胀核
        kernel = np.ones((dilation_size, dilation_size), np.uint8)

        # 对filtered_frame进行膨胀操作
        dilated = cv2.dilate(mask, kernel, iterations=1)

        # 找到轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建新的掩码，只保留面积在area_min和area_max之间的区域
        new_mask = np.zeros_like(mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area_min <= area <= area_max:
                cv2.drawContours(new_mask, [contour], 0, 255, -1)

        # 应用新的掩码到filtered_frame
        filtered_frame = cv2.bitwise_and(frame, frame, mask=new_mask)

        # 计算识别到的颜色区域个数
        # 在主循环中，绘制直方图之前更新 color_regions_history
        color_regions_history = []  # 添加这行

        color_regions_count = len([c for c in contours if area_min <= cv2.contourArea(c) <= area_max])
        color_regions_history.append(color_regions_count)

        # 限制 history 的大小以防止内存问题
        if len(color_regions_history) > 300:  # 假设我们只保留最近300帧的历史
            color_regions_history = color_regions_history[-300:]




        centers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area_min <= area <= area_max:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centers.append((cX, cY))

        # 对中心点进行分组
        centers.sort(key=lambda x: x[0] if use_x_coord else x[1])
        groups = []
        current_group = []

        for center in centers:
            if not current_group or abs(
                    center[0 if use_x_coord else 1] - current_group[-1][0 if use_x_coord else 1]) < coord_threshold:
                current_group.append(center)
            else:
                groups.append(current_group)
                current_group = [center]

        if current_group :
            groups.append(current_group)



        groups_line = []
        # 在当前帧上绘制
        for group in groups:
            if len(group) == 2 and dist_group(group):
                cv2.line(frame_org, group[0], group[1], (0, 255, 0), 2)
                groups_line.append(group)
            elif len(group) > 2:
                for point in group:
                    cv2.circle(frame_org, point, 30, (0, 255, 0), 2)

        # 显示识别到的颜色区域个数
        cv2.putText(frame_org, f"Color Regions: {color_regions_count}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

        # 计算最近的一组点

        #print("get_groups", groups_line)
        direction ,nearest_group, min_distance = nearest_group_movement(groups_line, cut_start_point, cut_end_point)

        #print("nearest point",nearest_group)

        # 记录尺寸变化

        #dist_history = []  # 添加这行


        dist_history.append(min_distance)

        # 限制 history 的大小以防止内存问题
        if len(dist_history) > 300:  # 假设我们只保留最近300帧的历史
            dist_history = dist_history[-300:]



        # 在视频帧上绘制历史记录直方图
        draw_histogram(dist_history, frame_org)




        #绘制最近的匹配线
        cv2.line(frame_org,nearest_group[0],nearest_group[1],(255,0,255),5)

        #匹配线的距离
        cv2.putText(frame_org, f"Dist: {min_distance}", (nearest_group[1][0]+300,nearest_group[1][1]+100),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
        #识别点移动方向
        cv2.putText(frame_org, f"Direction: {direction}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

        if motor_connect ==True:
            if min_distance > 5:
                motor_control.move_to_position(4, 180)  # 移动第一个电机到90度位置
            elif min_distance< -5:
                motor_control.move_to_position(4,0)
            else:
                motor_control.move_to_position(4,90)
        # 在视频帧上绘制历史记录直方图
        #draw_histogram(color_regions_history, frame)

        # 显示结果
        # cv2.imshow('Original Video', frame)
        # cv2.imshow('Filtered Video', filtered_frame)

        # 创建并显示并排视图
        #print(f"Original frame shape: {frame.shape}, dtype: {frame.dtype}")
        #print(f"Filtered frame shape: {filtered_frame.shape}, dtype: {filtered_frame.dtype}")
        #print(f"Filtered frame min: {np.min(filtered_frame)}, max: {np.max(filtered_frame)}")

        # 尝试归一化 filtered_frame
        if filtered_frame.dtype != np.uint8:
            filtered_frame = cv2.normalize(filtered_frame, None, 0, 255, cv2.NORM_MINMAX)
            filtered_frame = filtered_frame.astype(np.uint8)

        if len(filtered_frame.shape) == 2:
            filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)

        # 单独显示 filtered_frame 进行检查
        # cv2.imshow('Filtered Frame', filtered_frame)
        print("boundary",boundary)
        frame = draw_roi_boundaries(frame_org,boundary,use_x_coord)
        frame = cv2.line(frame,cut_points[0],cut_points[1],(255,0,0),thickness=1)
        # 创建并显示并排视图
        combined_frame = create_side_by_side_display(frame, filtered_frame, 'Original and Filtered Video',
                                                     scale_factor=0.4)

        if combined_frame is None:
            print("Failed to create side-by-side display")
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    # 关闭摄像头
    camera.MV_CC_CloseDevice()
    camera.MV_CC_DestroyHandle()


if __name__ == "__main__":
    #video_path = 'video(3).mp4'  # 可以更改为其他视频路径


    main()