import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading

# 全局变量
colors = []
zoom_factor = 4
zoom_size = 150
avg_color_block_size = 100  # 平均颜色块的大小

# 全局变量用于滑块和复选框
area_threshold = 200
color_tolerance = 20
coord_threshold = 50
use_x_coord = False
dilation_size = 5

def zoom_effect(event, x, y, flags, param):
    """鼠标事件的回调函数，用于显示放大效果和选择颜色"""
    global colors, frame, zoom_img, avg_color_img

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
                      (x + zoom_size // 2, y + zoom_size // 2), (0, 255, 0), 2)
        zoom_img[y - zoom_size // 2:y + zoom_size // 2, x - zoom_size // 2:x + zoom_size // 2] = zoomed

    elif event == cv2.EVENT_LBUTTONDOWN:
        color = frame[y, x]
        colors.append(color)
        print(f"选中的颜色: {color}")
        update_avg_color()

def update_avg_color():
    """更新平均颜色图像"""
    global avg_color_img, colors
    if colors:
        avg_color = np.mean(colors, axis=0).astype(int)
        avg_color_img = np.full((frame.shape[0], avg_color_block_size, 3), avg_color, dtype=np.uint8)
        cv2.putText(avg_color_img, f"RGB: {avg_color}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) if np.mean(avg_color) < 128 else (0, 0, 0), 1)
    else:
        avg_color_img = np.zeros((frame.shape[0], avg_color_block_size, 3), dtype=np.uint8)

def refresh_parameters():
    """刷新参数"""
    global area_threshold, color_tolerance, coord_threshold, use_x_coord, dilation_size

    area_threshold = area_threshold_var.get()
    color_tolerance = color_tolerance_var.get()
    coord_threshold = coord_threshold_var.get()
    use_x_coord = bool(use_x_coord_var.get())
    dilation_size = 2 * dilation_size_var.get() + 3

def create_gui():
    """创建 Tkinter GUI 界面"""
    global area_threshold_var, color_tolerance_var, coord_threshold_var, use_x_coord_var, dilation_size_var

    root = tk.Tk()
    root.title("参数调整")

    tk.Label(root, text="Area Threshold").pack()
    area_threshold_var = tk.IntVar(value=area_threshold)
    tk.Scale(root, from_=0, to=300, orient=tk.HORIZONTAL, variable=area_threshold_var).pack()

    tk.Label(root, text="Color Tolerance").pack()
    color_tolerance_var = tk.IntVar(value=color_tolerance)
    tk.Scale(root, from_=0, to=50, orient=tk.HORIZONTAL, variable=color_tolerance_var).pack()

    tk.Label(root, text="Coord Threshold").pack()
    coord_threshold_var = tk.IntVar(value=coord_threshold)
    tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, variable=coord_threshold_var).pack()

    tk.Label(root, text="Dilation Size").pack()
    dilation_size_var = tk.IntVar(value=(dilation_size - 3) // 2)
    tk.Scale(root, from_=0, to=3, orient=tk.HORIZONTAL, variable=dilation_size_var).pack()

    use_x_coord_var = tk.IntVar(value=use_x_coord)
    tk.Checkbutton(root, text="Use X Coord", variable=use_x_coord_var).pack()

    tk.Button(root, text="刷新", command=refresh_parameters).pack()

    root.mainloop()

def main(video_path):
    """主函数，处理视频并应用颜色过滤和标记"""
    global frame, zoom_img, avg_color_img

    # 启动 Tkinter GUI 线程
    gui_thread = threading.Thread(target=create_gui, daemon=True)
    gui_thread.start()

    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    if not ret:
        print("无法读取视频")
        return

    zoom_img = frame.copy()
    avg_color_img = np.zeros((frame.shape[0], avg_color_block_size, 3), dtype=np.uint8)

    cv2.namedWindow('Pick Color')
    cv2.setMouseCallback('Pick Color', zoom_effect)

    print("请点击选择多个颜色点，按Enter键结束选择")

    # 选择颜色点
    while True:
        display_img = np.hstack((zoom_img, avg_color_img))
        cv2.imshow('Pick Color', display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break

    cv2.destroyAllWindows()

    if not colors:
        print("未选择颜色，程序退出")
        return

    avg_color = np.mean(colors, axis=0).astype(int)
    print(f"平均颜色: {avg_color}")

    # 获取视频的原始尺寸
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置窗口大小
    cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Video', width, height)
    cv2.namedWindow('Filtered Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Filtered Video', width, height)

    # 处理每一帧
    while True:
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频到开始
            continue

        # 创建颜色掩码
        lower = np.array([max(0, c - color_tolerance) for c in avg_color])
        upper = np.array([min(255, c + color_tolerance) for c in avg_color])
        mask = cv2.inRange(frame, lower, upper)

        # 应用掩码到原始帧
        filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # 创建膨胀核
        kernel = np.ones((dilation_size, dilation_size), np.uint8)

        # 对filtered_frame进行膨胀操作
        dilated = cv2.dilate(mask, kernel, iterations=1)

        # 找到轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建新的掩码，只保留面积小于area_threshold的区域
        new_mask = np.zeros_like(mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < area_threshold:
                cv2.drawContours(new_mask, [contour], 0, 255, -1)

        # 应用新的掩码到filtered_frame
        filtered_frame = cv2.bitwise_and(frame, frame, mask=new_mask)

        # 计算识别到的颜色区域个数
        color_regions_count = len([c for c in contours if cv2.contourArea(c) < area_threshold])

        centers = []
        for contour in contours:
            if cv2.contourArea(contour) < area_threshold:
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
            if not current_group or abs(center[0 if use_x_coord else 1] - current_group[-1][0 if use_x_coord else 1]) < coord_threshold:
                current_group.append(center)
            else:
                groups.append(current_group)
                current_group = [center]

        if current_group:
            groups.append(current_group)

        # 在当前帧上绘制
        for group in groups:
            if len(group) == 2:
                cv2.line(frame, group[0], group[1], (0, 255, 0), 2)
            elif len(group) > 2:
                for point in group:
                    cv2.circle(frame, point, 30, (0, 255, 0), 2)

        # 显示识别到的颜色区域个数
        cv2.putText(frame, f"Color Regions: {color_regions_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('Original Video', frame)
        cv2.imshow('Filtered Video', filtered_frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'video(3).mp4'  # 可以更改为其他视频路径
    main(video_path)