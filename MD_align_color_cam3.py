import cv2
import numpy as np
import tkinter as tk
import threading
import camera_utils
from tkinter import ttk

# 全局变量
colors = []
zoom_factor = 4
zoom_size = 150
avg_color_block_size = 100  # 平均颜色块的大小
avg_color = None
is_picking_color = True  # 标识是否在拾取颜色状态


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


def zoom_effect(event, x, y, flags, param):
    """鼠标事件的回调函数，用于显示放大效果和选择颜色"""
    global colors, frame, zoom_img, avg_color_img, click_count, first_click_pos, picker_dist

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
            first_click_pos = None
            update_avg_color()
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


def reselect_colors_callback():
    """重新拾取颜色的回调函数"""
    global colors, is_picking_color
    colors = []  # 清空已选择的颜色
    is_picking_color = True  # 重新进入颜色拾取模式
    print("重新进入颜色拾取模式")


def create_gui():
    """创建 Tkinter GUI 界面"""
    global area_min_var, area_max_var, color_tolerance_var, coord_threshold_var, use_x_coord_var, dilation_size_var
    global avg_color_display, lower_tolerance_display, upper_tolerance_display, picker_dist_label

    root = tk.Tk()
    root.title("参数调整")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    row = 0


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

    # 坐标容差
    ttk.Label(main_frame, text="坐标容差").grid(row=row, column=0, sticky=tk.W)
    coord_threshold_var = tk.IntVar(value=coord_threshold)
    ttk.Scale(main_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=coord_threshold_var, length=200).grid(row=row, column=1)
    ttk.Label(main_frame, textvariable=coord_threshold_var).grid(row=row, column=2)
    row += 1

    # 膨胀尺寸
    ttk.Label(main_frame, text="膨胀尺寸").grid(row=row, column=0, sticky=tk.W)
    dilation_size_var = tk.IntVar(value=(dilation_size - 3) // 2)
    ttk.Scale(main_frame, from_=0, to=7, orient=tk.HORIZONTAL, variable=dilation_size_var, length=200).grid(row=row, column=1)
    ttk.Label(main_frame, textvariable=dilation_size_var).grid(row=row, column=2)
    row += 1

    # 切换X方向
    use_x_coord_var = tk.IntVar(value=use_x_coord)
    ttk.Checkbutton(main_frame, text="切换X方向", variable=use_x_coord_var).grid(row=row, column=0, columnspan=3, sticky=tk.W)
    row += 1

    # 按钮
    ttk.Button(main_frame, text="刷新", command=refresh_parameters).grid(row=row, column=0, pady=10)
    ttk.Button(main_frame, text="重新拾取颜色", command=reselect_colors_callback).grid(row=row, column=1, pady=10)
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


def draw_histogram(history, frame):
    """在视频帧上绘制颜色区域计数的历史记录直方图"""
    hist_height = 100
    hist_width = 300
    hist = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)

    if len(history) > hist_width:
        history = history[-hist_width:]

    max_val = max(history) if history else 1

    # 添加这个检查
    if max_val == 0:
        max_val = 1  # 避免除以零

    for i, val in enumerate(history):
        height = int(hist_height * val / max_val)
        cv2.line(hist, (i, hist_height), (i, hist_height - height), (0, 255, 0), 1)

    # 添加背景和边框
    cv2.rectangle(hist, (0, 0), (hist_width - 1, hist_height - 1), (255, 255, 255), 1)

    # 在直方图上添加标签
    cv2.putText(hist, "Color Regions History", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 计算直方图在帧中的位置
    x_offset = 100  # frame.shape[1] - hist_width - 10
    y_offset = 50

    # 使用 addWeighted 来叠加直方图到原始帧上
    roi = frame[y_offset:y_offset + hist_height, x_offset:x_offset + hist_width]
    dst = cv2.addWeighted(roi, 1, hist, 0.7, 0)
    frame[y_offset:y_offset + hist_height, x_offset:x_offset + hist_width] = dst


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


def main():
    """主函数，处理视频并应用颜色过滤和标记"""
    global frame, zoom_img, avg_color_img, colors, avg_color, is_picking_color, click_count


    # 启动 Tkinter GUI 线程
    gui_thread = threading.Thread(target=create_gui, daemon=True)
    gui_thread.start()

    # 打开视频文件
    # video = cv2.VideoCapture(video_path)
    camera = camera_utils.open_camera()
    if camera is None:
        print("无法打开摄像头")
        return

    while True:
        if is_picking_color:
            # ret, frame = video.read()

            frame = camera_utils.get_frame(camera)

            if 0 == 1:
                print("无法读取视频或视频读取完毕")
                return

            zoom_img = frame.copy()
            avg_color_img = np.zeros((frame.shape[0], avg_color_block_size, 3), dtype=np.uint8)

            cv2.namedWindow('Pick Color')
            cv2.setMouseCallback('Pick Color', zoom_effect)

            print("请在当前帧上选择颜色点，按Enter键结束选择")

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

        # ret, frame = video.read()
        # if not ret:
        #    video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频到开始
        #    continue
        frame = camera_utils.get_frame(camera)
        frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)

        if frame is None:
            print("无法获取帧")
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

        # 在视频帧上绘制历史记录直方图
        draw_histogram(color_regions_history, frame)

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

        # 在视频帧上绘制历史记录直方图
        # draw_histogram(color_regions_history, frame)

        # 显示结果
        # cv2.imshow('Original Video', frame)
        # cv2.imshow('Filtered Video', filtered_frame)

        # 创建并显示并排视图
        print(f"Original frame shape: {frame.shape}, dtype: {frame.dtype}")
        print(f"Filtered frame shape: {filtered_frame.shape}, dtype: {filtered_frame.dtype}")
        print(f"Filtered frame min: {np.min(filtered_frame)}, max: {np.max(filtered_frame)}")

        # 尝试归一化 filtered_frame
        if filtered_frame.dtype != np.uint8:
            filtered_frame = cv2.normalize(filtered_frame, None, 0, 255, cv2.NORM_MINMAX)
            filtered_frame = filtered_frame.astype(np.uint8)

        if len(filtered_frame.shape) == 2:
            filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)

        # 单独显示 filtered_frame 进行检查
        # cv2.imshow('Filtered Frame', filtered_frame)

        # 创建并显示并排视图
        combined_frame = create_side_by_side_display(frame, filtered_frame, 'Original and Filtered Video',
                                                     scale_factor=0.5)

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
    video_path = 'video(3).mp4'  # 可以更改为其他视频路径
    main()