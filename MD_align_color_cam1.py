import cv2
import numpy as np
import tkinter as tk
import threading
import camera_utils

# 全局变量
colors = []
zoom_factor = 4
zoom_size = 150
avg_color_block_size = 100
avg_color = None
is_picking_color = False
is_camera_open = False
frame = None
video_source = None
zoom_img = None

# 全局变量用于滑块和复选框
area_min = 50
area_max = 200
color_tolerance = 20
coord_threshold = 50
use_x_coord = False
dilation_size = 5
color_regions_history = []

def mouse_callback(event, x, y, flags, param):
    global frame, zoom_img
    if frame is not None:
        zoom_img = zoom_effect(frame, x, y, zoom_factor, zoom_size)

def zoom_effect(frame, x, y, zoom_factor=2, zoom_size=150):
    h, w = frame.shape[:2]

    # Ensure x and y are within the frame
    x = max(zoom_size // 2, min(x, w - zoom_size // 2))
    y = max(zoom_size // 2, min(y, h - zoom_size // 2))

    # Extract the region to zoom
    small_img = frame[y - zoom_size // 2:y + zoom_size // 2,
                      x - zoom_size // 2:x + zoom_size // 2]

    # Zoom the extracted region
    zoomed = cv2.resize(small_img, None, fx=zoom_factor, fy=zoom_factor)

    # Crop the zoomed image to match zoom_size
    zoomed = zoomed[zoomed.shape[0] // 2 - zoom_size // 2:zoomed.shape[0] // 2 + zoom_size // 2,
                    zoomed.shape[1] // 2 - zoom_size // 2:zoomed.shape[1] // 2 + zoom_size // 2]

    # Create a copy of the frame to modify
    zoom_img = frame.copy()

    # Place the zoomed image back into the frame
    zoom_img[y - zoom_size // 2:y + zoom_size // 2,
             x - zoom_size // 2:x + zoom_size // 2] = zoomed

    return zoom_img

def update_avg_color():
    global avg_color_img, colors, avg_color, frame
    if colors and frame is not None:
        avg_color = np.mean(colors, axis=0).astype(int)
        avg_color_img = np.full((frame.shape[0], avg_color_block_size, 3), avg_color, dtype=np.uint8)
        cv2.putText(avg_color_img, f"RGB: {avg_color}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) if np.mean(avg_color) < 128 else (0, 0, 0), 1)
    else:
        avg_color_img = np.zeros((frame.shape[0] if frame is not None else 480, avg_color_block_size, 3), dtype=np.uint8)

def refresh_parameters():
    global area_min, area_max, color_tolerance, coord_threshold, use_x_coord, dilation_size

    area_min = area_min_var.get()
    area_max = area_max_var.get()
    color_tolerance = color_tolerance_var.get()
    coord_threshold = coord_threshold_var.get()
    use_x_coord = bool(use_x_coord_var.get())
    dilation_size = 2 * dilation_size_var.get() + 3

def reselect_colors_callback():
    global colors, is_picking_color
    colors = []
    is_picking_color = True
    print("请在视频窗口中选择颜色，选择完毕后按Enter键")

def open_camera_callback():
    global is_camera_open, video_source
    if not is_camera_open:
        video_source = camera_utils.open_camera()
        if video_source is not None:
            is_camera_open = True
            print("摄像头已打开")
        else:
            print("无法打开摄像头")

def create_gui():
    global area_min_var, area_max_var, color_tolerance_var, coord_threshold_var, use_x_coord_var, dilation_size_var

    root = tk.Tk()
    root.title("参数调整")

    tk.Label(root, text="Area Min").pack()
    area_min_var = tk.IntVar(value=area_min)
    tk.Scale(root, from_=0, to=500, orient=tk.HORIZONTAL, variable=area_min_var).pack()

    tk.Label(root, text="Area Max").pack()
    area_max_var = tk.IntVar(value=area_max)
    tk.Scale(root, from_=0, to=500, orient=tk.HORIZONTAL, variable=area_max_var).pack()

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
    tk.Button(root, text="重新拾取颜色", command=reselect_colors_callback).pack()
    tk.Button(root, text="打开摄像头", command=open_camera_callback).pack()

    root.mainloop()

def draw_histogram(history, frame):
    hist_height = 100
    hist_width = 300
    hist = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)

    if len(history) > hist_width:
        history = history[-hist_width:]

    max_val = max(history) if history else 1

    for i, val in enumerate(history):
        cv2.line(hist, (i, hist_height), (i, hist_height - int(hist_height * val / max_val)), (255, 0, 0), 1)

    x_offset = frame.shape[1] - hist_width - 10
    y_offset = 10
    frame[y_offset:y_offset + hist_height, x_offset:x_offset + hist_width] = hist

def main():
    global frame, zoom_img, avg_color_img, colors, avg_color, is_picking_color, is_camera_open, video_source

    gui_thread = threading.Thread(target=create_gui, daemon=True)
    gui_thread.start()

    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', mouse_callback)

    while True:
        if is_camera_open and video_source is not None:
            frame = camera_utils.get_frame(video_source)
            if frame is None:
                print("无法从摄像头获取帧")
                break
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "请打开摄像头", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if zoom_img is None:
            zoom_img = frame.copy()

        if is_picking_color:
            update_avg_color()
            display_img = np.hstack((zoom_img, avg_color_img))
            cv2.imshow('Video', display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                is_picking_color = False
                if not colors:
                    print("未选择颜色，请重新选择")
                    continue
                avg_color = np.mean(colors, axis=0).astype(int)
                print(f"平均颜色: {avg_color}")
        else:
            if avg_color is not None:
                lower = np.array([max(0, c - color_tolerance) for c in avg_color])
                upper = np.array([min(255, c + color_tolerance) for c in avg_color])
                mask = cv2.inRange(frame, lower, upper)

                filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

                kernel = np.ones((dilation_size, dilation_size), np.uint8)
                dilated = cv2.dilate(mask, kernel, iterations=1)

                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                new_mask = np.zeros_like(mask)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area_min <= area <= area_max:
                        cv2.drawContours(new_mask, [contour], 0, 255, -1)

                filtered_frame = cv2.bitwise_and(frame, frame, mask=new_mask)

                color_regions_count = len([c for c in contours if area_min <= cv2.contourArea(c) <= area_max])
                color_regions_history.append(color_regions_count)

                centers = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area_min <= area <= area_max:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            centers.append((cX, cY))

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

                for group in groups:
                    if len(group) == 2:
                        cv2.line(frame, group[0], group[1], (0, 255, 0), 2)
                    elif len(group) > 2:
                        for point in group:
                            cv2.circle(frame, point, 30, (0, 255, 0), 2)

                cv2.putText(frame, f"Color Regions: {color_regions_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                draw_histogram(color_regions_history, frame)

                cv2.imshow('Video', frame)
                cv2.imshow('Filtered Video', filtered_frame)
            else:
                cv2.imshow('Video', zoom_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if video_source is not None:
        camera_utils.close_camera(video_source)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()