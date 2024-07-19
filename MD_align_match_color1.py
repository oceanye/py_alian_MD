import cv2
import numpy as np

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
    global avg_color_img, colors
    if colors:
        avg_color = np.mean(colors, axis=0).astype(int)
        avg_color_img = np.full((frame.shape[0], avg_color_block_size, 3), avg_color, dtype=np.uint8)
        cv2.putText(avg_color_img, f"RGB: {avg_color}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) if np.mean(avg_color) < 128 else (0, 0, 0), 1)
    else:
        avg_color_img = np.zeros((frame.shape[0], avg_color_block_size, 3), dtype=np.uint8)

def on_area_threshold_change(value):
    global area_threshold
    area_threshold = value


def on_color_tolerance_change(value):
    global color_tolerance
    color_tolerance = value


def on_coord_threshold_change(value):
    global coord_threshold
    coord_threshold = value


def on_coord_toggle_change(value):
    global use_x_coord
    use_x_coord = bool(value)


def on_dilation_size_change(value):
    global dilation_size
    dilation_size = 2 * value + 3  # 将值映射到 3, 5, 7, 9


# 步骤1: 打开视频并选择颜色
video = cv2.VideoCapture('video(3).mp4')
ret, frame = video.read()
if not ret:
    print("无法读取视频")
    exit()

zoom_img = frame.copy()
# 在主循环之前，初始化avg_color_img
avg_color_img = np.zeros((frame.shape[0], avg_color_block_size, 3), dtype=np.uint8)

cv2.namedWindow('Pick Color')
cv2.setMouseCallback('Pick Color', zoom_effect)

print("请点击选择多个颜色点，按Enter键结束选择")
# 在主循环中修改显示代码
while True:
    display_img = np.hstack((zoom_img, avg_color_img))
    cv2.imshow('Pick Color', display_img)
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter key
        break

cv2.destroyAllWindows()

if not colors:
    print("未选择颜色，程序退出")
    exit()

# 计算平均颜色
color = np.mean(colors, axis=0).astype(int)
print(f"平均颜色: {color}")

# 获取视频的原始尺寸
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建窗口并设置为原始尺寸
cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original Video', width, height)
cv2.namedWindow('Filtered Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Filtered Video', width, height)

# 创建滑块和复选框
cv2.createTrackbar('Area Threshold', 'Original Video', 200, 300, on_area_threshold_change)
cv2.createTrackbar('Color Tolerance', 'Original Video', 20, 30, on_color_tolerance_change)
cv2.createTrackbar('Coord Threshold', 'Original Video', 50, 100, on_coord_threshold_change)
cv2.createTrackbar('Dilation Size', 'Original Video', 1, 3, on_dilation_size_change)
cv2.createTrackbar('Use X Coord', 'Original Video', 0, 1, on_coord_toggle_change)

# 步骤2: 处理每一帧
while True:
    ret, frame = video.read()
    if not ret:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频到开始
        continue

    # 创建颜色掩码
    lower = np.array([max(0, c - color_tolerance) for c in color])
    upper = np.array([min(255, c + color_tolerance) for c in color])
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

    # 步骤3: 对中心点进行分组
    centers.sort(key=lambda x: x[0] if use_x_coord else x[1])  # 按X或Y坐标排序
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

    # 步骤4: 在当前帧上绘制
    for group in groups:
        if len(group) == 2:
            # 如果组内有2个点，就连直线
            cv2.line(frame, group[0], group[1], (0, 255, 0), 2)
        elif len(group) > 2:
            # 如果大于两个点，就以各个点为圆心，画出半径30的圆
            for point in group:
                cv2.circle(frame, point, 30, (0, 255, 0), 2)

    # 在原图顶部添加文字，显示识别到的颜色区域个数
    cv2.putText(frame, f"Color Regions: {color_regions_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Original Video', frame)
    cv2.imshow('Filtered Video', filtered_frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()