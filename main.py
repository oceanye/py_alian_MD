import cv2

def extract_frames(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧率
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 读取视频的每一帧
    success, frame = cap.read()
    count = 0

    # 逐帧保存为JPEG图像
    while success:
        # 构建输出文件路径
        output_path = f"{output_folder}/frame_{count:04d}.jpg"

        # 保存帧为JPEG图像
        cv2.imwrite(output_path, frame)

        # 读取下一帧
        success, frame = cap.read()

        count += 1

    # 释放视频文件句柄
    cap.release()

if __name__ == "__main__":
    # 输入AVI文件路径和输出文件夹路径
    avi_file_path = "Video_20240311134648883.avi"
    output_folder_path = "subfolder"

    # 提取帧并保存为JPEG图像
    extract_frames(avi_file_path, output_folder_path)
