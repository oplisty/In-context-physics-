import os
import cv2
import numpy as np
import imageio

# 输入 & 输出路径
base_path = "/ssd1/jinxiu/ss2dataset/square_classified"
output_path = "/ssd1/jinxiu/ss2dataset/pytorch-i3d/concat_videos"

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# 获取所有类别文件夹
categories = [cat for cat in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, cat))]

# 遍历每个类别文件夹并排序
top_10_folders_per_category = {}

for category in categories:
    category_path = os.path.join(base_path, category)
    
    # 获取子文件夹
    subfolders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
    
    # 按照最后四位数字降序排序
    sorted_folders = sorted(subfolders, key=lambda x: float(x.split('_')[-1]), reverse=True)
    
    # 取前10个
    top_10_folders_per_category[category] = sorted_folders[:10]

# 处理拼接视频
for category, folders in top_10_folders_per_category.items():
    for i in range(len(folders) - 1):  # 按顺序两两拼接
        folder1 = folders[i]
        folder2 = folders[i + 1]

        folder1_path = os.path.join(base_path, category, folder1)
        folder2_path = os.path.join(base_path, category, folder2)

        # 获取 .mp4 文件
        mp4_files_1 = [f for f in os.listdir(folder1_path) if f.endswith(".mp4")]
        mp4_files_2 = [f for f in os.listdir(folder2_path) if f.endswith(".mp4")]

        if len(mp4_files_1) < 1 or len(mp4_files_2) < 1:
            continue

        # 选择第一个 mp4 进行拼接
        video1_path = os.path.join(folder1_path, mp4_files_1[0])
        video2_path = os.path.join(folder2_path, mp4_files_2[0])

        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)

        if not cap1.isOpened() or not cap2.isOpened():
            continue

        # 获取帧率 & 取最小帧率
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        fps = min(fps1, fps2) if fps1 > 0 and fps2 > 0 else 30

        # 获取帧数 & 取最小帧数
        frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        min_frames = min(frame_count1, frame_count2)

        # 获取视频分辨率 & 取最小高度
        width1, height1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width2, height2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        min_size = min(height1, height2)

        # 输出文件路径
        output_filename = f"{category}_{folder1}_{folder2}.mp4"
        output_filepath = os.path.join(output_path, output_filename)

        # 初始化 imageio 视频写入器
        writer = imageio.get_writer(output_filepath, fps=fps, codec="libx264")

        for _ in range(min_frames):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            # 调整帧大小
            frame1 = cv2.resize(frame1, (min_size, min_size))
            frame2 = cv2.resize(frame2, (min_size, min_size))

            # 颜色通道 BGR -> RGB
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            # 水平拼接
            concat_frame = np.hstack((frame1, frame2))
            writer.append_data(concat_frame)

        # 释放资源
        cap1.release()
        cap2.release()
        writer.close()

        print(f"拼接完成: {output_filename}")
