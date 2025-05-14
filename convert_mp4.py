# import imageio

# def convert_webm_to_mp4(input_path, output_path, fps=30):
#     """
#     将 .webm 文件转换为 .mp4 格式。

#     参数:
#     - input_path: str, 输入 .webm 文件路径
#     - output_path: str, 输出 .mp4 文件路径
#     - fps: int, 目标帧率 (默认 30)
#     """
#     reader = imageio.get_reader(input_path, format="ffmpeg")
#     writer = imageio.get_writer(output_path, format="ffmpeg", fps=fps)

#     for frame in reader:
#         writer.append_data(frame)

#     reader.close()
#     writer.close()
#     print(f"✅ Converted {input_path} → {output_path}")

# # 设置路径
# webm_path = "/ssd1/jinxiu/ss2dataset/classified_videos/Covering_[something]_with_[something]/10.webm"
# mp4_path = webm_path.replace(".webm", ".mp4")

# # 转换为 mp4
# convert_webm_to_mp4(webm_path, mp4_path)

import os
import imageio

def convert_webm_to_mp4(input_path, output_path, fps=30):
    """
    将 .webm 文件转换为 .mp4 格式。

    参数:
    - input_path: str, 输入 .webm 文件路径
    - output_path: str, 输出 .mp4 文件路径
    - fps: int, 目标帧率 (默认 30)
    """
    reader = imageio.get_reader(input_path, format="ffmpeg")
    writer = imageio.get_writer(output_path, format="ffmpeg", fps=fps)

    for frame in reader:
        writer.append_data(frame)

    reader.close()
    writer.close()
    print(f"✅ Converted {input_path} → {output_path}")

# 设置根目录
root_dir = "/ssd1/jinxiu/ss2dataset/classified_videos_mp4/classified_videos"

# 遍历所有子文件夹，查找 .webm 文件并转换
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".webm"):
            webm_path = os.path.join(subdir, file)
            mp4_path = webm_path.replace(".webm", ".mp4")

            # 转换文件
            convert_webm_to_mp4(webm_path, mp4_path)
