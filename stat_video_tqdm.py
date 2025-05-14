import os
import json
import ffmpeg
from tqdm import tqdm  # 导入tqdm库用于显示进度条

# 设置视频目录路径和输出JSON文件路径
video_dataset_path = '/ssd1/jinxiu/ss2dataset/20bn-something-something-v2'  # 修改为你的视频目录路径
info_json_output_path = '/ssd1/jinxiu/ss2dataset/ss-v2_info_json_1.json'  # 修改为你的输出路径


def get_video_info(video_path):
    """ 获取单个视频的持续时间和总帧数 """
    try:
        # 使用ffmpeg.probe获取视频元数据
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='stream=duration,r_frame_rate')
        
        # 获取持续时间
        format_info = probe.get('format', {})
        if 'duration' in format_info:
            duration = float(format_info['duration'])
        else:
            raise ValueError(f"视频 {video_path} 不包含持续时间信息")

        # 获取帧数
        frames = None
        if 'nb_frames' in probe['streams'][0]:
            frames = int(probe['streams'][0]['nb_frames'])
        
        # 如果没有找到帧数，则通过帧率估算帧数
        if frames is None:
            if 'r_frame_rate' in probe['streams'][0]:
                frame_rate = eval(probe['streams'][0]['r_frame_rate'])
                frames = int(duration * frame_rate)
            else:
                raise ValueError(f"视频 {video_path} 不包含帧率信息，无法计算帧数")

        return {
            'duration': duration,
            'frames': frames
        }
    except Exception as e:
        return None

def collect_video_info(video_dataset_path):
    """ 遍历目录下所有视频并收集信息 """
    video_info_dict = {}
    
    # 获取所有的webm视频文件，按序号排序
    video_files = sorted([file for file in os.listdir(video_dataset_path) if file.endswith('.webm')], key=lambda x: int(x.split('.')[0]))
    
    # 使用tqdm显示进度条，只显示当前视频名称和平均视频长度
    total_duration = 0
    processed_count = 0
    
    with tqdm(total=len(video_files), desc="处理视频", unit="视频", ncols=100) as pbar:
        for file in video_files:
            video_path = os.path.join(video_dataset_path, file)
            video_info = get_video_info(video_path)
            
            if video_info:
                video_info_dict[file] = video_info
                total_duration += video_info['duration']
                processed_count += 1
                
                # 计算并显示平均视频长度
                avg_duration = total_duration / processed_count if processed_count > 0 else 0
                pbar.set_postfix({"当前视频": file, "平均长度 (秒)": f"{avg_duration:.2f}"})
            
            pbar.update(1)
    
    return video_info_dict

def save_info_to_json(info_dict, output_path):
    """ 将视频信息保存到JSON文件 """
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(info_dict, json_file, indent=4, ensure_ascii=False)

# 主程序
video_info = collect_video_info(video_dataset_path)
save_info_to_json(video_info, info_json_output_path)
print(f"视频信息已保存到: {info_json_output_path}")