import os
import json
import torch
import cv2
from transformers import AutoModelForCausalLM, AutoProcessor

# 路径设置
video_dir = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/datasets/sth_datasets"
save_path = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/datasets/sth_metadata.json"
dataset_output_dir = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/datasets/sth_datasets_train"

# 加载 VideoLLaMA3 模型
model_name = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/VideoLLaMA3/VideoLLaMA3-2B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# 检查视频是否有效
def check_video_valid(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count > 0  # 确保视频有帧

# 生成视频 caption 的函数
def generate_caption(video_path):
    try:
        if not check_video_valid(video_path):
            print(f"Skipping invalid video: {video_path}")
            return None

        question = "只描述这个视频的右半部分，忽略左半部分, 不超过50词，直接描述右边的部分，不要出现right 之类的字眼，the right side is xxx, 直接补充xxx中的内容"
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": 5, "max_frames": 128}},
                    {"type": "text", "text": question},
                ]
            },
        ]

        inputs = processor(conversation=conversation, return_tensors="pt")

        if "pixel_values" not in inputs or inputs["pixel_values"] is None:
            print(f"Warning: No pixel values for {video_path}")
            return None

        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        output_ids = model.generate(**inputs, max_new_tokens=128)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

# 处理所有 MP4 文件
metadata = []
for filename in sorted(os.listdir(video_dir)):
    if filename.endswith(".mp4"):
        video_path = os.path.join(video_dir, filename)
        caption = generate_caption(video_path)

        # 如果 caption 生成失败，跳过这个视频
        if caption is None:
            print(f"Skipping {filename} due to processing error.")
            continue
        
        # 目标路径
        target_file_path = os.path.join(dataset_output_dir, filename)
        print("caption:", caption)
        metadata.append({
            "file_path": target_file_path,
            "text": f"The two-panel video features a physics phenomenon transferring, with [LEFT] showing a reference physics phenomenon and [RIGHT] demonstrating the same physics in: {caption}",
            "type": "video"
        })

# 保存 JSON 文件
with open(save_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Metadata saved to {save_path}")
