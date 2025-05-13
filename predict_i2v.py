import json
import os

import numpy as np
import torch
from diffusers import (AutoencoderKL, CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from transformers import T5EncoderModel, T5Tokenizer
from omegaconf import OmegaConf
from PIL import Image
import imageio

from cogvideox.models.transformer3d import CogVideoXTransformer3DModel
from cogvideox.models.autoencoder_magvit import AutoencoderKLCogVideoX
from cogvideox.pipeline.pipeline_cogvideox import CogVideoX_Fun_Pipeline
from cogvideox.pipeline.pipeline_cogvideox_inpaint import CogVideoX_Fun_Pipeline_Inpaint
from cogvideox.utils.lora_utils import merge_lora, unmerge_lora
from cogvideox.utils.utils import get_image_to_video_latent, save_videos_grid, get_video_to_video_latent

# Low gpu memory mode, this is used when the GPU memory is under 16GB
low_gpu_memory_mode = False

# model path
model_name          = "/ssd2/jinxiu/weights/cogvideo/models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP"

# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" "DDIM_Cog" and "DDIM_Origin"
sampler_name        = "DDIM_Origin"

# Load pretrained model if need
transformer_path    = None 
vae_path            = None
lora_path           = None

lora_path = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/output_dir/checkpoint-200.safetensors"

# Other params
sample_size         = [384, 672]
# V1.0 and V1.1 support up to 49 frames of video generation,
# while V1.5 supports up to 85 frames.  
video_length        = 49
fps                 = 8

# If you want to generate ultra long videos, please set partial_video_length as the length of each sub video segment
partial_video_length = None
overlap_video_length = 4

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
# If you want to generate from text, please set the validation_image_start = None and validation_image_end = None



# 写一个python 代码，使得新生成的视频符合sample_size ，左半部分是input video，右半部分是reference video，都需要进行一定的resize，对于每个视频，都需要保证其帧率为8，帧数为47，reference_image 为reference_video 的第一帧

# 写一个python 代码，对于validation_video 的左半部分保持不变，右半部分除了第一帧和原视频保持一致外，其他都设置为纯黑色，也就是值为0

# input_video_path = "/ssd1/jinxiu/PhysVideoGen/CogVideoX-Fun/basketball1.mp4"
# reference_video_path = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/1.0_1.0_Four basketballs spinning randomly in the air in free fall.mp4"
input_video_path = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/PhysicsVideo/1.0_1.0_A white flag flaps in fierce winds.mp4"
reference_video_path = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/PhysicsVideo/1.0_1.0_A white T-shirt flutters in fierce winds.mp4"


import cv2

# 输入的拼接视频路径
concat_video = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/datasets/sth_datasets_train/Covering_[something]_with_[something]_pair_143_0_8138_pair_509_0_8083.mp4"

# 目标拆分后的视频路径
reference_video_new = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/PhysicsVideo/reference_video_split.mp4"
input_video_new = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/PhysicsVideo/input_video_split.mp4"

# 读取视频
reader = imageio.get_reader(concat_video, format="FFMPEG")
fps = reader.get_meta_data()['fps']
frames = [frame for frame in reader]
total_frames = len(frames)

# 目标帧数
target_frames = 47

def interpolate_frames(frame_list, num_frames):
    """ 逐帧插值，使视频达到 num_frames 帧 """
    original_len = len(frame_list)
    if original_len == num_frames:
        return frame_list  # 已是目标帧数
    
    new_frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1) * (original_len - 1)
        idx = int(np.floor(alpha))  # 当前帧索引
        next_idx = min(idx + 1, original_len - 1)  # 下一帧索引
        blend_ratio = alpha - idx  # 插值比例

        # 进行帧插值
        interpolated_frame = cv2.addWeighted(frame_list[idx], 1 - blend_ratio, frame_list[next_idx], blend_ratio, 0)
        new_frames.append(interpolated_frame.astype(np.uint8))

    return new_frames

# 获取左右半边的视频帧
left_frames = [frame[:, :frame.shape[1] // 2, :] for frame in frames]
right_frames = [frame[:, frame.shape[1] // 2:, :] for frame in frames]

# 进行插值，生成 47 帧
left_frames_resized = interpolate_frames(left_frames, target_frames)
right_frames_resized = interpolate_frames(right_frames, target_frames)

# 重新写入新视频
writer_ref = imageio.get_writer(reference_video_new, fps=fps, format="FFMPEG")
writer_input = imageio.get_writer(input_video_new, fps=fps, format="FFMPEG")

for frame in left_frames_resized:
    writer_ref.append_data(frame)

for frame in right_frames_resized:
    writer_input.append_data(frame)


# 关闭资源
# reader.close()
# writer_ref.close()
# writer_input.close()

print(f"左半视频已保存为: {reference_video_new}，帧数: {target_frames}")
print(f"右半视频已保存为: {input_video_new}，帧数: {target_frames}")

for frame in reader:
    height, width, _ = frame.shape
    mid = width // 2  # 计算中点

    left_frame = frame[:, :mid, :]   # 左半边
    right_frame = frame[:, mid:, :]  # 右半边

    writer_ref.append_data(left_frame)   # 保存左半部分
    writer_input.append_data(right_frame)  # 保存右半部分

# 关闭读写器
reader.close()
writer_ref.close()
writer_input.close()

print(f"左半视频已保存为: {reference_video_new}")
print(f"右半视频已保存为: {input_video_new}")


reference_video_path = reference_video_new
input_video_path = input_video_new


output_video_path = "/ssd1/jinxiu/PhysVideoGen/output_combined_video.mp4"
reference_image_path = "/ssd1/jinxiu/PhysVideoGen/reference_image.png"

# Function to extract frames from video
def extract_frames(video_path, target_size, frame_count):
    reader = imageio.get_reader(video_path, format='ffmpeg')
    frames = []
    total_frames = reader.get_length()
    interval = max(1, total_frames // frame_count)

    for i, frame in enumerate(reader):
        if i % interval == 0:
            img = Image.fromarray(frame)
            img_resized = img.resize((target_size[1], target_size[0]))
            frames.append(np.array(img_resized))

        if len(frames) == frame_count:
            break

    reader.close()
    return frames

# Function to create a side-by-side video
def create_combined_video(input_frames, reference_frames, output_path, fps):
    height, width, _ = input_frames[0].shape
    combined_width = width * 2
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264')

    for input_frame, reference_frame in zip(input_frames, reference_frames):
        combined_frame = np.hstack((input_frame, reference_frame))
        writer.append_data(combined_frame)

    writer.close()

# Main process
input_frames = extract_frames(input_video_path, sample_size, video_length)
reference_frames = extract_frames(reference_video_path, sample_size, video_length)

# Save the first frame of the reference video as the reference image
Image.fromarray(reference_frames[0]).save(reference_image_path)

# Create combined video
create_combined_video(input_frames, reference_frames, output_video_path, fps)

print(f"Combined video saved at: {output_video_path}")
print(f"Reference image saved at: {reference_image_path}")


from PIL import Image
import imageio
import numpy as np


def modify_video(video_path):
    # 读取视频文件
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    # 获取视频的第一帧
    first_frame = reader.get_data(0)
    # 将第一帧转换为 PIL 图像
    first_frame_pil = Image.fromarray(first_frame)
    width, height = first_frame_pil.size
    left_width = width // 2
    right_width = width - left_width
    # 存储修改后的帧
    modified_frames = []
    # 处理第一帧，保持右半部分不变
    first_right_frame = first_frame_pil.crop((left_width, 0, width, height))
    first_frame_pil.paste(first_right_frame, (left_width, 0))
    modified_frames.append(np.array(first_frame_pil))
    # 处理后续帧
    for i, frame in enumerate(reader):
        if i == 0:  # 跳过第一帧，因为已经处理过
            continue
        frame_pil = Image.fromarray(frame)
        # 创建一个全黑的右半部分
        black_right = Image.new('RGB', (right_width, height), (0, 0, 0))
        # 保留左半部分
        left_frame = frame_pil.crop((0, 0, left_width, height))
        # 拼接左半部分和黑色右半部分
        modified_frame = Image.new('RGB', (width, height))
        modified_frame.paste(left_frame, (0, 0))
        modified_frame.paste(black_right, (left_width, 0))
        modified_frames.append(np.array(modified_frame))
    # 将修改后的帧保存为新的视频
    writer = imageio.get_writer('modified_video.mp4', fps=fps)
    for frame in modified_frames:
        writer.append_data(frame)
    writer.close()


# validation_video = "/ssd1/jinxiu/PhysVideoGen/CogVideoX-Fun/basketball1_concatenated_47frames_fps8.mp4"
validation_video = output_video_path
modify_video(validation_video)


validation_video = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/modified_video.mp4"
validation_image_start = os.path.join(os.path.dirname(validation_video), "asset", "1.png")

# Ensure the "asset" directory exists, create it if it doesn't
os.makedirs(os.path.dirname(validation_image_start), exist_ok=True)

# Read the first frame of the video using imageio
reader = imageio.get_reader(validation_video)

# Get the first frame
first_frame = reader.get_data(0)  # Index 0 for the first frame

# Convert the frame to a PIL Image
pil_image = Image.fromarray(first_frame)

# Save the image
pil_image.save(validation_image_start)

print(f"First frame saved as {validation_image_start}")

validation_image_end    = None

# prompts
# prompt                  = "The dog is shaking head. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."

prompt_structure = "The two-panel video features a physics phenomenon transferring, with [LEFT] showing a reference physics phenomenon and [RIGHT] demonstrating the same physics in "

driving_prompt = " A white T-shirt flutters in fierce winds "
driving_prompt = " a person's hand moving back and forth over an orange surface, possibly cleaning or wiping it"
# driving_prompt = "A basketball is falling"

icl = True

prompt = prompt_structure + driving_prompt  

negative_prompt         = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "
guidance_scale          = 6.0
seed                    = 43
num_inference_steps     = 50
lora_weight             = 0.55
save_path               = "samples/cogvideox-fun-videos_i2v"

transformer = CogVideoXTransformer3DModel.from_pretrained_2d(
    model_name, 
    subfolder="transformer",
    low_cpu_mem_usage=True,
).to(weight_dtype)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
vae = AutoencoderKLCogVideoX.from_pretrained(
    model_name, 
    subfolder="vae"
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

text_encoder = T5EncoderModel.from_pretrained(
    model_name, subfolder="text_encoder", torch_dtype=weight_dtype
)
# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM_Cog": CogVideoXDDIMScheduler,
    "DDIM_Origin": DDIMScheduler,
}[sampler_name]
scheduler = Choosen_Scheduler.from_pretrained(
    model_name, 
    subfolder="scheduler"
)

if transformer.config.in_channels != vae.config.latent_channels:
    pipeline = CogVideoX_Fun_Pipeline_Inpaint.from_pretrained(
        model_name,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype
    )
else:
    pipeline = CogVideoX_Fun_Pipeline.from_pretrained(
        model_name,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype
    )
if low_gpu_memory_mode:
    pipeline.enable_sequential_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight)

if partial_video_length is not None:
    partial_video_length = int((partial_video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (partial_video_length - 1) // vae.config.temporal_compression_ratio + 1
    if partial_video_length != 1 and transformer.config.patch_size_t is not None and latent_frames % transformer.config.patch_size_t != 0:
        additional_frames = transformer.config.patch_size_t - latent_frames % transformer.config.patch_size_t
        partial_video_length += additional_frames * vae.config.temporal_compression_ratio
        
    init_frames = 0
    last_frames = init_frames + partial_video_length
    while init_frames < video_length:
        if last_frames >= video_length:
            _partial_video_length = video_length - init_frames
            _partial_video_length = int((_partial_video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
            latent_frames = (_partial_video_length - 1) // vae.config.temporal_compression_ratio + 1
            if _partial_video_length != 1 and transformer.config.patch_size_t is not None and latent_frames % transformer.config.patch_size_t != 0:
                additional_frames = transformer.config.patch_size_t - latent_frames % transformer.config.patch_size_t
                _partial_video_length += additional_frames * vae.config.temporal_compression_ratio

            if _partial_video_length <= 0:
                break
        else:
            _partial_video_length = partial_video_length

        input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image, None, video_length=_partial_video_length, sample_size=sample_size)
        
        with torch.no_grad():
            sample = pipeline(
                prompt, 
                num_frames = _partial_video_length,
                negative_prompt = negative_prompt,
                height      = sample_size[0],
                width       = sample_size[1],
                generator   = generator,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps,

                video        = input_video,
                mask_video   = input_video_mask
            ).videos
        
        if init_frames != 0:
            mix_ratio = torch.from_numpy(
                np.array([float(_index) / float(overlap_video_length) for _index in range(overlap_video_length)], np.float32)
            ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            
            new_sample[:, :, -overlap_video_length:] = new_sample[:, :, -overlap_video_length:] * (1 - mix_ratio) + \
                sample[:, :, :overlap_video_length] * mix_ratio
            new_sample = torch.cat([new_sample, sample[:, :, overlap_video_length:]], dim = 2)

            sample = new_sample
        else:
            new_sample = sample

        if last_frames >= video_length:
            break

        validation_image = [
            Image.fromarray(
                (sample[0, :, _index].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
            ) for _index in range(-overlap_video_length, 0)
        ]

        init_frames = init_frames + _partial_video_length - overlap_video_length
        last_frames = init_frames + _partial_video_length
else:
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
    if video_length != 1 and transformer.config.patch_size_t is not None and latent_frames % transformer.config.patch_size_t != 0:
        additional_frames = transformer.config.patch_size_t - latent_frames % transformer.config.patch_size_t
        video_length += additional_frames * vae.config.temporal_compression_ratio


    
    # validation_video = "/ssd1/jinxiu/PhysVideoGen/CogVideoX-Fun/basketball1_concatenated_47frames_fps8.mp4"
    validation_video_mask = None
    input_video_1, input_video_mask, clip_image = get_video_to_video_latent(validation_video, video_length=video_length, sample_size=sample_size, validation_video_mask=validation_video_mask, fps=fps)


    input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, validation_image_end, video_length=video_length, sample_size=sample_size)

    with torch.no_grad():
        sample = pipeline(
            prompt, 
            num_frames = video_length,
            negative_prompt = negative_prompt,
            height      = sample_size[0],
            width       = sample_size[1],
            generator   = generator,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,
            video        = input_video_1[:,:,:47],
            mask_video   = input_video_mask[:,:,:47],
            icl = icl,
        ).videos

        # /ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/samples/cogvideox-fun-videos_i2v/00000002.mp4
        # icl = True, video = input_video_1

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight)

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

index = len([path for path in os.listdir(save_path)]) + 1
prefix = str(index).zfill(8)

if video_length == 1:
    video_path = os.path.join(save_path, prefix + ".png")

    image = sample[0, :, 0]
    image = image.transpose(0, 1).transpose(1, 2)
    image = (image * 255).numpy().astype(np.uint8)
    image = Image.fromarray(image)
    image.save(video_path)
else:
    video_path = os.path.join(save_path, prefix + ".mp4")
    save_videos_grid(sample, video_path, fps=fps)
