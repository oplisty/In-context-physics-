# export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP"
# export DATASET_NAME="/ssd1/jinxiu/PhysVideoGen/CogVideoX-Fun/datasets/internal_datasets/"
# export DATASET_META_NAME="/ssd1/jinxiu/PhysVideoGen/CogVideoX-Fun/datasets/internal_datasets/metadata.json"
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# NCCL_DEBUG=INFO

# python -m debugpy --listen 9501 --wait-for-client -m accelerate.commands.launch --mixed_precision="bf16" /ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/scripts/train_lora.py \
#   --pretrained_model_name_or_path=/ssd2/jinxiu/weights/cogvideo/models/Diffusion_Transformer/CogVideoX-Fun-V1.1-5b-InP \
#   --train_data_dir=/ssd1/jinxiu/PhysVideoGen/CogVideoX-Fun/datasets/internal_datasets/ \
#   --train_data_meta=/ssd1/jinxiu/PhysVideoGen/CogVideoX-Fun/datasets/internal_datasets/metadata.json \
#   --image_sample_size=1280 \
#   --video_sample_size=256 \
#   --token_sample_size=512 \
#   --video_sample_stride=3 \
#   --video_sample_n_frames=49 \
#   --train_batch_size=1 \
#   --video_repeat=1 \
#   --gradient_accumulation_steps=1 \
#   --dataloader_num_workers=8 \
#   --num_train_epochs=100 \
#   --checkpointing_steps=50 \
#   --learning_rate=1e-04 \
#   --seed=42 \
#   --output_dir="output_dir" \
#   --gradient_checkpointing \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --vae_mini_batch=1 \
#   --max_grad_norm=0.05 \
#   --random_hw_adapt \
#   --training_with_video_token_length \
#   --random_frame_crop \
#   --enable_bucket \
#   --low_vram \
#   --train_mode="inpaint"


# # 
# # Training command for CogVideoX-Fun-V1.5
# # export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP"
# # export DATASET_NAME="datasets/internal_datasets/"
# # export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# # export NCCL_IB_DISABLE=1
# # export NCCL_P2P_DISABLE=1
# # NCCL_DEBUG=INFO

# # accelerate launch --mixed_precision="bf16" scripts/train_lora.py \
# #   --pretrained_model_name_or_path=$MODEL_NAME \
# #   --train_data_dir=$DATASET_NAME \
# #   --train_data_meta=$DATASET_META_NAME \
# #   --image_sample_size=1024 \
# #   --video_sample_size=256 \
# #   --token_sample_size=512 \
# #   --video_sample_stride=3 \
# #   --video_sample_n_frames=85 \
# #   --train_batch_size=1 \
# #   --video_repeat=1 \
# #   --gradient_accumulation_steps=1 \
# #   --dataloader_num_workers=8 \
# #   --num_train_epochs=100 \
# #   --checkpointing_steps=50 \
# #   --learning_rate=1e-04 \
# #   --seed=42 \
# #   --output_dir="output_dir" \
# #   --gradient_checkpointing \
# #   --mixed_precision="bf16" \
# #   --adam_weight_decay=3e-2 \
# #   --adam_epsilon=1e-10 \
# #   --vae_mini_batch=1 \
# #   --max_grad_norm=0.05 \
# #   --random_hw_adapt \
# #   --training_with_video_token_length \
# #   --enable_bucket \
# #   --low_vram \
# #   --train_mode="inpaint" 

export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP"

# export DATASET_NAME="/ssd1/jinxiu/PhysVideoGen/CogVideoX-Fun/datasets/internal_datasets/"
# export DATASET_META_NAME="/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/datasets/internal_datasets/metadata.json"

export DATASET_NAME="datasets/datasettry"
export DATASET_META_NAME="datasets/datasettry.json"



export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO



# train in image condition mode

# 

# python -m debugpy --listen 9501 --wait-for-client -m accelerate.commands.launch --multi_gpu --num_processes 4 --main_process_port 29502 --mixed_precision="bf16" /ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/scripts/train_lora.py \
# python -m debugpy --listen 9501 --wait-for-client -m accelerate.commands.launch --main_process_port 29502 --mixed_precision="bf16" /ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/scripts/train_lora.py \
# python -m debugpy --listen 9501 --wait-for-client -m accelerate.commands.launch --main_process_port 29502 --mixed_precision="bf16" /ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/scripts/train_lora.py \

# debug

# python -m debugpy --listen 9501 --wait-for-client -m accelerate.commands.launch --multi_gpu --num_processes 4 --main_process_port 29502 --mixed_precision="bf16" /ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/scripts/train_lora.py \
#   --pretrained_model_name_or_path=/ssd2/jinxiu/weights/cogvideo/models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP \
#   --train_data_dir=/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/datasets/internal_datasets \
#   --train_data_meta=/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/datasets/internal_datasets/metadata.json \
#   --image_sample_size=1280 \
#   --video_sample_size=256 \
#   --token_sample_size=512 \
#   --video_sample_stride=3 \
#   --video_sample_n_frames=49 \
#   --train_batch_size=1 \
#   --video_repeat=1 \
#   --gradient_accumulation_steps=1 \
#   --dataloader_num_workers=8 \
#   --num_train_epochs=5000 \
#   --checkpointing_steps=50 \
#   --learning_rate=1e-04 \
#   --seed=42 \
#   --output_dir="output_dir" \
#   --gradient_checkpointing \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --vae_mini_batch=1 \
#   --max_grad_norm=0.05 \
#   --random_hw_adapt \
#   --training_with_video_token_length \
#   --random_frame_crop \
#   --enable_bucket \
#   --low_vram \
#   --train_mode="inpaint"

# inference

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# accelerate launch --multi_gpu --num_processes 2 --main_process_port 29502 --mixed_precision="bf16" scripts/train_lora.py \

# 修改 去除batch size


# train_batch_size = 10 ✅
# train_batch_size = 12

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard --main_process_port 29503 scripts/train_lora.py \
  --pretrained_model_name_or_path=/ssd2/jinxiu/weights/cogvideo/models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=10 \
  --video_repeat=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=1000 \
  --checkpointing_steps=100 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --low_vram \
  --train_mode="inpaint" 


# 
# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".

# 修改以下代码，使其能够连接vscode端口调试


# accelerate launch --mixed_precision="bf16" /ssd1/jinxiu/PhysVideoGen/CogVideoX-Fun/scripts/train_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATASET_NAME \
#   --train_data_meta=$DATASET_META_NAME \
#   --image_sample_size=1280 \
#   --video_sample_size=256 \
#   --token_sample_size=512 \
#   --video_sample_stride=3 \
#   --video_sample_n_frames=49 \
#   --train_batch_size=1 \
#   --video_repeat=1 \
#   --gradient_accumulation_steps=1 \
#   --dataloader_num_workers=8 \
#   --num_train_epochs=100 \
#   --checkpointing_steps=50 \
#   --learning_rate=1e-04 \
#   --seed=42 \
#   --output_dir="output_dir" \
#   --gradient_checkpointing \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --vae_mini_batch=1 \
#   --max_grad_norm=0.05 \
#   --random_hw_adapt \
#   --training_with_video_token_length \
#   --random_frame_crop \
#   --enable_bucket \
#   --low_vram \
#   --train_mode="inpaint" 

# 启动 debugpy 调试器并监听端口 9501，等待调试器客户端连接
