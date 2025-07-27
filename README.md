# Readme

## 环境要求

- Linux 系统（推荐 Ubuntu 20.04+）
- NVIDIA GPU（显存建议 ≥24GB）
- Conda 包管理器
- CUDA 11.7+（需与驱动版本匹配）

## 安装步骤

### 1. 下载模型权重

```
mkdir -p models/Diffusion_Transformer
git clone https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP
```

### 2. 创建虚拟环境

```
conda create -n incontext python=3.9 -y
conda activate incontext
```

### 3. 安装依赖库

```
pip install -r requirement.txt
pip install transformers==4.31.0
```

> **注**：若未提供 requirement.txt，请创建并包含以下内容：
>
> ```
> Pillow
> einops
> safetensors
> timm
> tomesd
> torch>=2.1.2
> torchdiffeq
> torchsde
> xformers
> decord
> datasets
> numpy
> scikit-image
> opencv-python
> omegaconf
> SentencePiece
> albumentations
> imageio[ffmpeg]
> imageio[pyav]
> tensorboard
> beautifulsoup4
> ftfy
> func_timeout
> deepspeed
> accelerate>=0.25.0
> gradio>=3.41.2,<=3.48.0
> diffusers>=0.30.1
> transformers>=4.37.2
> ```

### 4. 配置缓存路径（防止磁盘占满）

```
# 创建新缓存目录（替换 /path/to 为实际路径）
mkdir /path/to/.triton_cache

# 设置临时环境变量（当前会话有效）
export TRITON_CACHE_DIR=/path/to/.triton_cache
```

### 5. 配置 CUDA 路径

```
# 查找 CUDA 路径
which nvcc  # 输出类似 /usr/local/cuda-11.8/bin/nvcc

# 设置环境变量（替换为实际路径）
export CUDA_HOME=/usr/local/cuda-11.8

# 验证配置
echo $CUDA_HOME
```

### 6. 修改模型加载代码

在模型加载处添加以下配置：

```
extra_tokens = [f"<extra_id_{i}>" for i in range(100)]

tokenizer = T5Tokenizer.from_pretrained(
    'models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP',
    subfolder="tokenizer",
    additional_special_tokens=extra_tokens,  # 添加特殊token
    legacy=True  # 兼容旧版T5
)
```

### 7. 启动训练

```
nohup bash scripts/train_lora.sh > train.log 2>&1 &
```

## 监控训练进度

```
tail -f train.log
```

## 常见问题

1. **磁盘空间不足**：
   - 确保设置 `TRITON_CACHE_DIR` 到有足够空间的分区
   - 定期清理缓存：`rm -rf /path/to/.triton_cache/*`
2. **CUDA 路径错误**：
   - 运行 `nvcc --version` 验证安装
   - 确保 `CUDA_HOME` 指向包含 `bin/nvcc` 的目录
3. **依赖冲突**：
   - 使用新创建的虚拟环境 `incontext`
   - 严格安装指定版本：`pip install transformers==4.31.0`
4. **OOM 错误**：
   - 尝试减少 batch size
   - 使用 `deepspeed` 配置优化内存分配

> 建议部署前预留至少 50GB 磁盘空间，训练过程中监控 GPU 显存使用情况。