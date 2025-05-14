import os
import re
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from pytorch_i3d import InceptionI3d

def video_to_tensor(video):
    """Convert a numpy.ndarray (T x H x W x C) to a torch.FloatTensor (C x T x H x W)."""
    return torch.from_numpy(video.transpose([3, 0, 1, 2]))

def load_video(video_path):
    """Load an MP4 video and return it as a normalized torch tensor."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))  # Resize to I3D input size
        frame = (frame / 255.0) * 2 - 1  # Normalize to [-1, 1]
        frames.append(frame)
    cap.release()
    return np.asarray(frames, dtype=np.float32)

def extract_features(video_path, model):
    """Extract features from a video using I3D."""
    video = load_video(video_path)
    video_tensor = video_to_tensor(video).unsqueeze(0).cuda()  # Add batch dimension
    
    with torch.no_grad():
        features = model.extract_features(video_tensor)
        features = features.squeeze(0).permute(1, 2, 3, 0).reshape(-1, features.shape[1])
        features = features.mean(dim=0)  # Global feature pooling
    return features.cpu()

def compute_similarity(features1, features2):
    """Compute cosine similarity between two feature vectors."""
    return F.cosine_similarity(features1.unsqueeze(0), features2.unsqueeze(0)).item()

def run(video1, video2, model_path):
    """Process two videos and compute similarity."""
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(model_path))
    i3d.cuda()
    i3d.eval()
    
    features1 = extract_features(video1, i3d)
    features2 = extract_features(video2, i3d)
    similarity = compute_similarity(features1, features2)
    print(f"Similarity: {similarity:.4f}")
    return similarity

# 根目录路径
directory = "/ssd1/jinxiu/ss2dataset/square_classified"
model_path = "/ssd1/jinxiu/ss2dataset/pytorch-i3d/models/rgb_charades.pt"

# 正则表达式匹配已计算相似度的目录，例如 pair_308_0_6663
processed_pattern = re.compile(r"pair_\d+_\d+_\d+")

# 遍历 square_classified 下的所有 pair_x 目录
for root, _, files in os.walk(directory):
    folder_name = os.path.basename(root)

    # 如果目录名匹配 `pair_x_0_6663` 这种格式，说明已计算过
    if processed_pattern.match(folder_name):
        print(f"⏭️ Skipping {root}, similarity already computed.")
        continue

    mp4_files = [os.path.join(root, f) for f in files if f.endswith(".mp4")]

    # 确保每个 pair_x 目录内只有 2 个 .mp4 文件
    if len(mp4_files) == 2:
        video1, video2 = mp4_files
        similarity = run(video1, video2, model_path)
        
        # 格式化相似度值，替换 `.` 为 `_`
        similarity_str = f"{similarity:.4f}".replace(".", "_")
        
        # 生成新文件夹名称
        new_folder_name = f"{folder_name}_{similarity_str}"
        new_folder_path = os.path.join(os.path.dirname(root), new_folder_name)
        
        # 重命名文件夹
        os.rename(root, new_folder_path)
        print(f"✅ Renamed {root} → {new_folder_path}")


