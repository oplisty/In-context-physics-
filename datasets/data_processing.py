# # 写一个python 代码，


# from PIL import Image
# import imageio
# import numpy as np
# import os



# input_video_path = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/PhysicsVideo/1.0_1.0_A white flag flaps in fierce winds.mp4"
# reference_video_path = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/PhysicsVideo/1.0_1.0_A white T-shirt flutters in fierce winds.mp4"

# reference_video_path = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/PhysicsVideo/1.0_1.0_A white flag flaps in fierce winds.mp4"
# input_video_path = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/PhysicsVideo/1.0_1.0_A white T-shirt flutters in fierce winds.mp4"

# output_video_path = "/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/datasets/internal_datasets/train/000002.mp4"
# reference_image_path = "/ssd1/jinxiu/PhysVideoGen/reference_image.png"

# sample_size = [384, 672]
# video_length        = 85
# fps                 = 16


# # Function to extract frames from video
# def extract_frames(video_path, target_size, frame_count):
#     reader = imageio.get_reader(video_path, format='ffmpeg')
#     frames = []
#     total_frames = reader.get_length()
#     interval = max(1, total_frames // frame_count)

#     for i, frame in enumerate(reader):
#         if i % interval == 0:
#             img = Image.fromarray(frame)
#             img_resized = img.resize((target_size[1], target_size[0]))
#             frames.append(np.array(img_resized))

#         if len(frames) == frame_count:
#             break

#     reader.close()
#     return frames

# # Function to create a side-by-side video
# def create_combined_video(input_frames, reference_frames, output_path, fps):
#     height, width, _ = input_frames[0].shape
#     combined_width = width * 2
#     writer = imageio.get_writer(output_path, fps=fps, codec='libx264')

#     for input_frame, reference_frame in zip(input_frames, reference_frames):
#         combined_frame = np.hstack((input_frame, reference_frame))
#         writer.append_data(combined_frame)

#     writer.close()

# # Main process
# input_frames = extract_frames(input_video_path, sample_size, video_length)
# reference_frames = extract_frames(reference_video_path, sample_size, video_length)

# # Save the first frame of the reference video as the reference image
# Image.fromarray(reference_frames[0]).save(reference_image_path)

# # Create combined video
# create_combined_video(input_frames, reference_frames, output_video_path, fps)

# print(f"Combined video saved at: {output_video_path}")
# print(f"Reference image saved at: {reference_image_path}")



# def modify_video(video_path, sample_size):
#     # Read the video file
#     reader = imageio.get_reader(video_path)
#     fps = reader.get_meta_data()['fps']
    
#     # Get the first frame and convert it to PIL image
#     first_frame = reader.get_data(0)
#     first_frame_pil = Image.fromarray(first_frame)
#     width, height = first_frame_pil.size
    
#     left_width = width // 2
#     right_width = width - left_width
    
#     # Store the modified frames
#     modified_frames = []
    
#     # Process the first frame, keeping the right part unchanged
#     # Instead of modifying the right half, just crop and paste as before
#     first_right_frame = first_frame_pil.crop((left_width, 0, width, height))
#     first_frame_pil.paste(first_right_frame, (left_width, 0))
    
#     # Resize the first frame to match the sample size
#     first_frame_resized = first_frame_pil.resize((sample_size[1], sample_size[0]))  # (width, height)
#     modified_frames.append(np.array(first_frame_resized))
    
#     # Process the subsequent frames
#     for i, frame in enumerate(reader):
#         if i == 0:  # Skip the first frame as it's already processed
#             continue
#         frame_pil = Image.fromarray(frame)
        
#         # Crop the left part of the current frame
#         left_frame = frame_pil.crop((0, 0, left_width, height))
        
#         # Instead of creating a black right half, keep the right half unchanged
#         right_frame = frame_pil.crop((left_width, 0, width, height))
        
#         # Combine the left part and the unchanged right part
#         modified_frame = Image.new('RGB', (width, height))
#         modified_frame.paste(left_frame, (0, 0))
#         modified_frame.paste(right_frame, (left_width, 0))
        
#         # Resize the modified frame to match the sample size
#         modified_frame_resized = modified_frame.resize((sample_size[1], sample_size[0]))  # (width, height)
#         modified_frames.append(np.array(modified_frame_resized))
    
#     # Write the modified frames to a new video file
#     output_video_path = video_path
#     writer = imageio.get_writer(output_video_path, fps=fps)
    
#     for frame in modified_frames:
#         writer.append_data(frame)
    
#     writer.close()

#     print(f"Modified video saved at: {output_video_path}")

# # Example usage
# # validation_video = "/ssd1/jinxiu/PhysVideoGen/CogVideoX-Fun/basketball1_concatenated_47frames_fps8.mp4"
# validation_video = output_video_path
# sample_size = [384, 672]  # Example size to resize to
# modify_video(validation_video, sample_size)

import json
import os


def modify_json_file():
    input_file_path = '/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/datasets/internal_datasets/metadata.json'
    output_file_path = '/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/datasets/internal_datasets/modified_metadata.json'
    video_dir = '/ssd1/jinxiu/PhysVideoGen/Cogvideo1.5/CogVideoX-Fun/datasets/internal_datasets/train'

    try:
        with open(input_file_path, 'r') as f:
            data = json.load(f)

        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

        if len(data) < len(video_files):
            for i in range(len(data), len(video_files)):
                new_obj = {
                    "file_path": os.path.join(video_dir, video_files[i]),
                    "text": "The two-panel video features a physics phenomenon transferring, with [LEFT] showing a reference physics phenomenon and [RIGHT] demonstrating the same physics in: A white Flag flutters in fierce winds in the sky",
                    "type": "video"
                }
                data.append(new_obj)
        elif len(data) > len(video_files):
            data = data[:len(video_files)]

        for i, item in enumerate(data):
            item['file_path'] = os.path.join(video_dir, video_files[i])

        with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully modified and saved to {output_file_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}. The input file or video directory was not found.")
    except json.JSONDecodeError as e:
        print(f"Error: {e}. There was an issue decoding the JSON file.")


if __name__ == "__main__":
    modify_json_file()