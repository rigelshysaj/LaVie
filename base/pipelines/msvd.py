import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class VideoDatasetMsvd(Dataset):
    def __init__(self, annotations_file, video_dir, tokenizer, transform=None, target_size=(320, 512), fixed_frame_count=16, is_train=True):
        self.video_dir = video_dir
        self.transform = transform
        self.target_size = target_size
        self.fixed_frame_count = fixed_frame_count
        self.tokenizer = tokenizer
        self.is_train = is_train
        
        # Read annotations file and store descriptions in a dictionary
        self.video_descriptions = {}
        with open(annotations_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                video_id = parts[0]
                description = ' '.join(parts[1:])
                if video_id not in self.video_descriptions:
                    self.video_descriptions[video_id] = [description]
                else:
                    self.video_descriptions[video_id].append(description)
        
        # Get the list of video files in the YouTubeClips folder
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]

    def __len__(self):
        return len(self.video_files)

    def tokenize_caption(self, caption):
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        a = inputs.input_ids.squeeze(0)
        b = inputs.input_ids

        print(f"input_ids.squeeze(0) shape: {a.shape}, dtype: {a.dtype}")
        print(f"input_ids shape: {b.shape}, dtype: {b.dtype}")

        return b

    def preprocess_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.target_size)
            frames.append(frame)
        cap.release()
        
        if len(frames) < self.fixed_frame_count:
            frames += [frames[-1]] * (self.fixed_frame_count - len(frames))
        else:
            frames = frames[:self.fixed_frame_count]
        
        frames_np = np.array(frames)
        frames_np = frames_np.astype(np.float32) / 255.0
        frames_np = (frames_np - 0.5) / 0.5
        
        video = torch.tensor(frames_np).permute(3, 0, 1, 2)
        
        # Extract middle frame
        mid_frame = frames[len(frames) // 2]
        mid_frame_np = np.array(mid_frame)
        mid_frame_np = mid_frame_np.astype(np.float32) / 255.0
        mid_frame_np = (mid_frame_np - 0.5) / 0.5
        mid_frame = torch.tensor(mid_frame_np).permute(2, 0, 1)
        
        if self.transform:
            video = self.transform(video)
            mid_frame = self.transform(mid_frame)
        
        return video, mid_frame

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file)

        try:
            video, mid_frame = self.preprocess_video(video_path)
            
            video_id = os.path.splitext(video_file)[0]
            descriptions = self.video_descriptions.get(video_id, ["No description available"])
            
            # Choose a random description if in training mode, otherwise take the first one
            description = random.choice(descriptions) if self.is_train else descriptions[0]
            
            input_ids = self.tokenize_caption(description)
            
            return {"pixel_values": video, "input_ids": input_ids, "mid_frame": mid_frame}
        
        except Exception as e:
            print(f"Skipping video {video_file} due to error: {e}")
            return None