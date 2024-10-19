from torch.utils.data import Dataset
import torch
import cv2
import os
import numpy as np
import random
from torchvision import transforms

class VideoDatasetMsvd(Dataset):
    def __init__(self, annotations_file, video_dir, transform=None, target_size=(512, 320), fixed_frame_count=16, augmentation_factor=5):
        self.video_dir = video_dir
        self.transform = transform
        self.target_size = target_size
        self.fixed_frame_count = fixed_frame_count
        self.augmentation_factor = augmentation_factor

        # Legge il file annotations.txt e memorizza le descrizioni in un dizionario
        self.video_descriptions = {}
        with open(annotations_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                video_id = parts[0]
                description = ' '.join(parts[1:])
                if video_id not in self.video_descriptions:
                    self.video_descriptions[video_id] = description

        # Ottieni la lista dei file video nella cartella YouTubeClips
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]

    def __len__(self):
        return len(self.video_files) * self.augmentation_factor

    def apply_augmentation(self, frames):
        augmented_frames = []
        for frame in frames:
            # Applica augmentation casuale
            if random.random() < 0.5:
                frame = cv2.flip(frame, 1)  # Flip orizzontale
            if random.random() < 0.5:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # Rotazione di 90 gradi
            if random.random() < 0.5:
                brightness = random.uniform(0.8, 1.2)
                frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
            augmented_frames.append(frame)
        return augmented_frames

    def __getitem__(self, idx):
        video_idx = idx // self.augmentation_factor
        aug_idx = idx % self.augmentation_factor

        video_file = self.video_files[video_idx]
        video_path = os.path.join(self.video_dir, video_file)

        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
            cap.release()

            if len(frames) == 0:
                raise ValueError(f"No frames were read from the video {video_file}")

            # Ensure we have exactly fixed_frame_count frames
            if len(frames) < self.fixed_frame_count:
                frames = frames + [frames[-1]] * (self.fixed_frame_count - len(frames))
            elif len(frames) > self.fixed_frame_count:
                frames = frames[:self.fixed_frame_count]

            # Apply data augmentation
            if aug_idx > 0:
                frames = self.apply_augmentation(frames)

            # Convert to numpy array and ensure all frames have the same shape
            frames_np = np.array([cv2.resize(frame, self.target_size) for frame in frames])
            
            frames_np = frames_np.astype(np.float32) / 255.0
            frames_np = (frames_np - 0.5) / 0.5
            video = torch.tensor(frames_np)

            # Extract a central frame
            mid_frame = frames[len(frames) // 2]
            mid_frame_np = np.array(mid_frame)
            mid_frame = torch.tensor(mid_frame_np)

            video_id = os.path.splitext(video_file)[0]
            descriptions = self.video_descriptions.get(video_id, "")

            if self.transform:
                video = self.transform(video)
                mid_frame = self.transform(mid_frame)

            return video, descriptions, mid_frame
        except Exception as e:
            print(f"Skipping video {video_file} due to error: {e}")
            return None, None, None