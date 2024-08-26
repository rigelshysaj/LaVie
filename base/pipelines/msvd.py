from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import os
import numpy as np

class VideoDatasetMsvd(Dataset):
    def __init__(self, annotations_file, video_dir, transform=None, target_size=(320, 512), fixed_frame_count=16):
        self.video_dir = video_dir
        self.transform = transform
        self.target_size = target_size
        self.fixed_frame_count = fixed_frame_count
        
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

        print(f"video_files: {self.video_files}")
        print(f"video_descriptions: {self.video_descriptions}")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file)

        try:

            # Carica il video utilizzando OpenCV
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
            cap.release()

            print(f"len frames: {len(frames)}")

            
            # Se il numero di frame è inferiore a fixed_frame_count, ripeti l'ultimo frame
            if len(frames) < self.fixed_frame_count:
                frames += [frames[-1]] * (self.fixed_frame_count - len(frames))  # Ripeti l'ultimo frame
            else:
                # Prendi i primi fixed_frame_count frame
                frames = frames[:self.fixed_frame_count]
            
            frames_np = np.array(frames)
            frames_np = frames_np.astype(np.float32) / 255.0  # Normalizza in [0, 1]
            frames_np = (frames_np - 0.5) / 0.5

            video = torch.tensor(frames_np).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

            print(f"video shape: {video.shape}, dtype: {video.dtype}")
            
            # Estrarre un frame centrale
            mid_frame = frames[len(frames) // 2]
            mid_frame_np = np.array(mid_frame)

            mid_frame_np = mid_frame_np.astype(np.float32) / 255.0  # Normalizza in [0, 1]
            #mid_frame_np = (mid_frame_np - 0.5) / 0.5

            mid_frame = torch.tensor(mid_frame_np).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

            print(f"mid_frame shape: {mid_frame.shape}, dtype: {mid_frame.dtype}")
            
            # Ottieni le descrizioni del video
            video_id = os.path.splitext(video_file)[0]
            descriptions = self.video_descriptions.get(video_id, [])

            #print(f"description of __getitem__: {descriptions} video_id: {video_id}")

            # Save the cropped video
            output_video_path = os.path.join("/content/drive/My Drive/", f"{video_id}_cropped.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 30.0, self.target_size)
            for frame in frames:
                out.write(frame)
            out.release()
            print(f"Saved cropped video to {output_video_path}")

            # Save the mid-frame as an image
            output_image_path = os.path.join("/content/drive/My Drive/", f"{video_id}_mid_frame.png")
            cv2.imwrite(output_image_path, cv2.cvtColor(mid_frame_np, cv2.COLOR_RGB2BGR))
            print(f"Saved mid-frame image to {output_image_path}")
            
            # Applica trasformazioni, se presenti
            if self.transform:
                video = self.transform(video)
                mid_frame = self.transform(mid_frame)
            
            return video, descriptions, mid_frame
        
        except Exception as e:
            print(f"Skipping video {video_file} due to error: {e}")
            return None, None, None