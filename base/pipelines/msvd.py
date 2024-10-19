import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class VideoDatasetMsvd(Dataset):
    def __init__(self, annotations_file, video_dir, transform=None, target_size=(512, 320), fixed_frame_count=16, augmentations_per_video=4):
        self.video_dir = video_dir
        self.transform = transform
        self.target_size = target_size
        self.fixed_frame_count = fixed_frame_count
        self.augmentations_per_video = augmentations_per_video  # Numero di augmentazioni per video
        
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
        
        # Definisci le trasformazioni di augmentazione
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(size=self.target_size, scale=(0.8, 1.0)),
            # Aggiungi altre trasformazioni se necessario
        ])
        
        # Se non hai già una trasformazione, usa solo le augmentazioni
        if self.transform is None:
            self.transform = self.augmentation_transforms

    def __len__(self):
        return len(self.video_files) * self.augmentations_per_video

    def __getitem__(self, idx):
        # Determina quale video e quale augmentazione utilizzare
        video_idx = idx // self.augmentations_per_video
        augmentation_idx = idx % self.augmentations_per_video
        
        video_file = self.video_files[video_idx]
        video_path = os.path.join(self.video_dir, video_file)

        try:
            # Carica il video utilizzando OpenCV
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converti da BGR a RGB
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
            cap.release()

            if len(frames) < self.fixed_frame_count:
                frames += [frames[-1]] * (self.fixed_frame_count - len(frames))  # Ripeti l'ultimo frame
            else:
                # Prendi i primi fixed_frame_count frame
                frames = frames[:self.fixed_frame_count]
            
            frames_np = np.array(frames)
            frames_np = frames_np.astype(np.float32) / 255.0  # Normalizza in [0, 1]
            frames_np = (frames_np - 0.5) / 0.5

            video = torch.tensor(frames_np)  # Forma: (num_frames, H, W, C)
            video = video.permute(0, 3, 1, 2)  # Cambia la forma a (num_frames, C, H, W)
            
            # Estrarre un frame centrale
            mid_frame = frames[len(frames) // 2]
            mid_frame_np = np.array(mid_frame).astype(np.float32) / 255.0
            mid_frame_np = (mid_frame_np - 0.5) / 0.5
            mid_frame = torch.tensor(mid_frame_np).permute(2, 0, 1)  # Forma: (C, H, W)
            
            # Ottieni le descrizioni del video
            video_id = os.path.splitext(video_file)[0]
            descriptions = self.video_descriptions.get(video_id, [])
            
            # Applica trasformazioni di augmentazione specifiche
            if self.transform:
                # Applica le trasformazioni frame-wise
                augmented_frames = []
                for frame in video:
                    pil_frame = transforms.ToPILImage()(frame)
                    pil_frame = self.augmentation_transforms(pil_frame)
                    frame = transforms.ToTensor()(pil_frame)
                    augmented_frames.append(frame)
                video = torch.stack(augmented_frames)
                
                # Applica le trasformazioni al frame centrale
                pil_mid_frame = transforms.ToPILImage()(mid_frame)
                pil_mid_frame = self.augmentation_transforms(pil_mid_frame)
                mid_frame = transforms.ToTensor()(pil_mid_frame)
                mid_frame = (mid_frame - 0.5) / 0.5  # Rinominalizzazione dopo ToTensor

            return video, descriptions, mid_frame
        
        except Exception as e:
            print(f"Skipping video {video_file} due to error: {e}")
            return None, None, None
