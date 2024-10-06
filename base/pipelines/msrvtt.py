import os
import json
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import torch

class MSRVTTDataset(Dataset):
    def __init__(self, video_dir, annotation_file, split='validate', transform=None):
        self.video_dir = video_dir
        self.transform = transform
        self.split = split

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        split_values = set(video['split'] for video in data['videos'])
        print(f"Valori unici del campo 'split': {split_values}")

        self.videos = [video for video in data['videos'] if video['split'] == self.split]
        print(f"Numero di video nello split '{self.split}': {len(self.videos)}")

        self.captions = {}
        split_video_ids = set([video['video_id'] for video in self.videos])

        for sentence in data['sentences']:
            video_id = sentence['video_id']
            if video_id in split_video_ids:
                if video_id not in self.captions:
                    self.captions[video_id] = []
                self.captions[video_id].append(sentence['caption'])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_info = self.videos[idx]
        video_id = video_info['video_id']
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Il video {video_path} non esiste.")

        frames = self._load_video_frames(video_path)

        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
        else:
            frames = torch.stack([transforms.ToTensor()(frame) for frame in frames])

        captions = self.captions.get(video_id, [])
        caption = random.choice(captions) if captions else ""

        return frames, caption, video_id

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame = cap.read()
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
            success, frame = cap.read()
        cap.release()
        return frames
    
def collate_fn(batch):
    videos, captions, video_ids = zip(*batch)
    
    min_length = min(video.shape[0] for video in videos)
    truncated_videos = [video[:min_length] for video in videos]
    videos_tensor = torch.stack(truncated_videos)

    return videos_tensor, captions, video_ids

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MSRVTTDataset(
        video_dir='/content/drive/My Drive/msrvtt/TrainValVideo',
        annotation_file='/content/drive/My Drive/msrvtt/train_val_annotation/train_val_videodatainfo.json',
        split='validate',
        transform=transform
    )

    print(f"Lunghezza del dataset: {len(dataset)}")

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for i, (videos, captions, video_ids) in enumerate(data_loader):
        print(f"Batch {i}:")
        print(f"Video shape: {videos.shape}")
        print(f"Numero di caption: {len(captions)}")
        print(f"Numero di video ID: {len(video_ids)}")
        
        for j in range(len(video_ids)):
            print(f"  Elemento {j}:")
            print(f"    Video ID: {video_ids[j]}")
            print(f"    Caption: {captions[j]}")
        
        break  # Per esempio, fermiamoci dopo il primo batch