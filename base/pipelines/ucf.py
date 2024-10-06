import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from torchvision.io import read_video
import numpy as np
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ConvertImageDtype, Normalize


class UCF101Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, num_frames=16):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.num_frames = num_frames

        # Construct the full path to the CSV file
        csv_file_path = os.path.join(self.root_dir, self.csv_file)

        # Load annotations from the CSV file
        self.annotations = pd.read_csv(csv_file_path, sep=',')

        print("Columns in self.annotations:", self.annotations.columns)

        # Create a mapping from class labels to indices
        self.classes = sorted(self.annotations['label'].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_info = self.annotations.iloc[idx]
        clip_path = video_info['clip_path'].lstrip('/')
        video_path = os.path.join(self.root_dir, clip_path)
        label = self.class_to_idx[video_info['label']]

        # Carica i frame del video
        video_frames, _, _ = read_video(video_path, pts_unit='sec')

        # Assicurati che il video abbia il numero desiderato di frame
        frames = self.process_frames(video_frames)

        # frames shape: [C, T, H, W]
        # Permute to [T, C, H, W] per applicare le trasformazioni
        frames = frames.permute(1, 0, 2, 3)

        if self.transform:
            # Applica la trasformazione a ciascun frame
            frames = torch.stack([self.transform(frame) for frame in frames])
        else:
            # Normalizza i frame se nessuna trasformazione è fornita
            frames = frames.float() / 255.0

        # Permute di nuovo a [C, T, H, W]
        frames = frames.permute(1, 0, 2, 3)

        sample = {'frames': frames, 'label': label}
        return sample


    def process_frames(self, video_frames):
        num_video_frames = video_frames.shape[0]

        if num_video_frames > self.num_frames:
            indices = np.linspace(0, num_video_frames - 1, num=self.num_frames, dtype=int)
            frames = video_frames[indices]
        elif num_video_frames < self.num_frames:
            last_frame = video_frames[-1].unsqueeze(0)
            repeat_times = self.num_frames - num_video_frames
            padding = last_frame.repeat(repeat_times, 1, 1, 1)
            frames = torch.cat((video_frames, padding), dim=0)
        else:
            frames = video_frames

        frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]
        return frames


if __name__ == "__main__":
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize frames to 224x224
        ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean and std
                            std=[0.229, 0.224, 0.225])
    ])

    # Create the dataset
    train_dataset = UCF101Dataset(
        csv_file='train.csv',
        root_dir='/content/drive/My Drive/UCF101',  # Replace with your actual path
        transform=transform,
        num_frames=16
    )

    # Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    # Example of iterating through the DataLoader
    for batch in train_loader:
        frames = batch['frames']  # Tensor of shape [batch_size, C, T, H, W]
        labels = batch['label']   # Tensor of shape [batch_size]
        print(f"Frames shape: {frames.shape}")
        print(f"Labels: {labels}")
        break  # Remove this break to iterate over the entire dataset

    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        # Estrai i componenti del batch
        frames = batch['frames']          # Tensor di forma [batch_size, C, T, H, W]
        labels = batch['label']           # Tensor di forma [batch_size]
        print(f"Frames shape: {frames.shape}")
        print(f"Labels: {labels}")

        # Determina la dimensione del batch
        batch_size = labels.size(0)       # Alternativamente: len(labels)

        # Itera su ogni elemento nel batch
        for idx in range(batch_size):
            print(f"\n  Elemento {idx + 1} in Batch {batch_idx}:")
            print(f"    Label: {labels[idx].item()}")
            print(f"    Frames shape: {frames[idx].shape}")  # [C, T, H, W]
        break
