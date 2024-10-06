import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.io as io

class MSRVTTDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory con tutti i dati.
            split (string): 'train', 'val' o 'test'.
            transform (callable, optional): Trasformazioni opzionali da applicare ai video.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Percorso al file di annotazioni
        annotation_file = os.path.join(root_dir, 'train_val_annotation', 'train_val_videodatainfo.json')

        # Carica le annotazioni
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # Ottieni i video e le frasi
        videos = data['videos']
        sentences = data['sentences']

        # Filtra i video in base allo split
        self.videos = [video for video in videos if video['split'] == split]
        video_ids = set(video['video_id'] for video in self.videos)

        # Crea una mappa da video_id a informazioni video
        self.video_dict = {video['video_id']: video for video in self.videos}

        # Associa le didascalie ai video corrispondenti
        self.samples = []
        for sentence in sentences:
            if sentence['video_id'] in video_ids:
                self.samples.append({
                    'video_id': sentence['video_id'],
                    'caption': sentence['caption']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample['video_id']
        caption = sample['caption']

        # Percorso al video
        video_path = os.path.join(self.root_dir, 'TrainValVideo', f"{video_id}.mp4")

        # Carica il video usando torchvision
        video, _, _ = io.read_video(video_path, pts_unit='sec')

        # Applica le trasformazioni se presenti
        if self.transform:
            video = self.transform(video)

        return video, caption


if __name__ == "__main__":
    from torch.utils.data import DataLoader


    # Definisci eventuali trasformazioni
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ridimensiona i frame del video
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalizza i frame
                            std=[0.229, 0.224, 0.225])
    ])

    # Crea il dataset
    dataset = MSRVTTDataset(root_dir='/content/drive/My Drive/msrvtt', split='val', transform=transform)

    # Crea il DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for videos, captions in dataloader:
        print(f"video: {videos}, captions: {captions}")