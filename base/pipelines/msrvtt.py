import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.io as io

class MSRVTTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Percorso al file di annotazioni
        annotation_file = os.path.join(root_dir, 'train_val_annotation', 'train_val_videodatainfo.json')

        # Carica le annotazioni
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # Ottieni le frasi
        sentences = data['sentences']

        # Ottieni la lista dei video disponibili nella cartella TrainValVideo
        video_dir = os.path.join(root_dir, 'TrainValVideo')
        available_videos = set(os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith('.mp4'))

        # Filtra le frasi per includere solo quelle i cui video sono disponibili
        self.samples = [s for s in sentences if s['video_id'] in available_videos]

        print(f"Numero di campioni disponibili: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample['video_id']
        caption = sample['caption']

        # Percorso al video
        video_path = os.path.join(self.root_dir, 'TrainValVideo', f"{video_id}.mp4")

        # Carica il video usando torchvision
        video, _, info = io.read_video(video_path, pts_unit='sec')

        # Applica le trasformazioni se presenti
        if self.transform:
            video = self.transform(video)

        return video, caption

    



if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Definisci eventuali trasformazioni
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Crea il dataset utilizzando i video disponibili
    dataset = MSRVTTDataset(root_dir='/content/drive/My Drive/msrvtt', transform=transform)

    # Crea il DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Itera attraverso il DataLoader
    for videos, captions in dataloader:
        # Qui puoi inserire il codice per l'addestramento o la validazione del tuo modello
        print(f"videos: {videos}, captions: {captions}" )

