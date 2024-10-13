import os
import json
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import torch
import clip
from tqdm import tqdm
import fine_tuning
from torch.utils.data import Subset


class MSRVTTDataset(Dataset):
    def __init__(self, video_dir, annotation_file, split='validate', transform=None):
        """
        Args:
            video_dir (string): Directory con i video .mp4.
            annotation_file (string): Path al file train_val_videodatainfo.json.
            split (string): 'train' o 'validate'.
            transform (callable, optional): Trasformazioni da applicare ai frame.
        """
        self.video_dir = video_dir
        self.transform = transform
        self.split = split

        # Carica le annotazioni
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # Stampiamo i valori unici del campo 'split'
        split_values = set(video['split'] for video in data['videos'])
        print(f"Valori unici del campo 'split': {split_values}")

        # Filtra i video per lo split specificato
        self.videos = [video for video in data['videos'] if video['split'] == self.split]
        print(f"Numero di video nello split '{self.split}': {len(self.videos)}")

        # Crea un mapping da video_id a caption
        self.captions = {}
        # Creiamo un set di video_id per lo split specificato per efficienza
        split_video_ids = set([video['video_id'] for video in self.videos])

        for sentence in data['sentences']:
            video_id = sentence['video_id']
            if video_id in split_video_ids:
                if video_id not in self.captions:
                    self.captions[video_id] = []
                self.captions[video_id].append(sentence['caption'])

        # Definisci una trasformazione per il frame specifico
        self.frame_transform = transforms.Compose([
            transforms.ToTensor(),          # Converte in tensore
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Cambia da [C, H, W] a [H, W, C]
            transforms.Lambda(lambda x: x * 255),  # Scala a [0, 255]
            transforms.Lambda(lambda x: x.byte())  # Converte in torch.uint8
        ])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # Ottieni le informazioni del video
        video_info = self.videos[idx]
        video_id = video_info['video_id']
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")

        # Verifica se il file video esiste
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Il video {video_path} non esiste.")

        # Leggi il video e ottieni i frame
        frames = self._load_video_frames(video_path)

        if self.transform:
            transformed_frames = torch.stack([self.transform(frame) for frame in frames])
        else:
            transformed_frames = torch.stack([transforms.ToTensor()(frame) for frame in frames])

        # Seleziona un frame specifico (es. il primo frame)
        selected_frame = frames[1]  # Puoi cambiare l'indice per selezionare un altro frame

        # Applica le trasformazioni definite per il frame
        frame_tensor = self.frame_transform(selected_frame)

        # Ottieni le didascalie (captions) associate al video
        captions = self.captions.get(video_id, [])
        if captions:
            # Seleziona una didascalia a caso
            caption = random.choice(captions)
        else:
            caption = ""

        sample = {
            'video': transformed_frames,     # Tensor dei frame del video
            'caption': caption,              # Didascalia
            'video_id': video_id,            # ID del video
            'frame': frame_tensor            # Frame specifico con shape [1, 320, 512, 3] e dtype torch.uint8
        }
        return sample
    

    def _load_video_frames(self, video_path):
        """
        Carica i frame dal video specificato.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame = cap.read()
        while success:
            # Converti il frame da BGR a RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
            success, frame = cap.read()
        cap.release()
        return frames
    

def collate_fn(batch):
    # Estrai i componenti del batch
    videos = [sample['video'] for sample in batch]
    captions = [sample['caption'] for sample in batch]
    video_ids = [sample['video_id'] for sample in batch]
    frames = [sample['frame'] for sample in batch]

    # Stack dei video
    videos_tensor = torch.stack(videos)  # [batch_size, num_frames, C, H, W]

    # Stack dei frame specifici
    frames_tensor = torch.stack(frames)  # [batch_size, 1, 320, 512, 3]

    return {
        'video': videos_tensor,
        'caption': captions,
        'video_id': video_ids,
        'frame': frames_tensor
    }

