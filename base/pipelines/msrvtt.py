import os
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2

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

        # Applica le trasformazioni se specificate
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Ottieni le didascalie (captions) associate al video
        captions = self.captions.get(video_id, [])

        sample = {'video': frames, 'captions': captions, 'video_id': video_id}
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


if __name__ == "__main__":

    # Definisci le trasformazioni (se necessario)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Crea il dataset
    dataset = MSRVTTDataset(
        video_dir='/content/drive/My Drive/msrvtt/TrainValVideo',
        annotation_file='/content/drive/My Drive/msrvtt/train_val_annotation/train_val_videodatainfo.json',
        split='validate',
        transform=transform
    )

    print(f"Lunghezza del dataset: {len(dataset)}")

    # Se il dataset non è vuoto, prova ad accedere al primo elemento
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Video ID: {sample['video_id']}")
        print(f"Numero di frame: {len(sample['video'])}")
        print(f"Captions: {sample['captions']}")
    else:
        print("Il dataset è vuoto.")

