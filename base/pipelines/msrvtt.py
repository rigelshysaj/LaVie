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

        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
        else:
            frames = torch.stack([transforms.ToTensor()(frame) for frame in frames])

        # Ottieni le didascalie (captions) associate al video
        captions = self.captions.get(video_id, [])
        if captions:
            # Seleziona una didascalia a caso
            caption = random.choice(captions)
        else:
            caption = ""

        sample = {'video': frames, 'caption': caption, 'video_id': video_id}
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

    # Tronca tutti i video alla lunghezza minima
    truncated_videos = [video[:100] for video in videos]

    # Stack dei video
    videos_tensor = torch.stack(truncated_videos)

    return {'video': videos_tensor, 'caption': captions, 'video_id': video_ids}

def get_clip_similarity(clip_model, preprocess, text, image, device):
    with torch.no_grad():
        text_features = clip_model.encode_text(clip.tokenize(text).to(device))
        image_features = clip_model.encode_image(preprocess(image).unsqueeze(0).to(device))
        
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).item()
    return similarity

def evaluate_msrvtt_zero_shot(lavie_fine_tuned, clip_model, preprocess, dataset, device):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    total_similarity = 0
    num_videos = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        video = batch['video'].squeeze(0).to(device)  # (num_frames, C, H, W)
        caption = batch['caption'][0]  # Prendiamo solo una didascalia
        
        # Genera il video con lavie_fine_tuned
        with torch.no_grad():
            generated_video = lavie_fine_tuned.generate(caption)  # Assumiamo che questo sia il modo corretto di chiamare il tuo modello
        
        # Calcola la similarità CLIP per ogni frame
        frame_similarities = []
        for frame in generated_video:
            similarity = get_clip_similarity(clip_model, preprocess, caption, frame)
            frame_similarities.append(similarity)
        
        # Calcola la media delle similarità per questo video
        avg_similarity = sum(frame_similarities) / len(frame_similarities)
        total_similarity += avg_similarity
        num_videos += 1
    
    # Calcola la similarità media su tutti i video
    average_similarity = total_similarity / num_videos
    return average_similarity



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Carica il modello CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Carica il tuo modello pre-addestrato
    lavie_fine_tuned = torch.load("path/to/your/lavie_fine_tuned_model.pth")
    lavie_fine_tuned.to(device)
    lavie_fine_tuned.eval()
    
    # Prepara il dataset
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
    
    # Esegui la valutazione
    average_similarity = evaluate_msrvtt_zero_shot(lavie_fine_tuned, clip_model, preprocess, dataset, device)
    
    print(f"Average CLIP Similarity (CLIPSIM): {average_similarity:.4f}")

   

    #print(f"Lunghezza del dataset: {len(dataset)}")

    # Create the DataLoader
    #data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

