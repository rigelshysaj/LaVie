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
    truncated_videos = [video[:5] for video in videos]

    # Stack dei video
    videos_tensor = torch.stack(truncated_videos)

    return {'video': videos_tensor, 'caption': captions, 'video_id': video_ids}


def get_clip_similarity(clip_model, preprocess, text, image, device):
    with torch.no_grad():
        # Preprocess the image
        image_input = preprocess(image).unsqueeze(0).to(device)
        # Tokenize the text
        text_input = clip.tokenize([text]).to(device)
        
        # Compute features
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        
        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (image_features @ text_features.T).item()

    return similarity


def evaluate_msrvtt_clip_similarity(clip_model, preprocess, dataset, device):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    total_gt_similarity = 0  # For ground truth videos
    total_gen_similarity = 0  # For generated videos
    num_videos = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        # Ground Truth Video Frames and Caption
        gt_video = batch['video'].squeeze(0).to(device)  # (num_frames, C, H, W)
        caption = batch['caption'][0]  # Single caption
        video_id = batch['video_id'][0]
        
        # Generate Video from Caption using Your Model
        with torch.no_grad():
            generated_video_frames = fine_tuning.model(caption)  # Assuming this returns frames in tensor format
        
        # Ensure frames are in the correct format (e.g., list of PIL Images)
        gt_frames = [transforms.ToPILImage()(frame.cpu()) for frame in gt_video]
        gen_frames = [frame for frame in generated_video_frames]

        
        # Compute CLIP Similarity for Ground Truth Video
        gt_frame_similarities = []
        for frame in gt_frames:
            similarity = get_clip_similarity(clip_model, preprocess, caption, frame, device)
            gt_frame_similarities.append(similarity)
        avg_gt_similarity = sum(gt_frame_similarities) / len(gt_frame_similarities)
        total_gt_similarity += avg_gt_similarity
        
        # Compute CLIP Similarity for Generated Video
        gen_frame_similarities = []
        for frame in gen_frames:
            similarity = get_clip_similarity(clip_model, preprocess, caption, frame, device)
            gen_frame_similarities.append(similarity)
        avg_gen_similarity = sum(gen_frame_similarities) / len(gen_frame_similarities)
        total_gen_similarity += avg_gen_similarity
        
        num_videos += 1
    
    # Compute Average CLIPSIM Scores
    average_gt_similarity = total_gt_similarity / num_videos
    average_gen_similarity = total_gen_similarity / num_videos
    
    return average_gt_similarity, average_gen_similarity


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Carica il modello CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Prepara le trasformazioni
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Regola secondo necessità
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Inizializza il dataset
    dataset = MSRVTTDataset(
        video_dir='/content/drive/My Drive/msrvtt/TrainValVideo',
        annotation_file='/content/drive/My Drive/msrvtt/train_val_annotation/train_val_videodatainfo.json',
        split='validate',
        transform=transform
    )
    
    # Imposta un seme per la riproducibilità (opzionale)
    random.seed(42)
    
    # Verifica che il dataset abbia almeno 10 campioni
    if len(dataset) < 5:
        raise ValueError("Il dataset contiene meno di 10 campioni.")
    
    # Seleziona casualmente 10 indici
    subset_indices = random.sample(range(len(dataset)), 5)
    
    # Crea il sottoinsieme del dataset
    subset_dataset = Subset(dataset, subset_indices)
    
    # Esegui la valutazione sul sottoinsieme
    average_gt_similarity, average_gen_similarity = evaluate_msrvtt_clip_similarity(
        clip_model, preprocess, subset_dataset, device
    )
    
    print(f"Average Ground Truth CLIP Similarity (CLIPSIM): {average_gt_similarity:.4f}")
    print(f"Average Generated Video CLIP Similarity (CLIPSIM): {average_gen_similarity:.4f}")

