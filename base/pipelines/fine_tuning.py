from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import os
import json
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
import argparse
from omegaconf import OmegaConf
import sys
sys.path.append(os.path.split(sys.path[0])[0])
from models import get_models
from download import find_model
from transformers import CLIPTokenizer, CLIPTextModel


class VideoDatasetMsvd(Dataset):
    def __init__(self, annotations_file, video_dir, transform=None):
        self.video_dir = video_dir
        self.transform = transform
        
        # Legge il file annotations.txt e memorizza le descrizioni in un dizionario
        self.video_descriptions = {}
        with open(annotations_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                video_id = parts[0]
                description = ' '.join(parts[1:])
                if video_id not in self.video_descriptions:
                    self.video_descriptions[video_id] = []
                self.video_descriptions[video_id].append(description)
        
        # Ottieni la lista dei file video nella cartella YouTubeClips
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file)
        
        # Carica il video utilizzando OpenCV
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        video = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        
        # Estrarre un frame centrale
        mid_frame = frames[len(frames) // 2]
        mid_frame = torch.tensor(mid_frame, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # Ottieni le descrizioni del video
        video_id = os.path.splitext(video_file)[0]
        descriptions = self.video_descriptions.get(video_id, [])
        
        # Applica trasformazioni, se presenti
        if self.transform:
            video = self.transform(video)
            mid_frame = self.transform(mid_frame)
        
        return video, descriptions, mid_frame




class VideoDatasetMsrvtt(Dataset):
    def __init__(self, data, video_folder):
        self.videos = [video for video in data['videos'] if os.path.exists(os.path.join(video_folder, f"{video['video_id']}.mp4"))]
        self.sentences = {sentence['video_id']: sentence['caption'] for sentence in data['sentences']}
        self.video_folder = video_folder

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_info = self.videos[idx]
        video_id = video_info['video_id']
        video_path = os.path.join(self.video_folder, f"{video_id}.mp4")

        # Estrarre un frame dal video
        frame = self.extract_frame(video_path, video_info['start time'])
        if frame is None:
            raise ValueError(f"Frame extraction failed for video {video_id}")

        # Ottenere la descrizione
        description = self.sentences.get(video_id, None)
        if description is None:
            raise ValueError(f"No description found for video {video_id}")

        # Convertire il frame in un tensore PyTorch
        frame_tensor = torch.tensor(frame).permute(2, 0, 1)  # Convert to CxHxW

        return video_path, description, frame_tensor
    


    def extract_frame(self, video_path, time_sec):
      
      cap = cv2.VideoCapture(video_path)
      if not cap.isOpened():
          print(f"Cannot open video file: {video_path}")
          cap.release()
          return None

      duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
      
      # Se il timestamp è maggiore della durata, usa il momento più vicino possibile alla fine del video
      if time_sec > duration:
          time_sec = duration - 0.1  # Ritira di poco per evitare di superare la fine effettiva
          print(f"Adjusted timestamp to {time_sec} sec due to out of bounds in {video_path}")

      cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
      ret, frame = cap.read()
      cap.release()
      if not ret:
          print(f"Failed to extract frame at {time_sec} sec from {video_path}")
          return None
      return frame


def train_lora_model(data, video_folder, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carica il modello UNet e applica LoRA
    sd_path = args.pretrained_path + "/stable-diffusion-v1-4"
    unet = get_models(args, sd_path).to(device, dtype=torch.float16)
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)

    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)

    # Load CLIP model and processor for image conditioning
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
    )
    

    unet.config = {"model_type": "UNet"}

    unet = get_peft_model(unet, lora_config)
    
    #dataset = VideoDatasetMsrvtt(data, video_folder)
    dataset = VideoDatasetMsvd(data, video_folder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)
    num_epochs = 3
    
    unet.train()
    
    for epoch in range(num_epochs):
        for video_path, description, frame_tensor in dataloader:
            optimizer.zero_grad()
            
            text_inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True).input_ids.to(unet.device)
            text_features = text_encoder(text_inputs)[0]

            image_inputs = clip_processor(images=frame_tensor, return_tensors="pt").pixel_values.to(unet.device)
            image_features = clip_model.get_image_features(image_inputs)

            encoder_hidden_states = torch.cat([text_features, image_features.unsqueeze(1).repeat(1, text_features.size(1), 1)], dim=-1)

            # Forward pass
            output = unet(
                sample=torch.randn(4, 4, 64, 64, 64).to(unet.device),
                timestep=torch.randint(0, 1000, (4,)).to(unet.device),
                encoder_hidden_states=encoder_hidden_states
            )
            loss = torch.nn.functional.mse_loss(output.sample, torch.randn_like(output.sample))

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs} completed with loss: {loss.item()}")
    
            
    unet.save_pretrained("/content/drive/My Drive/finetuned_lora_unet")
    
    return unet
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    # Determina se sei su Google Colab
    on_colab = 'COLAB_GPU' in os.environ

    if on_colab:
        # Percorso del dataset su Google Colab
        dataset_path = '/content/drive/My Drive/msvd'
    else:
        # Percorso del dataset locale (sincronizzato con Google Drive)
        dataset_path = '/path/to/your/Google_Drive/sync/folder/path/to/your/dataset'
    
    # Percorsi dei file
    video_folder = os.path.join(dataset_path, 'YouTubeClips')
    data = os.path.join(dataset_path, 'annotations.txt')
    
    train_lora_model(data, video_folder, OmegaConf.load(args.config))

    '''
    Questa parte commentata serve se devo usare il dataset msrvtt

    # Determina se sei su Google Colab
    on_colab = 'COLAB_GPU' in os.environ

    if on_colab:
        # Percorso del dataset su Google Colab
        dataset_path = '/content/drive/My Drive/msrvtt'
    else:
        # Percorso del dataset locale (sincronizzato con Google Drive)
        dataset_path = '/path/to/your/Google_Drive/sync/folder/path/to/your/dataset'
    
    # Percorsi dei file
    video_folder = os.path.join(dataset_path, 'TrainValVideo')
    annotation_folder = os.path.join(dataset_path, 'train_val_annotation')

    # File di annotazione
    json_file = os.path.join(annotation_folder, 'train_val_videodatainfo.json')
    category_file = os.path.join(annotation_folder, 'category.txt')

    # Caricare il file JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    
    train_lora_model(data, video_folder, OmegaConf.load(args.config))

    '''