from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Dict, List, Optional, Union
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
import torch.nn as nn
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
import einops
from diffusers.models import AutoencoderKL
from torch.utils.checkpoint import checkpoint
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    #randn_tensor,
    replace_example_docstring,
    BaseOutput,
)
logger = logging.get_logger(__name__)


class VideoDatasetMsvd(Dataset):
    def __init__(self, annotations_file, video_dir, transform=None, target_size=(512, 320), fixed_frame_count=16):
        self.video_dir = video_dir
        self.transform = transform
        self.target_size = target_size
        self.fixed_frame_count = fixed_frame_count
        
        # Legge il file annotations.txt e memorizza le descrizioni in un dizionario
        self.video_descriptions = {}
        with open(annotations_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                video_id = parts[0]
                description = ' '.join(parts[1:])
                if video_id not in self.video_descriptions:
                    #self.video_descriptions[video_id] = []
                    self.video_descriptions[video_id] = description
                #self.video_descriptions[video_id].append(description)
        
        # Ottieni la lista dei file video nella cartella YouTubeClips
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]

        print(f"dictionary of descriptions : {self.video_descriptions}")

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
            frame = cv2.resize(frame, self.target_size)
            frames.append(frame)
        cap.release()

        
        # Se il numero di frame è inferiore a fixed_frame_count, ripeti l'ultimo frame
        if len(frames) < self.fixed_frame_count:
            frames += [frames[-1]] * (self.fixed_frame_count - len(frames))  # Ripeti l'ultimo frame
        else:
            # Prendi i primi fixed_frame_count frame
            frames = frames[:self.fixed_frame_count]
        
        frames_np = np.array(frames, dtype=np.float32)
        video = torch.tensor(frames_np).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        
        # Estrarre un frame centrale
        mid_frame = frames[len(frames) // 2]
        mid_frame_np = np.array(mid_frame, dtype=np.float32)
        mid_frame = torch.tensor(mid_frame_np).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # Ottieni le descrizioni del video
        video_id = os.path.splitext(video_file)[0]
        descriptions = self.video_descriptions.get(video_id, [])

        print(f"description of __getitem__: {descriptions} video_id: {video_id}")
        
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

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg.children())[:18]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.layers(x)
        y_features = self.layers(y)
        return nn.functional.mse_loss(x_features, y_features)


def decode_latents(latents, vae):
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = einops.rearrange(latents, "b c f h w -> (b f) c h w")
    
    decoded_parts = []
    batch_size = 1

    for i in range(0, latents.shape[0], batch_size):
        latents_batch = latents[i:i+batch_size]
        
        # Usa vae.decode direttamente senza checkpoint
        decoded_batch = vae.decode(latents_batch).sample
        
        decoded_parts.append(decoded_batch)
    
    video = torch.cat(decoded_parts, dim=0)
    video = einops.rearrange(video, "(b f) c h w -> b f h w c", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    
    return video

'''
def decode_latents(latents, vae):
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = einops.rearrange(latents, "b c f h w -> (b f) c h w")
    
    # Utilizzare torch.no_grad() per risparmiare memoria
    with torch.no_grad():
        video = vae.decode(latents).sample

    video = einops.rearrange(video, "(b f) c h w -> b f h w c", f=video_length)
    video = ((video / 2 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().contiguous()
    return video
'''

def train_lora_model(data, video_folder, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Carica il modello UNet e applica LoRA
    sd_path = args.pretrained_path + "/stable-diffusion-v1-4"
    unet = get_models(args, sd_path).to(device, dtype=torch.float16)
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)

    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16).to(device)

    # Load CLIP model and processor for image conditioning
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16,
        target_modules=["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn2.to_q", "attn2.to_k", "attn2.to_v"]
    )

    unet = get_peft_model(unet, lora_config)
    
    #dataset = VideoDatasetMsrvtt(data, video_folder)
    dataset = VideoDatasetMsvd(annotations_file=data, video_dir=video_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)
    num_epochs = 3
    
    unet.train()
    unet.enable_xformers_memory_efficient_attention()
    unet.enable_gradient_checkpointing()
    text_encoder.eval()
    vae.eval()
       
    conta = 1

    
    attention_layer = nn.MultiheadAttention(embed_dim=768, num_heads=8).to(unet.device)
    #projection_layer = nn.Linear(64, 224).to(unet.device)

    accumulation_steps = 8

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        for i, (video, description, frame_tensor) in enumerate(dataloader):

            video = video.to(device)
            optimizer.zero_grad()
            print(f"Iterazione numero: {conta}")
            conta += 1

            print(f"description: {description}")
            print(f"video shape: {video.shape}, dtype: {video.dtype}")

            with torch.cuda.amp.autocast():

                text_inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True).input_ids.to(unet.device)
                text_features = text_encoder(text_inputs)[0].to(torch.float16)
                print(f"text_features shape: {text_features.shape}, dtype: {text_features.dtype}")

                image_inputs = clip_processor(images=frame_tensor, return_tensors="pt").pixel_values.to(unet.device)
                outputs = clip_model.vision_model(image_inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1].to(torch.float16)
                print(f"last_hidden_state shape: {last_hidden_state.shape}, dtype: {last_hidden_state.dtype}")
                
                # Trasponiamo le dimensioni per adattarsi al MultiheadAttention
                text_features = text_features.transpose(0, 1)
                last_hidden_state = last_hidden_state.transpose(0, 1)

                print(f"text_features_transpose shape: {text_features.shape}, dtype: {text_features.dtype}")
                print(f"last_hidden_state_transpose shape: {last_hidden_state.shape}, dtype: {last_hidden_state.dtype}")

                assert text_features.dtype == last_hidden_state.dtype, "text_features and last_hidden_state must have the same dtype"

                # Calcola l'attenzione
                attention_output, _ = attention_layer(text_features, last_hidden_state, last_hidden_state)

                print(f"attention_output shape: {attention_output.shape}, dtype: {attention_output.dtype}")
                
                # Ritorna alle dimensioni originali
                attention_output = attention_output.transpose(0, 1)

                print(f"attention_output_transpose shape: {attention_output.shape}, dtype: {attention_output.dtype}")

                #attention_output = projection_layer(attention_output).to(torch.float16)
                encoder_hidden_states = attention_output

                print(f"encoder_hidden_states shape: {encoder_hidden_states.shape}, dtype: {encoder_hidden_states.dtype}")

                timestep=torch.randint(0, 1000, (1,)).to(unet.device)

                print(f"timestep shape: {timestep.shape}, dtype: {timestep.dtype}")

                sample=torch.randn(1, 4, 16, 40, 64).to(unet.device, dtype=torch.float16)
                #sample=torch.randn(2, 4, 21, 32, 32).to(unet.device, dtype=torch.float16)

                # Forward pass
                output = unet(
                    sample=sample,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                output = decode_latents(output, vae)

                output = output.to(video.dtype) 

                # Riorganizza le dimensioni per combaciare con video
                output = output.permute(0, 4, 1, 2, 3)

                print(f"output shape: {output.shape}, dtype: {output.dtype}")

                print(f"UNet requires grad: {any(p.requires_grad for p in unet.parameters())}")
                print(f"VAE requires grad: {any(p.requires_grad for p in vae.parameters())}")
                print(f"Output requires grad: {output.requires_grad}")

                loss = torch.nn.functional.mse_loss(output, video)

                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                try:
                    scaler.step(optimizer)
                    scaler.update()
                except ValueError as e:
                    print(f"Skipping scaler step due to error: {e}")
                optimizer.zero_grad()


            del text_features, image_inputs, last_hidden_state, attention_output, encoder_hidden_states
            torch.cuda.empty_cache()

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