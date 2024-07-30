from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import cv2
import os
import json
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
import argparse
from omegaconf import OmegaConf
import imageio
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
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from transformers import get_cosine_schedule_with_warmup
from dataclasses import dataclass
from peft import PeftModel, LoraConfig
from PIL import Image
from torchvision import transforms

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

@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    video: torch.Tensor

def load_model_for_inference(checkpoint_dir, device, args):

    sd_path = args.pretrained_path + "/stable-diffusion-v1-4"

    # Carica il modello UNet e applica LoRA
    unet = get_models(args, sd_path).to(device, dtype=torch.float16)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
            "attn_temp.to_q", "attn_temp.to_k", "attn_temp.to_v", "attn_temp.to_out.0",
            "ff.net.0.proj", "ff.net.2"
        ]
    )
    
    # Applica LoRA al modello
    unet = get_peft_model(unet, lora_config)
    
    # Carica l'ultimo checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        unet.load_state_dict(checkpoint['model_state_dict'])
        print(f"Caricato checkpoint dall'epoca {checkpoint['epoch']}, iterazione {checkpoint['iteration']}")
    else:
        print("Nessun checkpoint trovato. Utilizzo del modello non addestrato.")

    
    unet.eval()
    
    # Carica gli altri componenti necessari
    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16).to(device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    text_encoder.eval()
    vae.eval()
    clip_model.eval()
    
    return unet, tokenizer, text_encoder, vae, clip_model, clip_processor, noise_scheduler


def inference(unet, tokenizer, text_encoder, vae, clip_model, clip_processor, noise_scheduler, description, input_image, device, guidance_scale=7.5):
    with torch.no_grad():
        # Preparazione del testo
        text_inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        text_features = text_encoder(text_inputs)[0].to(torch.float16)
        
        # Preparazione del video/immagine
        image_inputs = clip_processor(images=input_image, return_tensors="pt").pixel_values.to(unet.device)
        outputs = clip_model.vision_model(image_inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1].to(torch.float16)

        # Calcolo dell'attenzione (come nel training)
        attention_layer = torch.nn.MultiheadAttention(embed_dim=768, num_heads=8, dtype=torch.float16).to(device)
        text_features = text_features.transpose(0, 1)
        last_hidden_state = last_hidden_state.transpose(0, 1)

        text_features = text_features.to(torch.float16)
        last_hidden_state = last_hidden_state.to(torch.float16)
        attention_layer = attention_layer.to(torch.float16)

        attention_output, _ = attention_layer(text_features, last_hidden_state, last_hidden_state)
        encoder_hidden_states = attention_output.transpose(0, 1)
        
        # Generazione dell'output
        latents = torch.randn((1, 4, 16, 40, 64), device=device) #Campiona X_T ∼ N(0,I). Qui, latents rappresenta X_T
        
        noise_scheduler.set_timesteps(1000)
        
        for t in tqdm(noise_scheduler.timesteps):                               #Il ciclo for scorre attraverso i timestep t da T a 1. Il campionamento di z ∼ N(0, I) se t > 1, altrimenti z = 0 è implicito nel metodo noise_scheduler.step
            latent_model_input = noise_scheduler.scale_model_input(latents, t)

            latent_model_input = latent_model_input.to(torch.float16)
            #unet = unet.to(torch.float16)
            encoder_hidden_states = encoder_hidden_states.to(torch.float16)
            
            # Genera la previsione non condizionata
            uncond_embeddings = torch.zeros_like(encoder_hidden_states)
            noise_pred_uncond = unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=uncond_embeddings
            ).sample

            # Genera la previsione condizionata
            noise_pred = unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states
            ).sample

            # Applica la guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            
            # Compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample #Qui, noise_pred rappresenta ϵ_θ(x_t, t). La funzione noise_scheduler.step calcola x_t-1 utilizzando l'equazione descritta

        #print(f"latents1 shape: {latents.shape}, dtype: {latents.dtype}")
        
        # Decodifica dei latents in immagine
        latents = 1 / vae.config.scaling_factor * latents

        #print(f"latents2 shape: {latents.shape}, dtype: {latents.dtype}")

        video = decode_latents(latents, vae, gradient=False)
        
    return StableDiffusionPipelineOutput(video=video)




class VideoDatasetMsvd(Dataset):
    def __init__(self, annotations_file, video_dir, transform=None, target_size=(256, 256), fixed_frame_count=5):
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

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file)

        try:

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

            #if not descriptions or descriptions == []:
                #print(f"No description found for video {video_id}")
                #return None, None, None

            #print(f"description of __getitem__: {descriptions} video_id: {video_id}")
            
            # Applica trasformazioni, se presenti
            if self.transform:
                video = self.transform(video)
                mid_frame = self.transform(mid_frame)
            
            return video, descriptions, mid_frame
        
        except Exception as e:
            print(f"Skipping video {video_file} due to error: {e}")
            return None, None, None



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


def encode_latents(video, vae):
    video = video.to(torch.float16)

    # video ha forma [b, c, f, h, w]
    b, c, f, h, w = video.shape
    
    # Riarrangia il video in una serie di immagini
    video = einops.rearrange(video, "b c f h w -> (b f) c h w")
    
    encode_parts = []
    batch_size = 1  # Puoi aumentare questo valore se la tua GPU lo consente


    def encode_batch(batch):
        return vae.encode(batch).latent_dist.sample()
    
    for i in range(0, video.shape[0], batch_size):
        latents_batch = video[i:i+batch_size]
        latents_batch = latents_batch.requires_grad_(True)
        #print(f"latents_batch shape: {latents_batch.shape}, dtype: {latents_batch.dtype}") #shape: torch.Size([1, 3, 320, 512]), dtype: torch.float32

        # Usa checkpoint per risparmiare memoria
        encoded_batch = checkpoint(encode_batch, latents_batch, use_reentrant=False)

        #print(f"encoded_batch shape: {encoded_batch.shape}, dtype: {encoded_batch.dtype}") #shape: torch.Size([1, 4, 40, 64]), dtype: torch.float32

        encode_parts.append(encoded_batch)
        # Libera un po' di memoria
        #torch.cuda.empty_cache()

    latents = torch.cat(encode_parts, dim=0)
        
    # Riarrangia i latents per reintrodurre la dimensione temporale
    latents = einops.rearrange(latents, "(b f) c h w -> b c f h w", b=b)
    
    return latents
    
'''
def decode_latents(latents, vae):
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = einops.rearrange(latents, "b c f h w -> (b f) c h w")

    print(f"latents requires grad: {latents.requires_grad}")
    
    # Utilizzare torch.no_grad() per risparmiare memoria
    with torch.no_grad():
        video = vae.decode(latents).sample

    video = einops.rearrange(video, "(b f) c h w -> b f h w c", f=video_length)
    video = ((video / 2 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().contiguous()
    print(f"video requires grad: {video.requires_grad}")
    return video
'''

def decode_latents(latents, vae, gradient=True):
    video_length = latents.shape[2]
    #latents = 1 / 0.18215 * latents
    latents = einops.rearrange(latents, "b c f h w -> (b f) c h w")
    
    decoded_parts = []
    batch_size = 1  # Puoi aumentare questo valore se la tua GPU lo consente

    def decode_batch(batch):
        return vae.decode(batch).sample

    #print(f"latents shape: {latents.shape}, dtype: {latents.dtype}") #[16, 4, 40, 64] torch.float16
    
    for i in range(0, latents.shape[0], batch_size):
        latents_batch = latents[i:i+batch_size]
        latents_batch = latents_batch.requires_grad_(True)
        # Usa checkpoint per risparmiare memoria
        decoded_batch = checkpoint(decode_batch, latents_batch, use_reentrant=False)
        decoded_parts.append(decoded_batch)
        # Libera un po' di memoria
        #torch.cuda.empty_cache()
    
    #print(f"latents_batch shape: {latents_batch.shape}, dtype: {latents_batch.dtype}") #[1, 4, 40, 64] torch.float16


    # Concatena tutte le parti decodificate
    video = torch.cat(decoded_parts, dim=0)
    
    # Riorganizza le dimensioni del video
    video = einops.rearrange(video, "(b f) c h w -> b f h w c", f=video_length)
    
    if(gradient):
        # Normalizza e converti a uint8 senza perdere il tracciamento dei gradienti
        video = (video / 2 + 0.5) * 255
        video = video.add(0.5).clamp(0, 255)
    else:
        video = ((video / 2 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().contiguous()

    
    return video

def custom_collate(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None and x[2] is not None, batch))
    if len(batch) == 0:
        return None, None, None
    return torch.utils.data.dataloader.default_collate(batch)


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
        target_modules = [
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
            "attn_temp.to_q", "attn_temp.to_k", "attn_temp.to_v", "attn_temp.to_out.0",
            "ff.net.0.proj", "ff.net.2"]
    )

    unet = get_peft_model(unet, lora_config)

    batch_size=1
    
    #dataset = VideoDatasetMsrvtt(data, video_folder)
    dataset = VideoDatasetMsvd(annotations_file=data, video_dir=video_folder)
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
    print(f"Numero totale di elementi nel dataloader: {len(dataloader)}")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(dataloader) * batch_size),
    )

    num_epochs = 20
    checkpoint_dir = "/content/drive/My Drive/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_interval = 20  # Salva un checkpoint ogni 100 iterazioni

    start_epoch = 0
    iteration = 0
    if os.path.exists(os.path.join(checkpoint_dir, "latest_checkpoint.pth")):
        checkpoint = torch.load(os.path.join(checkpoint_dir, "latest_checkpoint.pth"))
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        print(f"Ripresa dell'addestramento dall'epoca {start_epoch}, iterazione {iteration}")

    
    unet.train()
    unet.enable_xformers_memory_efficient_attention()
    unet.enable_gradient_checkpointing()
    text_encoder.eval()
    vae.eval()
    
    attention_layer = nn.MultiheadAttention(embed_dim=768, num_heads=8).to(unet.device)
    #projection_layer = nn.Linear(64, 224).to(unet.device)

    accumulation_steps = 8

    scaler = torch.cuda.amp.GradScaler()

    noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")


    for epoch in range(num_epochs):

        if epoch < start_epoch:
            continue  # Salta le epoche già completate

        for i, batch in enumerate(dataloader):

            if epoch == start_epoch and i <= iteration:
                continue

            if batch[0] is None:
                continue

            video, description, frame_tensor = batch

            #print(f"frame_tensor shape: {frame_tensor.shape}, dtype: {frame_tensor.dtype}") #frame_tensor shape: torch.Size([1, 3, 320, 512]), dtype: torch.float32

            print(f"epoca {epoch}, iterazione {i}")

            video = video.to(device)
            optimizer.zero_grad()

            #print(f"train_lora_model video shape: {video.shape}, dtype: {video.dtype}") #[1, 3, 16, 320, 512] torch.float32

            text_inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True).input_ids.to(unet.device)
            text_features = text_encoder(text_inputs)[0].to(torch.float16)
            #print(f"train_lora_model text_features shape: {text_features.shape}, dtype: {text_features.dtype}") #[1, 10, 768] torch.float16

            image_inputs = clip_processor(images=frame_tensor, return_tensors="pt").pixel_values.to(unet.device)
            outputs = clip_model.vision_model(image_inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1].to(torch.float16)
            #print(f"train_lora_model last_hidden_state shape: {last_hidden_state.shape}, dtype: {last_hidden_state.dtype}") #[1, 50, 768] torch.float16
            
            # Trasponiamo le dimensioni per adattarsi al MultiheadAttention
            text_features = text_features.transpose(0, 1)
            last_hidden_state = last_hidden_state.transpose(0, 1)

            assert text_features.dtype == last_hidden_state.dtype, "text_features and last_hidden_state must have the same dtype"

            attention_layer = attention_layer.to(torch.float16)

            # Calcola l'attenzione
            attention_output, _ = attention_layer(text_features, last_hidden_state, last_hidden_state)

            #print(f"train_lora_model attention_output shape: {attention_output.shape}, dtype: {attention_output.dtype}") #[10, 1, 768] torch.float16
            
            # Ritorna alle dimensioni originali
            encoder_hidden_states = attention_output.transpose(0, 1)

            print(f"train_lora_model encoder_hidden_states shape: {encoder_hidden_states.shape}, dtype: {encoder_hidden_states.dtype}") #[1, 10, 768] torch.float16

            latents = encode_latents(video, vae)
            #print(f"train_lora_model latents1 shape: {latents.shape}, dtype: {latents.dtype}") #shape: torch.Size([1, 4, 16, 40, 64]), dtype: torch.float32

            latents = latents * vae.config.scaling_factor #Campiona x_0 ∼ q(x_0). Qui, latents rappresenta x_0

            #print(f"train_lora_model latents2 shape: {latents.shape}, dtype: {latents.dtype}") #shape: torch.Size([1, 4, 16, 40, 64]), dtype: torch.float32

            noise = torch.randn_like(latents) #campiona ϵ ∼ N(0, I). Qui, noise rappresenta ϵ

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device) #Campiona t ∼ Uniform({1, . . . , T}). Qui, timesteps rappresenta il campionamento di t

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) #Qui, noisy_latents rappresenta x_{t} = \sqrt{\overline{a}_{t}}x_{0} + \sqrt{1 - \overline{a}_{t}} \epsilon

            #print(f"train_lora_model noisy_latents shape: {noisy_latents.shape}, dtype: {noisy_latents.dtype}") #shape: torch.Size([1, 4, 16, 40, 64]), dtype: torch.float32

            with torch.cuda.amp.autocast():
                # Forward pass
                output = unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                #Qui, output rappresenta ϵ_θ(x_t, t)

            loss = F.mse_loss(output, noise) #calcola || \epsilon - \epsilon_{\theta}(x_{t}, t) ||^{2}

            loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()

            
            if (i + 1) % accumulation_steps == 0:
                try:
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                except ValueError as e:
                    print(f"Skipping scaler step due to error: {e}")
                optimizer.zero_grad()


            del text_features, image_inputs, last_hidden_state, attention_output, encoder_hidden_states
            torch.cuda.empty_cache()

            if i % len(dataloader) == 0:
                print(f"Epoch {epoch}/{num_epochs} completed with loss: {loss.item()}")


            # Salva un checkpoint
            if i % checkpoint_interval == 0:
                
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}_iter{i}.pth")
                '''
                torch.save({
                    'epoch': epoch,
                    'iteration': i,
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                '''

                # Aggiorna il checkpoint più recente
                torch.save({
                    'epoch': epoch,
                    'iteration': i,
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'loss': loss.item(),
                }, os.path.join(checkpoint_dir, "latest_checkpoint.pth"))

                print(f"Checkpoint salvato: {checkpoint_path}")
            
    unet.save_pretrained("/content/drive/My Drive/finetuned_lora_unet")
    
    return unet
    

def main(args):

    
    
    # Determina se sei su Google Colab
    on_colab = 'COLAB_GPU' in os.environ

    if on_colab:
        # Percorso del dataset su Google Colab
        dataset_path = '/content/drive/My Drive/msvd_small'
    else:
        # Percorso del dataset locale (sincronizzato con Google Drive)
        dataset_path = '/path/to/your/Google_Drive/sync/folder/path/to/your/dataset'
    
    # Percorsi dei file
    video_folder = os.path.join(dataset_path, 'YouTubeClips')
    data = os.path.join(dataset_path, 'annotations.txt')
    
    train_lora_model(data, video_folder, args)

    '''

    #inference

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = "/content/drive/My Drive/checkpoints"

    unet, tokenizer, text_encoder, vae, clip_model, clip_processor, noise_scheduler = load_model_for_inference(checkpoint_dir, device, args)

    description = "A squirrel eating"

    image_path = "/content/drive/My Drive/What-to-Feed-Squirrels.webp"

    image = Image.open(image_path)

    # Definisci la trasformazione
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()  # Converte l'immagine in un tensore
    ])

    # Applica la trasformazione all'immagine
    input_image = transform(image)

    image_tensor = input_image.unsqueeze(0).to(torch.float16)  # Aggiunge una dimensione per il batch

    print(f"image_tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")

    video = inference(unet, tokenizer, text_encoder, vae, clip_model, clip_processor, noise_scheduler, description, image_tensor, device, guidance_scale=7.5).video
    imageio.mimwrite(args.output_folder + 'ocean' + '.mp4', video[0], fps=8, quality=9) # highest quality is 10, lowest is 0

    print('save path {}'.format(args.output_folder))
    '''

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    main(OmegaConf.load(args.config))