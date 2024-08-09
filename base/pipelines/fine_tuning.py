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
from pathlib import Path
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
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
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler
from transformers import get_cosine_schedule_with_warmup
from dataclasses import dataclass
from peft import PeftModel, LoraConfig
from PIL import Image
from torchvision import transforms
import plotly.graph_objects as go
from inference import VideoGenPipeline

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

def load_model_for_inference(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sd_path = args.pretrained_path + "/stable-diffusion-v1-4"
    unet = get_models(args, sd_path).to(device, dtype=torch.float16)
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # Applica LoRA al modello
    unet = get_peft_model(unet, lora_config)
    
    # Carica l'ultimo checkpoint
    checkpoint_dir = "/content/drive/My Drive/checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        unet.load_state_dict(checkpoint['model_state_dict'])
        print(f"Caricato checkpoint dall'epoca {checkpoint['epoch']}, iterazione {checkpoint['iteration']}")
    else:
        print("Nessun checkpoint trovato. Utilizzo del modello non addestrato.")
    
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16).to(device)
    tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device) # huge

    # set eval mode
    unet.eval()
    vae.eval()
    text_encoder_one.eval()
    
    if args.sample_method == 'ddim':
        scheduler = DDIMScheduler.from_pretrained(sd_path, 
                                            subfolder="scheduler",
                                            beta_start=args.beta_start, 
                                            beta_end=args.beta_end, 
                                            beta_schedule=args.beta_schedule)
    elif args.sample_method == 'eulerdiscrete':
        scheduler = EulerDiscreteScheduler.from_pretrained(sd_path,
                                            subfolder="scheduler",
                                            beta_start=args.beta_start,
                                            beta_end=args.beta_end,
                                            beta_schedule=args.beta_schedule)
    elif args.sample_method == 'ddpm':
        scheduler = DDPMScheduler.from_pretrained(sd_path,
                                            subfolder="scheduler",
                                            beta_start=args.beta_start,
                                            beta_end=args.beta_end,
                                            beta_schedule=args.beta_schedule)
    else:
        raise NotImplementedError

    videogen_pipeline = VideoGenPipeline(vae=vae, 
                                text_encoder=text_encoder_one, 
                                tokenizer=tokenizer_one, 
                                scheduler=scheduler, 
                                unet=unet).to(device)
    videogen_pipeline.enable_xformers_memory_efficient_attention()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    video_grids = []
    for prompt in args.text_prompt:
        print('Processing the ({}) prompt'.format(prompt))
        videos = videogen_pipeline(prompt, 
                                video_length=args.video_length, 
                                height=args.image_size[0], 
                                width=args.image_size[1], 
                                num_inference_steps=args.num_sampling_steps,
                                guidance_scale=args.guidance_scale).video
        imageio.mimwrite(args.output_folder + prompt.replace(' ', '_') + '.mp4', videos[0], fps=8, quality=9) # highest quality is 10, lowest is 0
    
    print('save path {}'.format(args.output_folder))
    


class VideoDatasetMsvd(Dataset):
    def __init__(self, annotations_file, video_dir, transform=None, target_size=(320, 512), fixed_frame_count=16):
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
                    self.video_descriptions[video_id] = description
        
        # Ottieni la lista dei file video nella cartella YouTubeClips
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]

        print(f"video_files: {self.video_files}")
        print(f"video_descriptions: {self.video_descriptions}")

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
            
            frames_np = np.array(frames)
            frames_np = frames_np.astype(np.float32) / 255.0  # Normalizza in [0, 1]
            frames_np = (frames_np - 0.5) / 0.5

            video = torch.tensor(frames_np).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
            
            # Estrarre un frame centrale
            mid_frame = frames[len(frames) // 2]
            mid_frame_np = np.array(mid_frame)

            mid_frame_np = mid_frame_np.astype(np.float32) / 255.0  # Normalizza in [0, 1]
            mid_frame_np = (mid_frame_np - 0.5) / 0.5

            mid_frame = torch.tensor(mid_frame_np).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            # Ottieni le descrizioni del video
            video_id = os.path.splitext(video_file)[0]
            descriptions = self.video_descriptions.get(video_id, [])

            #print(f"description of __getitem__: {descriptions} video_id: {video_id}")
            
            # Applica trasformazioni, se presenti
            if self.transform:
                video = self.transform(video)
                mid_frame = self.transform(mid_frame)
            
            return video, descriptions, mid_frame
        
        except Exception as e:
            print(f"Skipping video {video_file} due to error: {e}")
            return None, None, None


def encode_latents_(video, vae):
    video = video.to(torch.float16)
    b, c, f, h, w = video.shape
    video = einops.rearrange(video, "b c f h w -> (b f) c h w")
    
    latents = vae.encode(video).latent_dist.sample()
    latents = einops.rearrange(latents, "(b f) c h w -> b c f h w", b=b)
    
    return latents

def encode_latents(video, vae):
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


def custom_collate(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None and x[2] is not None, batch))
    if len(batch) == 0:
        return None, None, None
    return torch.utils.data.dataloader.default_collate(batch)




def train_lora_model(data, video_folder, args):

    logging_dir = Path("/content/drive/My Drive/", "/content/drive/My Drive/")

    accelerator_project_config = ProjectConfiguration(project_dir="/content/drive/My Drive/", logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=None,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Carica il modello UNet e applica LoRA
    sd_path = args.pretrained_path + "/stable-diffusion-v1-4"
    unet = get_models(args, sd_path).to(device, dtype=torch.float16)
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)

    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16).to(device)

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float16
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    for param in unet.parameters():
        param.requires_grad_(False)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    
    unet = get_peft_model(unet, lora_config)
    
    #dataset = VideoDatasetMsrvtt(data, video_folder)
    dataset = VideoDatasetMsvd(annotations_file=data, video_dir=video_folder)
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
    print(f"Numero totale di elementi nel dataloader: {len(dataloader)}")

    #optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer = torch.optim.AdamW(lora_layers, 
        lr=1e-4, 
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )


    num_epochs = 100
    checkpoint_dir = "/content/drive/My Drive/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    count = 0
    start_epoch = 0
    iteration = 0
    if os.path.exists(os.path.join(checkpoint_dir, "latest_checkpoint.pth")):
        checkpoint = torch.load(os.path.join(checkpoint_dir, "latest_checkpoint.pth"))
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        print(f"Ripresa dell'addestramento dall'epoca {start_epoch}, iterazione {iteration}")

    
    unet.train()
    unet.enable_xformers_memory_efficient_attention()
    unet.enable_gradient_checkpointing()
    text_encoder.eval()
    vae.eval()

    noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    videogen_pipeline = VideoGenPipeline(vae=vae, 
                            text_encoder=text_encoder, 
                            tokenizer=tokenizer, 
                            scheduler=noise_scheduler, 
                            unet=unet).to(device)
    videogen_pipeline.enable_xformers_memory_efficient_attention()

    epoch_losses = []

    for epoch in range(num_epochs):

        batch_losses = []

        if epoch < start_epoch:
            continue  # Salta le epoche già completate

        for i, batch in enumerate(dataloader):

            count += 1

            if epoch == start_epoch and i <= iteration and (start_epoch != 0 or iteration != 0):
                continue

            if batch[0] is None:
                continue

            video, description, frame_tensor = batch

            print(f"epoca {epoch}, iterazione {i}")

            video = video.to(device)
        
            '''
            # Esegui check_inputs prima di elaborare il prompt
            videogen_pipeline.check_inputs(
                prompt=description[0],
                height=video.shape[2],  # Altezza del video
                width=video.shape[3],   # Larghezza del video
                callback_steps=1        # O il valore appropriato per il tuo caso
            )
            
            # Utilizzo di _encode_prompt per elaborare il testo
            text_embeddings = videogen_pipeline._encode_prompt(
                description[0],
                device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False  # Impostiamo questo a False per il training
            )
            '''

            text_inputs = tokenizer(description[0], return_tensors="pt", padding=True, truncation=True).input_ids.to(unet.device)
            text_features = text_encoder(text_inputs)[0].to(torch.float16)

            # Codifica i latenti dal video di input
            latents = encode_latents(video, vae)
            latents = latents * vae.config.scaling_factor

            # Genera rumore e timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, videogen_pipeline.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
            
            # Aggiungi rumore ai latenti
            noisy_latents = videogen_pipeline.scheduler.add_noise(latents, noise, timesteps)

            # Preparazione dell'input per l'UNet
            latent_model_input = videogen_pipeline.scheduler.scale_model_input(noisy_latents, timesteps)

            # Forward pass attraverso l'UNet
            noise_pred = unet(
                sample=latent_model_input,
                timestep=timesteps,
                encoder_hidden_states=text_features
            ).sample

            # Calcolo della loss
            loss = F.mse_loss(noise_pred, noise)

            batch_losses.append(loss.item())

            # Backpropagation e step dell'ottimizzatore
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_epoch_loss = sum(batch_losses) / len(batch_losses)
        print(f"Epoch {epoch}/{num_epochs} completed with average loss: {avg_epoch_loss}")
        epoch_losses.append(avg_epoch_loss)

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                for prompt in args.text_prompt:
                    print('Processing the ({}) prompt'.format(prompt))
                    videos = videogen_pipeline(prompt, 
                                            video_length=args.video_length, 
                                            height=args.image_size[0], 
                                            width=args.image_size[1], 
                                            num_inference_steps=args.num_sampling_steps,
                                            guidance_scale=args.guidance_scale).video
                    imageio.mimwrite("/content/drive/My Drive/" + f"sample_epoch_{epoch}.mp4", videos[0], fps=8, quality=9) # highest quality is 10, lowest is 0

                print('save path {}'.format("/content/drive/My Drive/"))

    
    num_epochs = len(epoch_losses)
    epochs = list(range(1, num_epochs + 1))  # Crea una lista [1, 2, 3, ..., num_epochs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=epoch_losses, mode='lines'))

    # Personalizza il layout
    fig.update_layout(
        title='Training Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss'
    )

    # Salva il grafico come immagine
    fig.write_image("/content/drive/My Drive/training_loss.png")

    # Opzionale: visualizza il grafico interattivo in Colab
    fig.show()

    print(epoch_losses)
    
    return unet


def training(args):
    
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    training(OmegaConf.load(args.config))
    #load_model_for_inference(OmegaConf.load(args.config))