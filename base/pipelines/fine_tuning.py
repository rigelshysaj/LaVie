from torch.utils.data import Dataset, DataLoader
from typing import List, Union
import torch
import os
from tqdm import tqdm
import plotly.graph_objs as go
import shutil
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
import argparse
from diffusers.utils.torch_utils import is_compiled_module
from peft.utils import get_peft_model_state_dict
from omegaconf import OmegaConf
import imageio
import logging
import math
import transformers
import torch.optim as optim
from diffusers.optimization import get_scheduler
import diffusers
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
import torch.nn.functional as F
from diffusers.utils import convert_state_dict_to_diffusers
import einops
from diffusers import AutoencoderKL, StableDiffusionPipeline
from diffusers.schedulers import DDPMScheduler
from dataclasses import dataclass
from peft import LoraConfig
from PIL import Image
from torchvision import transforms
import plotly.graph_objects as go
from inference import VideoGenPipeline
from msvd import VideoDatasetMsvd
from mapping import MappingNetwork
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from diffusers.utils import (
    logging,
    BaseOutput,
)

logger = logging.get_logger(__name__)

@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    video: torch.Tensor



def load_and_transform_image(path):
    image = Image.open(path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((320, 512)),
        transforms.ToTensor(),
    ])

    # Applica la trasformazione all'immagine
    input_image = transform(image)

    image_tensor = input_image.unsqueeze(0).to(torch.float32)  # Aggiunge una dimensione per il batch

    print(f"image_tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}") #shape: torch.Size([1, 3, 320, 512]), dtype: torch.float32

    return image_tensor


def inference(args, vae, text_encoder, tokenizer, noise_scheduler, clip_processor, clip_model, unet, original_unet, device, mapper, caption):
        
    mapper.dtype = next(mapper.parameters()).dtype

    with torch.no_grad():
        # Funzione per generare video
        def generate_video(unet, is_original):
            pipeline = VideoGenPipeline(
                vae=vae, 
                text_encoder=text_encoder, 
                tokenizer=tokenizer, 
                scheduler=noise_scheduler, 
                unet=unet,
                clip_processor=clip_processor,
                clip_model=clip_model,
                mapper=mapper
            ).to(device)

            pipeline.enable_xformers_memory_efficient_attention()


            if(not is_original):
                image_tensor = load_and_transform_image(args.image_path)
            
            
            print(f'Processing the ({caption}) prompt for {"original" if is_original else "fine-tuned"} model')
            videos = pipeline(
                caption,
                image_tensor=image_tensor if not is_original else None,
                video_length=args.video_length, 
                height=args.image_size[0], 
                width=args.image_size[1], 
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale
            ).video

            suffix = "original" if is_original else "fine_tuned"
            imageio.mimwrite(f"/content/drive/My Drive/{suffix}.mp4", videos[0], fps=8, quality=9)
            return videos[0]

        # Genera video con il modello fine-tuned
        video = generate_video(unet, is_original=False)

        generate_video(original_unet, is_original=True)

        #del original_unet #Poi quando fa l'inference la seconda volta non trova più unet e dice referenced before assignment
        torch.cuda.empty_cache()

        print('save path {}'.format("/content/drive/My Drive/"))

        return video
    


def encode_latents(video, vae):
    vae = vae.to(torch.float16)
    video = video.to(torch.float16)
    b, c, f, h, w = video.shape
    video = einops.rearrange(video, "b f h w c -> (b f) c h w")
    
    vae.enable_slicing()
    latents = vae.encode(video).latent_dist.sample()
    vae.disable_slicing()

    latents = einops.rearrange(latents, "(b f) c h w -> b c f h w", b=b)
    
    return latents


def custom_collate(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None and x[2] is not None, batch))
    if len(batch) == 0:
        return None, None, None
    return torch.utils.data.dataloader.default_collate(batch)

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)

def log_lora_weights(model, step):
       for name, param in model.named_parameters():
           if 'lora' in name:
               print(f"Step {step}: LoRA weight '{name}' mean = {param.data.mean().item():.6f}")


def setup_logging(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def load_models(args, device):
    sd_path = os.path.join(args.pretrained_path, "stable-diffusion-v1-4")
    
    # Carica e configura UNet
    unet = get_models(args, sd_path).to(device, dtype=torch.float16)
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)
    original_unet = get_models(args, sd_path).to(device, dtype=torch.float32)
    original_unet.load_state_dict(state_dict)
    
    # Mappatura e altri modelli
    mapper = MappingNetwork().to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
    
    # Disabilita i gradienti per alcuni modelli
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    for param in unet.parameters():
        param.requires_grad_(False)
    
    return {
        "unet": unet,
        "original_unet": original_unet,
        "mapper": mapper,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "noise_scheduler": noise_scheduler
    }


def configure_accelerator(args):
    logging_dir = Path("/content/drive/My Drive/", "/content/drive/My Drive/")
    accelerator_project_config = ProjectConfiguration(project_dir="/content/drive/My Drive/", logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # Disabilita AMP per MPS
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    
    setup_logging(accelerator)
    
    # Imposta il seme
    if args.seed is not None:
        set_seed(args.seed)
    
    device = accelerator.device
    return accelerator, device


def prepare_optimizer(args, accelerator, trainable_params):
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    return optimizer


def prepare_scheduler(args, accelerator, optimizer, len_dataloader):
    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(len_dataloader / args.gradient_accumulation_steps)
        num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        num_training_steps = args.max_train_steps
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps * accelerator.num_processes,
    )
    return lr_scheduler


def load_checkpoint(args, accelerator, output_dir, mapper, unet, num_update_steps_per_epoch):
    global_step = 0
    first_epoch = 0
    
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Ottieni l'ultimo checkpoint
            dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        
        if path:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            mapper.load_state_dict(torch.load(os.path.join(output_dir, path, 'mapper.pt')))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            global_step = 0
            first_epoch = 0
    else:
        global_step = 0
        first_epoch = 0
    
    return global_step, first_epoch


def save_checkpoint(args, accelerator, output_dir, global_step, unet, mapper):
    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    
    # Salva LoRA
    unwrapped_unet = unwrap_model(unet, accelerator=accelerator)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=save_path,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )
    
    # Salva il mapper
    torch.save(mapper.state_dict(), os.path.join(save_path, 'mapper.pt'))
    print("Modello salvato correttamente.")


def training_loop(args, accelerator, device, models, optimizer, lr_scheduler, train_dataloader, mapper_optimizer, mapper_scheduler):
    unet = models["unet"]
    mapper = models["mapper"]
    vae = models["vae"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
    clip_model = models["clip_model"]
    clip_processor = models["clip_processor"]
    noise_scheduler = models["noise_scheduler"]
    
    criterion = nn.CosineEmbeddingLoss()
    
    # Prepara tutto con l'accelerator
    (
        unet,
        mapper,
        optimizer,
        train_dataloader,
        lr_scheduler,
        mapper_optimizer,
        mapper_scheduler
    ) = accelerator.prepare(
        unet, mapper, optimizer, train_dataloader, lr_scheduler, mapper_optimizer, mapper_scheduler
    )
    
    # Calcola i passi di aggiornamento
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    global_step, first_epoch = load_checkpoint(args, accelerator, args.output_dir, mapper, unet, num_update_steps_per_epoch)
    
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    unet.enable_xformers_memory_efficient_attention()
    
    accumulators = {
        "total_loss": 0.0,
        "diffusion_loss": 0.0,
        "alignment_loss": 0.0,
        "cosine_similarity": 0.0,
        "steps": 0,
        "train_loss": 0.0
    }
    
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        mapper.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                try:
                    video, description, frame_tensor = batch
                    # Verifica se la descrizione è valida
                    _ = description[0]
                except Exception as e:
                    print(f"Skipping iteration due to error: {e}")
                    print(description)
                    continue
                
                print(f"Epoca {epoch}, Iterazione {step}, Global Step {global_step}")
                
                # Elaborazione dei latenti e aggiunta di rumore
                latents = encode_latents(video, vae) * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], latents.shape[2], 1, 1), device=latents.device
                    )
                
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Tokenizzazione e codifica testuale
                text_inputs = tokenizer(
                    list(description),
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                text_features = text_encoder(input_ids=text_inputs.input_ids).last_hidden_state.to(torch.float16)
                
                # Elaborazione delle immagini
                image_inputs = clip_processor(images=list(frame_tensor), return_tensors="pt").pixel_values.to(device)
                image_features = clip_model.vision_model(pixel_values=image_inputs).last_hidden_state.to(torch.float16)
                
                # Mappatura delle feature delle immagini nello spazio delle feature testuali
                mapped_image_features = mapper(image_features, text_features)
                mapped_image_embeddings_flat = mapped_image_features.reshape(-1, 768)
                text_embeddings_flat = text_features.reshape(-1, 768)
                
                # Normalizzazione
                mapped_image_embeddings_flat = F.normalize(mapped_image_embeddings_flat, p=2, dim=1)
                text_embeddings_flat = F.normalize(text_embeddings_flat, p=2, dim=1)
                
                # Calcolo della loss di mappatura
                target = torch.ones(mapped_image_embeddings_flat.size(0)).to(device)
                loss_mapper = criterion(mapped_image_embeddings_flat, text_embeddings_flat, target)
                
                # Combinazione delle feature
                combined_features = torch.cat([text_features, mapped_image_features], dim=1)
                encoder_hidden_states = combined_features
                
                # Determina il target in base al tipo di previsione
                if args.prediction_type is not None:
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Previsione del residuo di rumore e calcolo della loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                
                lambda_alignment = 0.1
                diffusion_loss = loss
                alignment_loss = loss_mapper
                total_loss = diffusion_loss + lambda_alignment * alignment_loss
                cosine_sim = F.cosine_similarity(mapped_image_embeddings_flat, text_embeddings_flat).mean().item()
                
                # Accumulo delle metriche
                accumulators["cosine_similarity"] += cosine_sim
                accumulators["total_loss"] += total_loss.detach().item()
                accumulators["diffusion_loss"] += diffusion_loss.detach().item()
                accumulators["alignment_loss"] += alignment_loss.detach().item()
                accumulators["steps"] += 1
                
                # Calcolo della loss media
                avg_loss = accelerator.gather(total_loss.repeat(args.train_batch_size)).mean()
                accumulators["train_loss"] += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagation
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                mapper_optimizer.step()
                lr_scheduler.step()
                mapper_scheduler.step()
                
                optimizer.zero_grad()
                mapper_optimizer.zero_grad()
                
                # Ottimizzazione e aggiornamento
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    if global_step % args.checkpointing_steps == 0:
                        # Calcolo delle loss medie
                        avg_total_loss = accumulators["total_loss"] / accumulators["steps"]
                        avg_diffusion_loss = accumulators["diffusion_loss"] / accumulators["steps"]
                        avg_alignment_loss = accumulators["alignment_loss"] / accumulators["steps"]
                        avg_cosine_similarity = accumulators["cosine_similarity"] / accumulators["steps"]
                        avg_train_loss = accumulators["train_loss"] / accumulators["steps"]
                        
                        # Resetta gli accumulatori
                        for key in accumulators:
                            accumulators[key] = 0.0
                        
                        # Log delle metriche
                        print(f"Total Loss: {avg_total_loss}")
                        print(f"Diffusion Loss: {avg_diffusion_loss}")
                        print(f"Alignment Loss: {avg_alignment_loss}")
                        print(f"Cosine Similarity: {avg_cosine_similarity}")
                        print(f"Learning Rate: {lr_scheduler.get_last_lr()[0]}")
                        print(f"Avg Train Loss: {avg_train_loss}")
                        
                        # Salvataggio del checkpoint
                        if accelerator.is_main_process:
                            # Gestione del limite dei checkpoint
                            if args.checkpoints_total_limit is not None:
                                checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[:num_to_remove]
                                    logger.info(f"Rimuovo i checkpoint: {', '.join(removing_checkpoints)}")
                                    for cp in removing_checkpoints:
                                        shutil.rmtree(os.path.join(args.output_dir, cp))
                            
                            save_checkpoint(args, accelerator, args.output_dir, global_step, unet, mapper)
                            
                            # Esecuzione dell'inference
                            inference(
                                args, 
                                models["vae"], 
                                models["text_encoder"], 
                                models["tokenizer"], 
                                models["noise_scheduler"], 
                                models["clip_processor"], 
                                models["clip_model"], 
                                models["unet"], 
                                models["original_unet"], 
                                device, 
                                models["mapper"], 
                                args.text_prompt
                            )
                    
                    if global_step >= args.max_train_steps:
                        break
        
        if global_step >= args.max_train_steps:
            break
    
    accelerator.end_training()
    
    # Creazione e salvataggio del grafico delle loss
    num_epochs = args.num_train_epochs
    epochs = list(range(1, num_epochs + 1))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=accumulators["total_loss"], mode='lines'))
    fig.update_layout(
        title='Training Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss'
    )
    fig.write_image("/content/drive/My Drive/training_loss.png")
    fig.show()


def run_inference(args, models, device, caption):
    vae = models["vae"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
    noise_scheduler = models["noise_scheduler"]
    clip_processor = models["clip_processor"]
    clip_model = models["clip_model"]
    unet = models["unet"]
    original_unet = models["original_unet"]
    mapper = models["mapper"]
    
    mapper.dtype = next(mapper.parameters()).dtype

    with torch.no_grad():
        # Funzione per generare video
        def generate_video(unet_model, is_original):
            pipeline = VideoGenPipeline(
                vae=vae, 
                text_encoder=text_encoder, 
                tokenizer=tokenizer, 
                scheduler=noise_scheduler, 
                unet=unet_model,
                clip_processor=clip_processor,
                clip_model=clip_model,
                mapper=mapper
            ).to(device)

            pipeline.enable_xformers_memory_efficient_attention()

            image_tensor = None
            if not is_original:
                image_tensor = load_and_transform_image(args.image_path)
            
            print(f'Processing the ({caption}) prompt for {"original" if is_original else "fine-tuned"} model')
            videos = pipeline(
                caption,
                image_tensor=image_tensor if not is_original else None,
                video_length=args.video_length, 
                height=args.image_size[0], 
                width=args.image_size[1], 
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale
            ).video

            suffix = "original" if is_original else "fine_tuned"
            save_path = f"/content/drive/My Drive/{suffix}.mp4"
            imageio.mimwrite(save_path, videos[0], fps=8, quality=9)
            print(f"Video salvato in: {save_path}")
            return videos[0]

        # Genera video con il modello fine-tuned
        video_fine_tuned = generate_video(unet, is_original=False)
        
        # Opzionale: Genera video con il modello originale
        # video_original = generate_video(original_unet, is_original=True)
        
        # Pulisce la cache CUDA
        torch.cuda.empty_cache()

        print('Salvataggio completato in /content/drive/My Drive/')
    
        return video_fine_tuned


def lora_model(data, video_folder, args, caption, training=True):
    accelerator, device = configure_accelerator(args)
    models = load_models(args, device)
    
    # Configurazione LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    weight_dtype = torch.float32 if accelerator.mixed_precision == "fp32" else (
        torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16
    )
    
    models["unet"].to(accelerator.device, dtype=weight_dtype)
    models["vae"].to(accelerator.device, dtype=weight_dtype)
    models["text_encoder"].to(accelerator.device, dtype=weight_dtype)
    
    models["unet"] = get_peft_model(models["unet"], lora_config)
    
    if args.mixed_precision == "fp16":
        cast_training_params([models["unet"], models["mapper"]], dtype=torch.float32)
    
    # Preparazione del dataset e dataloader
    dataset = VideoDatasetMsvd(annotations_file=data, video_dir=video_folder)
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=custom_collate)
    print(f"Numero totale di elementi nel dataloader: {len(train_dataloader)}")
    
    for param in models["mapper"].parameters():
        param.requires_grad = True
    
    lora_layers = filter(lambda p: p.requires_grad, models["unet"].parameters())
    trainable_params = list(lora_layers) + list(models["mapper"].parameters())
    
    if args.gradient_checkpointing:
        models["unet"].enable_gradient_checkpointing()
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if args.scale_lr:
        args.learning_rate *= args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    
    optimizer = prepare_optimizer(args, accelerator, trainable_params)
    lr_scheduler = prepare_scheduler(args, accelerator, optimizer, len(train_dataloader))
    
    # Ottimizzatore e scheduler per il mapper
    optimizer_mapper = optim.AdamW(models["mapper"].parameters(), lr=1e-4)
    scheduler_mapper = optim.lr_scheduler.StepLR(optimizer_mapper, step_size=10, gamma=0.1)
    
    if training:
        training_loop(args, accelerator, device, models, optimizer, lr_scheduler, train_dataloader, optimizer_mapper, scheduler_mapper)
    else:
        run_inference(args, models, device, caption)


def model(caption):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()
    
    # Determina se sei su Google Colab
    on_colab = 'COLAB_GPU' in os.environ
    
    if on_colab:
        dataset_path = '/content/drive/My Drive/msvd'
    else:
        dataset_path = '/path/to/your/Google_Drive/sync/folder/path/to/your/dataset'
    
    # Percorsi dei file
    video_folder = os.path.join(dataset_path, 'YouTubeClips')
    data = os.path.join(dataset_path, 'annotations.txt')
    
    # Carica la configurazione
    config = OmegaConf.load(args.config)
    
    # Avvia il modello in modalità inferenza
    lora_model(data, video_folder, config, caption, training=True)


if __name__ == "__main__":
    model("a lion is playing with a ball")