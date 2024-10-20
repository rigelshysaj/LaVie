from torch.utils.data import Dataset, DataLoader
from typing import List, Union
import torch
import msrvtt
import clip
import os
import ucf
from tqdm import tqdm
from itertools import cycle
import gc
from torchvision.transforms import Compose, Resize, ConvertImageDtype, Normalize
import torchvision.models.video as models_video
import plotly.graph_objs as go
from torch.utils.data import Subset
import random
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
#import matplotlib
#matplotlib.use('Agg')  # Use non-interactive backend
#import matplotlib.pyplot as plt
#import seaborn as sns

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
        transforms.ToTensor(),
    ])
    # Applica la trasformazione all'immagine
    input_image = transform(image)
    
    # Converte il tensore da [C, H, W] a [H, W, C]
    input_image = input_image.permute(1, 2, 0)
    
    # Aggiunge una dimensione per il batch e converte in uint8
    image_tensor = input_image.unsqueeze(0).mul(255).byte()
    
    print(f"image_tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}") #shape: torch.Size([1, 320, 512, 3]), dtype: torch.uint8
    return image_tensor
    

def inference(args, vae, text_encoder, tokenizer, noise_scheduler, clip_processor, clip_model, unet, original_unet, device, mapper, caption, eval_meth="", frame=None):

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

            # Gestione del caption sia per OmegaConf che per stringhe
            caption_text = caption[0] if OmegaConf.is_list(caption) else caption

            print(f'Processing the ({caption_text}) prompt for {"original" if is_original else "fine-tuned"} model')

            suffix = "original" if is_original else "fine_tuned"
            # Sostituisci gli spazi con underscore nel caption
            formatted_caption = caption_text.replace(' ', '_')

            image_name = f"{suffix}_{formatted_caption}_{eval_meth}.png"

            if(not is_original):
                if(frame is None):
                    image_tensor = load_and_transform_image(args.image_path)
                else:
                    image_tensor = frame

                    #frame_np = frame.squeeze(0).cpu().numpy()
                    # Crea un'immagine PIL
                    #image = Image.fromarray(frame_np)
                    
                    # Salva l'immagine
                    #image.save(f"/content/drive/My Drive/Images/{image_name}")
                    #print(f"Immagine salvata")
            
            
            videos = pipeline(
                caption_text,
                image_tensor=image_tensor if not is_original else None,
                video_length=args.video_length,
                height=args.image_size[0],
                width=args.image_size[1],
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale
            ).video

            # Crea il nome del file
            video_name = f"{suffix}_{formatted_caption}_{eval_meth}.mp4"
            # Usa il nuovo nome del file nella funzione mimwrite
            imageio.mimwrite(f"/content/drive/My Drive/Images/{video_name}", videos[0], fps=8, quality=9)
            #imageio.mimwrite(f"/content/drive/My Drive/Images/{suffix}.mp4", videos[0], fps=8, quality=9)
            return videos[0]

        # Genera video con il modello fine-tuned
        video = generate_video(unet, is_original=False)
        #generate_video(original_unet, is_original=True)

        #del original_unet #Poi quando fa l'inference la seconda volta non trova più unet e dice referenced before assignment
        torch.cuda.empty_cache()
        print('save path {}'.format("/content/drive/My Drive/Images"))
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


def lora_model(data, video_folder, args, method=1):

    logging_dir = Path("/content/drive/My Drive/", "/content/drive/My Drive/")

    accelerator_project_config = ProjectConfiguration(project_dir="/content/drive/My Drive/", logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Carica il modello UNet e applica LoRA
    sd_path = args.pretrained_path + "/stable-diffusion-v1-4"
    unet = get_models(args, sd_path).to(device, dtype=torch.float16)
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)

    # Carica il modello UNet originale
    #original_unet = get_models(args, sd_path).to(device, dtype=torch.float32)
    #original_unet.load_state_dict(state_dict)
    original_unet = None

    # Instantiate the mapping network
    mapper = MappingNetwork().to(unet.device)
    #mapper.load_state_dict(torch.load('/content/drive/My Drive/checkpoints/mapping_network_best.pth'))

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(device)
    # Load CLIP model and processor for image conditioning
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    for param in unet.parameters():
        param.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    
    unet = get_peft_model(unet, lora_config)


    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([unet, mapper], dtype=torch.float32)

    #dataset = VideoDatasetMsrvtt(data, video_folder)
    dataset = VideoDatasetMsvd(annotations_file=data, video_dir=video_folder)
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=custom_collate)
    print(f"Numero totale di elementi nel dataset: {len(dataset)}")
    print(f"Numero totale di elementi nel dataloader: {len(train_dataloader)}")

    for param in mapper.parameters():
        param.requires_grad = True

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    trainable_params = list(lora_layers) + list(mapper.parameters())

    #trainable_params = list(lora_layers)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
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

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )


    optimizer_mapper = optim.AdamW(mapper.parameters(), lr=1e-4)
    scheduler_mapper = optim.lr_scheduler.StepLR(optimizer_mapper, step_size=10, gamma=0.1)

    criterion = nn.CosineEmbeddingLoss()

    # Prepare everything with our `accelerator`.
    unet, mapper, optimizer, train_dataloader, lr_scheduler, optimizer_mapper, scheduler_mapper = accelerator.prepare(
        unet, mapper, optimizer, train_dataloader, lr_scheduler, optimizer_mapper, scheduler_mapper
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            # Load the mapper state dict
            mapper.load_state_dict(torch.load(os.path.join(args.output_dir, path, 'mapper.pt')))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    unet.enable_xformers_memory_efficient_attention()

    print(f"first_epoch: {first_epoch}")
    print(f"num_train_epochs: {args.num_train_epochs}")

    if(method==1):

        accum_total_loss = 0.0
        accum_diffusion_loss = 0.0
        accum_alignment_loss = 0.0
        accum_cosine_similarity = 0.0
        accum_steps = 0
        train_loss = 0.0

        for epoch in range(first_epoch, args.num_train_epochs):
            unet.train()
            mapper.train()

            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):

                    try:
                        video, description, frame_tensor = batch
                        description[0]
                    except Exception as e:
                        print(f"Skipping iteration due to error: {e}")
                        print(description)
                        continue

                    #print(f"frame_tensor shape: {frame_tensor.shape}, dtype: {frame_tensor.dtype}") #torch.Size([8, 320, 512, 3]), dtype: torch.uint8
                    #print(f"video shape: {video.shape}, dtype: {video.dtype}") #torch.Size([8, 16, 320, 512, 3]), dtype: torch.float32

                    print(f"epoca {epoch}, iterazione {step}, global_step {global_step}")

                    latents = encode_latents(video, vae)

                    #print(f"latents shape: {latents.shape}, dtype: {latents.dtype}") #torch.Size([8, 4, 16, 40, 64]), dtype: torch.float16


                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], latents.shape[2], 1, 1), device=latents.device
                        )

                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                                       
                    text_inputs = tokenizer(
                        list(description),
                        max_length=tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).to(unet.device)


                    text_features = text_encoder(
                        input_ids=text_inputs.input_ids,
                    ).last_hidden_state
                    
                    text_features=text_features.to(torch.float16)

                    #print(f"text_features shape: {text_features.shape}, dtype: {text_features.dtype}") #torch.Size([1, 77, 768]), dtype: torch.float16

                    image_inputs = clip_processor(images=list(frame_tensor), return_tensors="pt").pixel_values.to(unet.device)
                    image_features = clip_model.vision_model(
                        pixel_values=image_inputs,
                    ).last_hidden_state

                    #print(f"image_features shape: {image_features.shape}, dtype: {image_features.dtype}") #torch.Size([1, 257, 1024]), dtype: torch.float32

                    image_features=image_features.to(torch.float16)

                    # Map image embeddings to text embedding space using the mapping network
                    mapped_image_features = mapper(image_features, text_features)  # Shape: (batch_size, hidden_size)
                    #print(f"mapped_image_features shape: {mapped_image_features.shape}, dtype: {mapped_image_features.dtype}") #torch.Size([1, 77, 768]), dtype: torch.float32

                    mapped_image_embeddings_flat = mapped_image_features.mean(dim=1)
                    text_embeddings_flat = text_features.mean(dim=1)

                    # Normalize embeddings
                    mapped_image_embeddings_flat = F.normalize(mapped_image_embeddings_flat, p=2, dim=1)
                    text_embeddings_flat = F.normalize(text_embeddings_flat, p=2, dim=1)

                    target = torch.ones(mapped_image_embeddings_flat.size(0)).to(device)  # [batch_size * seq_len]
                    loss_positive = criterion(mapped_image_embeddings_flat, text_embeddings_flat, target)

                    # Genera coppie negative utilizzando in-batch negatives
                    # Shuffle le descrizioni per creare coppie negative
                    indices = torch.randperm(text_embeddings_flat.size(0))
                    text_embeddings_neg = text_embeddings_flat[indices]

                    target_negative = -torch.ones(mapped_image_embeddings_flat.size(0)).to(device)  # [batch_size * 77]
                    loss_negative = criterion(mapped_image_embeddings_flat, text_embeddings_neg, target_negative)

                    loss_mapper = loss_positive + loss_negative
                    
                    encoder_hidden_states = mapped_image_features
                    
                    
                    # Get the target for loss depending on the prediction type
                    if args.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    

                    # Predict the noise residual and compute loss
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                    if args.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                            dim=1
                        )[0]
                        if noise_scheduler.config.prediction_type == "epsilon":
                            mse_loss_weights = mse_loss_weights / snr
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            mse_loss_weights = mse_loss_weights / (snr + 1)

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    lambda_alignment = 0.2

                    # Calcolo della loss di diffusione
                    diffusion_loss = loss  # o rinomina 'loss' in 'diffusion_loss'

                    # Calcolo della loss di allineamento
                    alignment_loss = loss_mapper

                    # Loss totale
                    total_loss = diffusion_loss + lambda_alignment * alignment_loss

                    cosine_sim = F.cosine_similarity(mapped_image_embeddings_flat, text_embeddings_flat)
                    mean_cosine_sim = cosine_sim.mean().item()

                    # Accumula la similarità
                    accum_cosine_similarity += mean_cosine_sim

                    # Accumula le loss
                    accum_total_loss += total_loss.detach().item()
                    accum_diffusion_loss += diffusion_loss.detach().item()
                    accum_alignment_loss += alignment_loss.detach().item()
                    accum_steps += 1


                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(total_loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        params_to_clip = trainable_params
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    optimizer_mapper.step()
                    lr_scheduler.step()
                    scheduler_mapper.step()

                    optimizer_mapper.zero_grad()
                    optimizer.zero_grad()
                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    if global_step % args.checkpointing_steps == 0:

                        # Calcola le loss medie
                        avg_total_loss = accum_total_loss / accum_steps
                        avg_diffusion_loss = accum_diffusion_loss / accum_steps
                        avg_alignment_loss = accum_alignment_loss / accum_steps
                        avg_cosine_similarity = accum_cosine_similarity / accum_steps
                        avg_train_loss = train_loss / accum_steps

                        # Resetta gli accumulatori
                        accum_total_loss = 0.0
                        accum_diffusion_loss = 0.0
                        accum_alignment_loss = 0.0
                        accum_cosine_similarity = 0.0
                        accum_steps = 0
                        train_loss = 0.0

                        # Log nel progress bar

                        print(f"total_loss: {avg_total_loss}")
                        print(f"diffusion_loss: {avg_diffusion_loss}")
                        print(f"alignment_loss: {avg_alignment_loss}")
                        print(f"cosine_similarity: {avg_cosine_similarity}")
                        print(f"lr: {lr_scheduler.get_last_lr()[0]}")
                        print(f"avg_train_loss: {avg_train_loss}")

                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)

                            unwrapped_unet = unwrap_model(unet, accelerator=accelerator)
                            unet_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(unwrapped_unet)
                            )

                            StableDiffusionPipeline.save_lora_weights(
                                save_directory=save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )

                            # Save the mapper state dict
                            torch.save(mapper.state_dict(), os.path.join(save_path, 'mapper.pt'))

                            print("modello salvatooooooooooo")

                        inference(args, vae, text_encoder, tokenizer, noise_scheduler, clip_processor, clip_model, unet, original_unet, device, mapper, args.text_prompt)


                if global_step >= args.max_train_steps:
                    break

                    
        accelerator.end_training()

    elif(method==2):
        inference(args, vae, text_encoder, tokenizer, noise_scheduler, clip_processor, clip_model, unet, original_unet, device, mapper, args.text_prompt)

    elif(method==3):
        clip_model32, preprocess32 = clip.load("ViT-B/32", device=device)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Regola secondo necessità
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Inizializza il dataset
        datasetM = msrvtt.MSRVTTDataset(
            video_dir='/content/drive/My Drive/msrvtt/TrainValVideo',
            annotation_file='/content/drive/My Drive/msrvtt/train_val_annotation/train_val_videodatainfo.json',
            split='test',
            transform=transform
        )

        # Imposta un seme per la riproducibilità (opzionale)
        random.seed(42)
        
        # Seleziona casualmente 100 indici
        subset_indices = random.sample(range(len(datasetM)), 100)
        
        # Crea il sottoinsieme del dataset
        subset_dataset = Subset(datasetM, subset_indices)
        
        # Esegui la valutazione sul sottoinsieme
        average_gen_similarity = evaluate_msrvtt_clip_similarity(
            clip_model32, preprocess32, subset_dataset, device, args, vae, text_encoder, tokenizer, noise_scheduler, clip_processor, clip_model, unet, original_unet, mapper
        )
        
        #print(f"Average Ground Truth CLIP Similarity (CLIPSIM): {average_gt_similarity:.4f}")
        print(f"Average Generated Video CLIP Similarity (CLIPSIM): {average_gen_similarity:.4f}")
    
    else:
        # Definisci le trasformazioni per i video reali
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Ridimensiona i frame a 224x224
            ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        ])

        # Crea il dataset
        train_dataset = ucf.UCF101Dataset(
            csv_file='test.csv',
            root_dir='/content/drive/My Drive/UCF101',  # Sostituisci con il tuo percorso reale
            transform=transform,
            num_frames=16
        )

        class_names = train_dataset.classes
        num_classes = len(class_names)
        random.seed(42)
        subset_size = 101  # Your desired subset size
        samples_per_class = max(1, subset_size // num_classes)

        subset_indices = []

        for class_name in class_names:
            indices_in_class = train_dataset.class_to_indices[class_name]
            if len(indices_in_class) >= samples_per_class:
                selected_indices = random.sample(indices_in_class, samples_per_class)
            else:
                selected_indices = indices_in_class  # Take all if not enough samples
            subset_indices.extend(selected_indices)

        subset_train_dataset = Subset(train_dataset, subset_indices)

        # Crea il DataLoader per i video reali
        dataloader = DataLoader(subset_train_dataset, batch_size=8, shuffle=False, num_workers=4)

        # Carica il modello I3D pre-addestrato
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Utilizziamo R3D-18 come sostituto per I3D in torchvision
        i3d_model = models_video.r3d_18(pretrained=True)
        # Rimuovi l'ultimo strato per ottenere le feature
        i3d_model = nn.Sequential(*list(i3d_model.children())[:-1])  # Rimuovi l'ultimo strato
        i3d_model = i3d_model.to(device)
        i3d_model.eval()

        # Ottieni i nomi delle classi
        
        print(f"number of classes is: {num_classes}")

        # Inizializza le liste per le feature
        features_gen = []
        features_real = []

        print("Generazione e estrazione delle feature dai video sintetici...")
        for class_name in tqdm(class_names, desc="Generando video"):
            #for _ in range(2):
            indices = get_class_indices_in_subset(subset_train_dataset, class_name)
            idx = random.choice(indices)
            sample = subset_train_dataset[idx]
            one_frame = sample['frame']
            one_frame = one_frame.unsqueeze(0)
            print(f"one_frame shape: {one_frame.shape}, dtype: {one_frame.dtype}") #torch.Size([1, 240, 320, 3]), dtype: torch.uint8

            # Genera un video utilizzando fine_tuned_lavie
            video_tensor = inference(args, vae, text_encoder, tokenizer, noise_scheduler, clip_processor, clip_model, unet, original_unet, device, mapper, class_name, "FVD", one_frame) # [16, 320, 512, 3], uint8

            # Preprocessa il video generato
            video = ucf.preprocess_generated_video(video_tensor)  # [3, 16, 224, 224]

            # Aggiungi una dimensione batch
            video = video.unsqueeze(0)  # [1, 3, 16, 224, 224]

            # Estrai le feature utilizzando I3D
            feat = ucf.extract_i3d_features(video, i3d_model, device)  # [1, feature_dim, 1, 1, 1]
            feat = feat.view(feat.size(0), -1)  # Appiattisci a [1, feature_dim]
            features_gen.append(feat.squeeze(0).numpy())

        # Estrai le feature dai video reali
        print("Estrazione delle feature dai video reali...")
        for batch in tqdm(dataloader, desc="Processando video reali"):
            frames = batch['frames']  # [B, C, T, H, W]
            labels = batch['label']  # [B]

            # Estrai le feature utilizzando I3D
            feat = ucf.extract_i3d_features(frames, i3d_model, device)  # [B, feature_dim, 1, 1, 1]
            feat = feat.view(feat.size(0), -1)  # [B, feature_dim]
            features_real.extend(feat.cpu().numpy())

        # Converti le liste in array numpy
        features_gen = np.array(features_gen)
        features_real = np.array(features_real)

        # Calcola l'FVD
        print("Calcolando l'FVD...")
        fvd_score = ucf.compute_fvd(features_gen, features_real)
        print(f"FVD score: {fvd_score}")



def get_class_indices_in_subset(subset, class_name):
    indices_in_subset = []
    for i, original_idx in enumerate(subset.indices):
        label = subset.dataset.annotations.iloc[original_idx]['label']
        if label == class_name:
            indices_in_subset.append(i)
    return indices_in_subset


def evaluate_msrvtt_clip_similarity(clip_model32, preprocess32, dataset, device, args, vae, text_encoder, tokenizer, noise_scheduler, clip_processor, clip_model, unet, original_unet, mapper):

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=msrvtt.collate_fn)
    
    total_gen_similarity = 0  # For generated videos
    num_videos = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        # Ground Truth Video Frames and Caption
        caption = batch['caption'][0]  # Single caption
        frame = batch['frame'].to(device)

        print(f"frame__ shape: {frame.shape}, dtype: {frame.dtype}") #torch.Size([1, 240, 320, 3]), dtype: torch.uint8
        
        # Generate Video from Caption using Your Model
        with torch.no_grad():
            generated_video_frames = inference(args, vae, text_encoder, tokenizer, noise_scheduler, clip_processor, clip_model, unet, original_unet, device, mapper, caption, "clipsim", frame)
        
        gen_frames = process_video_frames(generated_video_frames)
        
        # Compute CLIP Similarity for Generated Video
        gen_frame_similarities = []
        for frame in gen_frames:
            similarity = get_clip_similarity(clip_model32, preprocess32, caption, frame, device)
            gen_frame_similarities.append(similarity)
        avg_gen_similarity = sum(gen_frame_similarities) / len(gen_frame_similarities)
        total_gen_similarity += avg_gen_similarity
        
        num_videos += 1

    average_gen_similarity = total_gen_similarity / num_videos
    
    return average_gen_similarity


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

        cosine_similarity = F.cosine_similarity(image_features, text_features).item()

        print(f"cosine_similirat is: {cosine_similarity}")
        
        # Compute similarity
        similarity = (image_features @ text_features.T).item()

        print(f"similarity is: {similarity}")

    return similarity

def process_video_frames(generated_video_frames):
    # generated_video_frames ha forma [16, 320, 512, 3]
    gen_frames = []
    for frame in generated_video_frames:
        # frame ha forma [320, 512, 3]
        # Convertiamo direttamente in immagine PIL
        pil_image = Image.fromarray(frame.cpu().numpy())
        gen_frames.append(pil_image)
    return gen_frames



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
    
    lora_model(data, video_folder, OmegaConf.load(args.config), method=1)