from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import cv2
import os
import json
from tqdm import tqdm
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
import torchvision.models as models
import torch.nn.functional as F
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
import einops
from diffusers import AutoencoderKL, DiffusionPipeline, StableDiffusionPipeline
from torch.utils.checkpoint import checkpoint
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler
from transformers import get_cosine_schedule_with_warmup
from dataclasses import dataclass
from peft import PeftModel, LoraConfig
from PIL import Image
from torchvision import transforms
import plotly.graph_objects as go
from inference import VideoGenPipeline
from arguments import Details
from msvd import VideoDatasetMsvd

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

    weight_dtype = torch.float32

    sd_path = args.pretrained_path + "/stable-diffusion-v1-4"
    unet = get_models(args, sd_path).to(device)
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)

    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(device)
    tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(device) # huge
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    scheduler = DDPMScheduler.from_pretrained(sd_path,
                                            subfolder="scheduler",
                                            beta_start=args.beta_start,
                                            beta_end=args.beta_end,
                                            beta_schedule=args.beta_schedule)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)

    # Applica LoRA al modello
    unet = get_peft_model(unet, lora_config)

    unet = accelerator.prepare(unet)
    
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    accelerator.print(f"Inference from checkpoint {path}")
    accelerator.load_state(os.path.join(args.output_dir, path))

    image_path = "/content/drive/My Drive/shoot.jpeg"

    image = Image.open(image_path)

    # Definisci la trasformazione
    transform = transforms.Compose([
        transforms.Resize((512, 320)),
        transforms.ToTensor()  # Converte l'immagine in un tensore
    ])

    # Applica la trasformazione all'immagine
    input_image = transform(image)

    image_tensor = input_image.unsqueeze(0).to(torch.float32)  # Aggiunge una dimensione per il batch

    print(f"image_tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")

    videogen_pipeline = VideoGenPipeline(vae=vae, 
                                text_encoder=text_encoder_one, 
                                tokenizer=tokenizer_one, 
                                scheduler=scheduler, 
                                unet=unet,
                                clip_processor=clip_processor,
                                clip_model=clip_model
                                ).to(device)
    videogen_pipeline.enable_xformers_memory_efficient_attention()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    video_grids = []
    for prompt in args.text_prompt:
        print('Processing the ({}) prompt'.format(prompt))
        videos = videogen_pipeline(prompt,
                                image_tensor=image_tensor, 
                                video_length=args.video_length, 
                                height=args.image_size[0], 
                                width=args.image_size[1], 
                                num_inference_steps=args.num_sampling_steps,
                                guidance_scale=args.guidance_scale).video
        imageio.mimwrite("/content/drive/My Drive/" + prompt.replace(' ', '_') + '.mp4', videos[0], fps=8, quality=9) # highest quality is 10, lowest is 0
    
    print('save path {}'.format("/content/drive/My Drive/" + prompt.replace(' ', '_') + '.mp4'))
    


def encode_latents(video, vae):
    vae = vae.to(torch.float16)
    video = video.to(torch.float16)
    b, c, f, h, w = video.shape
    video = einops.rearrange(video, "b c f h w -> (b f) c h w")
    
    latents = vae.encode(video).latent_dist.sample()
    latents = einops.rearrange(latents, "(b f) c h w -> b c f h w", b=b)
    
    return latents

def encode_latents_(video, vae):
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


def train_lora_model(data, video_folder, args):

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
    original_unet = get_models(args, sd_path).to(device, dtype=torch.float32)
    original_unet.load_state_dict(state_dict)

    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(device)
    # Load CLIP model and processor for image conditioning
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
        cast_training_params(unet, dtype=torch.float32)

    #dataset = VideoDatasetMsrvtt(data, video_folder)
    dataset = VideoDatasetMsvd(annotations_file=data, video_dir=video_folder)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
    print(f"Numero totale di elementi nel dataloader: {len(train_dataloader)}")

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

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
        lora_layers,
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

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
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

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
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
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    unet.enable_xformers_memory_efficient_attention()

    epoch_losses = []

    frame = None
    desc = ""

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        batch_losses = []
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                video, description, frame_tensor = batch

                #print(f"train_lora_model video shape: {video.shape}, dtype: {video.dtype}") #[1, 3, 16, 320, 512] torch.float32
                
                #print(f"frame_tensor shape: {frame_tensor.shape}, dtype: {frame_tensor.dtype}") #frame_tensor shape: torch.Size([1, 3, 320, 512]), dtype: torch.float32

                #print(f"description: {description[0]}")

                
                try:
                    #if(description[0] == 'a man cutting photo with a sword' and frame is None):
                        #print("yesssssssssss a man cutting photo with a sword")
                    frame = frame_tensor
                    desc = description[0]
                except Exception as e:
                    print("------------------START--------------------")
                    print(description)
                    print("------------------END--------------------")
                    continue
                

                print(f"epoca {epoch}, iterazione {step}, global_step {global_step}")

                latents = encode_latents(video, vae)
                #print(f"train_lora_model latents1 shape: {latents.shape}, dtype: {latents.dtype}") #shape: torch.Size([1, 4, 16, 40, 64]), dtype: torch.float32


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

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                #print(f"train_lora_model noisy_latents shape: {noisy_latents.shape}, dtype: {noisy_latents.dtype}") #shape: torch.Size([1, 4, 16, 40, 64]), dtype: torch.float32

                #print(f"description: {list(description)}")
                # Get the text embedding for conditioning

                text_inputs = tokenizer(
                    list(description),
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(unet.device)

                text_features = text_encoder(
                    text_inputs.input_ids,
                    return_dict=False
                )[0]


                #print(f"train_lora_model text_features shape: {text_features.shape}, dtype: {text_features.dtype}") #[1, 10, 768] torch.float16


                image_inputs = clip_processor(images=frame_tensor, return_tensors="pt").pixel_values.to(unet.device)
                outputs = clip_model.vision_model(image_inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1].to(torch.float16)
                #print(f"train_lora_model last_hidden_state shape: {last_hidden_state.shape}, dtype: {last_hidden_state.dtype}") #[1, 50, 768] torch.float16
                
                combination_tensor = torch.cat([text_features, last_hidden_state], dim=1)
                #print(f"train_lora_model combination_tensor shape: {combination_tensor.shape}, dtype: {combination_tensor.dtype}") #[10, 1, 768] torch.float16

                encoder_hidden_states = combination_tensor

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

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                batch_losses.append(train_loss)

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                #print(f"Step {global_step}: train_loss = {train_loss:.6f}")
                train_loss = 0.0

                if global_step % args.logging_steps == 0:
                    log_lora_weights(unet, global_step)

                if global_step % args.checkpointing_steps == 0:
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

                        print("modello salvatooooooooooo")

                        logger.info(f"Saved state to {save_path}")
                

                    with torch.no_grad():

                        print("------------------START DESC--------------------")
                        print(desc)
                        print("------------------END DESC--------------------")

                        # Funzione per generare video
                        def generate_video(unet, is_original):
                            pipeline = VideoGenPipeline(
                                vae=vae, 
                                text_encoder=text_encoder, 
                                tokenizer=tokenizer, 
                                scheduler=noise_scheduler, 
                                unet=unet,
                                clip_processor=clip_processor,
                                clip_model=clip_model
                            ).to(device)
                            pipeline.enable_xformers_memory_efficient_attention()


                            print(f'Processing the ({desc}) prompt for {"original" if is_original else "fine-tuned"} model')
                            videos = pipeline(
                                desc,
                                image_tensor=frame if not is_original else None,
                                video_length=args.video_length, 
                                height=args.image_size[0], 
                                width=args.image_size[1], 
                                num_inference_steps=args.num_sampling_steps,
                                guidance_scale=args.guidance_scale
                            ).video
                            
                            suffix = "original" if is_original else "fine_tuned"
                            imageio.mimwrite(f"/content/drive/My Drive/{suffix}_sample_epoch_{epoch}.mp4", videos[0], fps=8, quality=9)
                            del videos
                            
                            del pipeline
                            torch.cuda.empty_cache()

                        # Genera video con il modello fine-tuned
                        generate_video(unet, is_original=False)

                        generate_video(original_unet, is_original=True)

                        #del original_unet #Poi quando fa l'inference la seconda volta non trova più unet e dice referenced before assignment
                        torch.cuda.empty_cache()

                        print('save path {}'.format("/content/drive/My Drive/"))

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


        avg_epoch_loss = sum(batch_losses) / len(batch_losses)
        #print(f"Epoch {epoch}/{args.num_train_epochs} completed with average loss: {avg_epoch_loss}")
        epoch_losses.append(avg_epoch_loss)      

                
    
    num_epochs = len(epoch_losses)
    epochs = list(range(1, num_epochs + 1))  # Crea una lista [1, 2, 3, ..., num_epochs]
    print(f"num_epochs: {num_epochs}")

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


def training(args):
    
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
    
    train_lora_model(data, video_folder, args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    training(OmegaConf.load(args.config))
    #load_model_for_inference(OmegaConf.load(args.config))