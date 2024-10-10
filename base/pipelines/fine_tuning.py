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


def inference(args, vae, text_encoder, tokenizer, noise_scheduler, clip_processor, clip_model, unet, device, mapper, caption):
        
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


def lora_model(data, video_folder, args, caption, training=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models required for both training and inference
    sd_path = os.path.join(args.pretrained_path, "stable-diffusion-v1-4")
    unet = get_models(args, sd_path).to(device, dtype=torch.float16)
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)

    # Instantiate the mapping network
    mapper = MappingNetwork().to(device)
    # Optionally load mapper state dict if available
    mapper.load_state_dict(torch.load('/content/drive/My Drive/checkpoints/mapping_network.pth'))
    

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    if training:
        # Initialize Accelerator and Training Configurations
        logging_dir = Path(args.output_dir, args.logging_dir)

        accelerator_project_config = ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=logging_dir
        )

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

        # Set the training seed
        if args.seed is not None:
            set_seed(args.seed)

        # Set models to train mode and enable gradients
        unet.train()
        mapper.train()
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        for param in mapper.parameters():
            param.requires_grad = True

        # Determine weight data type based on mixed precision
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        unet.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        # Prepare LoRA configuration
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet = get_peft_model(unet, lora_config)

        # Cast trainable parameters to fp32 if using mixed precision
        if args.mixed_precision == "fp16":
            cast_training_params([unet, mapper], dtype=torch.float32)

        # Initialize Dataset and Data Loader
        dataset = VideoDatasetMsvd(annotations_file=data, video_dir=video_folder)
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=custom_collate
        )
        print(f"Numero totale di elementi nel dataloader: {len(train_dataloader)}")

        lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
        trainable_params = list(lora_layers) + list(mapper.parameters())

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs
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
            num_update_steps_per_epoch = math.ceil(
                len_train_dataloader_after_sharding / args.gradient_accumulation_steps
            )
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

        optimizer_mapper = torch.optim.AdamW(mapper.parameters(), lr=1e-4)
        scheduler_mapper = torch.optim.lr_scheduler.StepLR(optimizer_mapper, step_size=10, gamma=0.1)

        criterion = nn.CosineEmbeddingLoss()

        # Prepare everything with our `accelerator`
        (
            unet,
            mapper,
            optimizer,
            train_dataloader,
            lr_scheduler,
            optimizer_mapper,
            scheduler_mapper,
        ) = accelerator.prepare(
            unet,
            mapper,
            optimizer,
            train_dataloader,
            lr_scheduler,
            optimizer_mapper,
            scheduler_mapper,
        )

        # Recalculate training steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
                logger.warning(
                    f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                    f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                    f"This inconsistency may result in the learning rate scheduler not functioning properly."
                )
        # Recalculate number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Initialize trackers
        if accelerator.is_main_process:
            accelerator.init_trackers("text2image-fine-tune", config=vars(args))

        global_step = 0
        first_epoch = 0

        # Load from checkpoint if available
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

        epoch_losses = []

        print(f"first_epoch: {first_epoch}")
        print(f"num_train_epochs: {args.num_train_epochs}")

        # Training Loop
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

                    latents = encode_latents(video, vae)
                    latents = latents * vae.config.scaling_factor

                    # Sample noise
                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], latents.shape[2], 1, 1), device=latents.device
                        )

                    bsz = latents.shape[0]
                    # Sample a random timestep
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device
                    ).long()

                    # Add noise to the latents
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
                    ).last_hidden_state.to(torch.float16)

                    image_inputs = clip_processor(images=list(frame_tensor), return_tensors="pt").pixel_values.to(unet.device)
                    image_features = clip_model.vision_model(
                        pixel_values=image_inputs,
                    ).last_hidden_state.to(torch.float16)

                    # Map image embeddings to text embedding space using the mapping network
                    mapped_image_features = mapper(image_features, text_features)

                    mapped_image_embeddings_flat = mapped_image_features.reshape(-1, 768)
                    text_embeddings_flat = text_features.reshape(-1, 768)

                    # Normalize embeddings
                    mapped_image_embeddings_flat = F.normalize(mapped_image_embeddings_flat, p=2, dim=1)
                    text_embeddings_flat = F.normalize(text_embeddings_flat, p=2, dim=1)

                    target = torch.ones(mapped_image_embeddings_flat.size(0)).to(device)
                    loss_mapper = criterion(mapped_image_embeddings_flat, text_embeddings_flat, target)

                    # Combine features
                    combined_features = torch.cat([text_features, mapped_image_features], dim=1)
                    encoder_hidden_states = combined_features

                    # Get the target for loss
                    if args.prediction_type is not None:
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

                    lambda_alignment = 0.1

                    # Calculate total loss
                    diffusion_loss = loss
                    alignment_loss = loss_mapper
                    total_loss = diffusion_loss + lambda_alignment * alignment_loss

                    cosine_sim = F.cosine_similarity(mapped_image_embeddings_flat, text_embeddings_flat)
                    mean_cosine_sim = cosine_sim.mean().item()

                    # Accumulate metrics
                    accum_cosine_similarity += mean_cosine_sim
                    accum_total_loss += total_loss.detach().item()
                    accum_diffusion_loss += diffusion_loss.detach().item()
                    accum_alignment_loss += alignment_loss.detach().item()
                    accum_steps += 1

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

                # Update progress
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % args.checkpointing_steps == 0:
                        # Calculate average losses
                        avg_total_loss = accum_total_loss / accum_steps
                        avg_diffusion_loss = accum_diffusion_loss / accum_steps
                        avg_alignment_loss = accum_alignment_loss / accum_steps
                        avg_cosine_similarity = accum_cosine_similarity / accum_steps
                        avg_train_loss = train_loss / accum_steps

                        # Reset accumulators
                        accum_total_loss = 0.0
                        accum_diffusion_loss = 0.0
                        accum_alignment_loss = 0.0
                        accum_cosine_similarity = 0.0
                        accum_steps = 0
                        train_loss = 0.0

                        print(f"total_loss: {avg_total_loss}")
                        print(f"diffusion_loss: {avg_diffusion_loss}")
                        print(f"alignment_loss: {avg_alignment_loss}")
                        print(f"cosine_similarity: {avg_cosine_similarity}")
                        print(f"lr: {lr_scheduler.get_last_lr()[0]}")
                        print(f"avg_train_loss: {avg_train_loss}")

                        if accelerator.is_main_process:
                            # Manage checkpoints
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

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

                            print("Model checkpoint saved.")

                        # Optionally, perform inference to monitor progress
                        inference(
                            args,
                            vae,
                            text_encoder,
                            tokenizer,
                            noise_scheduler,
                            clip_processor,
                            clip_model,
                            unet,
                            device,
                            mapper,
                            args.text_prompt
                        )

                if global_step >= args.max_train_steps:
                    break

            # End of epoch

        accelerator.end_training()

    else:
        # Only perform inference
        unet.eval()
        mapper.eval()
        text_encoder.eval()
        vae.eval()
        clip_model.eval()

        return inference(
            args,
            vae,
            text_encoder,
            tokenizer,
            noise_scheduler,
            clip_processor,
            clip_model,
            unet,
            device,
            mapper,
            caption
        )

    

def model(caption):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    # Load configurations from the provided config file
    config = OmegaConf.load(args.config)
    args.__dict__.update(config)

    # Determine if you're on Google Colab
    on_colab = 'COLAB_GPU' in os.environ

    if on_colab:
        dataset_path = '/content/drive/My Drive/msvd'
    else:
        dataset_path = '/path/to/your/Google_Drive/sync/folder/path/to/your/dataset'

    video_folder = os.path.join(dataset_path, 'YouTubeClips')
    data = os.path.join(dataset_path, 'annotations.txt')

    return lora_model(data, video_folder, args, caption, training=False)




if __name__ == "__main__":
    model("a lion playing with a ball")