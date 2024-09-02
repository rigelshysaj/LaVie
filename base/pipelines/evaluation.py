import json
import random
import torch
import clip
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import os
from typing import List, Union
from omegaconf import OmegaConf
import torch
import cv2
import os
import json
from tqdm import tqdm
import shutil
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
import argparse
import imageio
import transformers
import diffusers
import sys
from pathlib import Path
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
sys.path.append(os.path.split(sys.path[0])[0])
from models import get_models
from download import find_model
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np
from diffusers import AutoencoderKL, StableDiffusionPipeline
from diffusers.schedulers import DDPMScheduler
from peft import LoraConfig
from PIL import Image
from inference import VideoGenPipeline

# Carica il modello CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Carica il modello Lavie fine-tuned
model_path = "path/to/your/finetuned/lavie/model"
pipeline = StableDiffusionPipeline.from_pretrained(model_path).to(device)

# Funzione per caricare i dati MSR-VTT
def load_msr_vtt_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Funzione per estrarre frames da un video
def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames

# Funzione per calcolare la similarità CLIP
def compute_clip_similarity(text, images):
    with torch.no_grad():
        text_features = clip_model.encode_text(clip.tokenize(text).to(device))
        image_features = clip_model.encode_image(torch.stack([clip_preprocess(Image.fromarray(img)) for img in images]).to(device))
        
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).mean()
    
    return similarity.item()

def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)

def inference(args, vae, text_encoder, tokenizer, noise_scheduler, clip_processor, clip_model, unet, original_unet, device, text, image_condition):
    with torch.no_grad():

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

        
        print(f'Processing the ({text}) prompt')
        videos = pipeline(
            text,
            image_tensor=image_condition,
            video_length=args.video_length, 
            height=args.image_size[0], 
            width=args.image_size[1], 
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.guidance_scale
        ).video

        imageio.mimwrite(f"/content/drive/My Drive/fine_tuned.mp4", videos[0], fps=8, quality=9)

        del videos
        del pipeline
        torch.cuda.empty_cache()

        print('save path {}'.format("/content/drive/My Drive/"))


# Funzione per generare video con Lavie usando text e image condition
def generate_video(args, text, image_condition):
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


    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Prepare everything with our `accelerator`.
    unet = accelerator.prepare(
        unet
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

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
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            

    unet.enable_xformers_memory_efficient_attention()

   
    inference(args, vae, text_encoder, tokenizer, noise_scheduler, clip_processor, clip_model, unet, original_unet, device, text, image_condition)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    # Carica i dati MSR-VTT
    msr_vtt_data = load_msr_vtt_data("path/to/train_val_videodatainfo.json")

    # Filtra i video del test set
    test_videos = [video for video in msr_vtt_data['videos']]

    # Seleziona casualmente 50 video per la valutazione
    random.seed(42)  # Per riproducibilità
    selected_videos = random.sample(test_videos, 50)

    # Dizionario per mappare video_id alle caption
    video_captions = {sentence['video_id']: sentence['caption'] for sentence in msr_vtt_data['sentences']}

    # Valutazione
    clip_similarities = []

    for video in tqdm(selected_videos):
        video_path = f"path/to/TrainValVideo/{video['video_id']}.mp4"
        
        # Estrai i frame dal video
        frames = extract_frames(video_path)
        print(f"frames shape: {frames.shape}, dtype: {frames.dtype}")
        
        # Seleziona casualmente una caption per il video
        caption = random.choice([sentence['caption'] for sentence in msr_vtt_data['sentences'] if sentence['video_id'] == video['video_id']])
        
        # Usa il primo frame come image condition
        image_condition = frames[0]
        print(f"image_condition shape: {image_condition.shape}, dtype: {image_condition.dtype}")
        
        # Genera un nuovo video con Lavie
        generated_frames = generate_video(OmegaConf.load(args.config), caption, image_condition)
        
        # Calcola la similarità CLIP per il video originale
        original_similarity = compute_clip_similarity(caption, frames)
        
        # Calcola la similarità CLIP per il video generato
        generated_similarity = compute_clip_similarity(caption, generated_frames)
        
        clip_similarities.append({
            'video_id': video['video_id'],
            'caption': caption,
            'original_similarity': original_similarity,
            'generated_similarity': generated_similarity
        })

    # Calcola le metriche finali
    avg_original_similarity = np.mean([s['original_similarity'] for s in clip_similarities])
    avg_generated_similarity = np.mean([s['generated_similarity'] for s in clip_similarities])

    print(f"Average CLIP similarity for original videos: {avg_original_similarity:.4f}")
    print(f"Average CLIP similarity for generated videos: {avg_generated_similarity:.4f}")

    # Salva i risultati
    with open('msr_vtt_evaluation_results.json', 'w') as f:
        json.dump({
            'clip_similarities': clip_similarities,
            'avg_original_similarity': avg_original_similarity,
            'avg_generated_similarity': avg_generated_similarity
        }, f, indent=2)