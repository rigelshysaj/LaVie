import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from transformers import CLIPTextModel
from diffusers import AutoencoderKL
from PIL import Image
import os
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import imageio
import torch.optim as optim
import torch.nn.functional as F



class MappingDataset(Dataset):
    def __init__(self, annotations_file, video_dir, clip_model, clip_processor, sd_tokenizer, sd_text_encoder, target_size=(512, 320)):
        self.video_dir = video_dir
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.sd_tokenizer = sd_tokenizer
        self.sd_text_encoder = sd_text_encoder
        self.target_size = target_size

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

        #print(f"len frames: {len(frames)}")

        
        # Estrarre un frame centrale
        mid_frame = frames[len(frames) // 2]
        mid_frame_np = np.array(mid_frame)

        mid_frame = torch.tensor(mid_frame_np)
        #print(f"mid_frame2 shape: {mid_frame.shape}, dtype: {mid_frame.dtype}") #shape: torch.Size([3, 320, 512]), dtype: torch.uint8
        
        # Ottieni le descrizioni del video
        video_id = os.path.splitext(video_file)[0]
        descriptions = self.video_descriptions.get(video_id, [])

        #print(f"description of __getitem__: {descriptions} video_id: {video_id}")
        

        # Tokenize and encode the caption using Stable Diffusion's tokenizer and text encoder
        text_inputs = self.sd_tokenizer(
            list(descriptions),
            max_length=self.sd_tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.sd_text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            ).last_hidden_state  # Shape: [max_length, 768]
            print(f"text_embeddings11 shape: {text_embeddings.shape}, dtype: {text_embeddings.dtype}")
            text_embeddings = text_embeddings.squeeze(0)
            print(f"text_embeddings22 shape: {text_embeddings.shape}, dtype: {text_embeddings.dtype}")

        # Encode the image using CLIP's image encoder
        image_inputs = self.clip_processor(images=mid_frame, return_tensors="pt")
        with torch.no_grad():
            image_embeddings = self.clip_model.vision_model(
                pixel_values=image_inputs.pixel_values
            ).last_hidden_state  # Shape: [num_patches, 1024]
            print(f"image_embeddings11 shape: {image_embeddings.shape}, dtype: {image_embeddings.dtype}")
            image_embeddings = image_embeddings.squeeze(0)
            print(f"image_embeddings22 shape: {image_embeddings.shape}, dtype: {image_embeddings.dtype}")

        return image_embeddings, text_embeddings
        



class MappingNetwork(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768, hidden_dim=512):
        super(MappingNetwork, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mapping(x)
    

def training(mapping_dataloader):
    # Instantiate the mapping network
    mapping_network = MappingNetwork().to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(mapping_network.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 10  # Adjust as needed

    for epoch in range(num_epochs):
        mapping_network.train()
        epoch_loss = 0
        for batch in mapping_dataloader:
            # Filter out None values
            batch = [item for item in batch if item is not None]
            if not batch:
                continue

            # batch è una lista di tuple (image_embeddings, text_embeddings)
            image_embeddings_batch, text_embeddings_batch = zip(*batch)

            image_embeddings_batch = torch.stack(image_embeddings_batch).to(device)  # Shape: [batch_size, num_patches, 1024]
            text_embeddings_batch = torch.stack(text_embeddings_batch).to(device)    # Shape: [batch_size, max_length, 768]

            # You may need to flatten the sequences
            image_embeddings_batch = image_embeddings_batch.view(-1, 1024)  # [batch_size * num_patches, 1024]
            text_embeddings_batch = text_embeddings_batch.view(-1, 768)     # [batch_size * max_length, 768]

            optimizer.zero_grad()
            mapped_embeddings = mapping_network(image_embeddings_batch)  # Shape: [batch_size * num_patches, 768]

            loss = criterion(mapped_embeddings, text_embeddings_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(mapping_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the trained mapping network
    torch.save(mapping_network.state_dict(), '/content/drive/My Drive/mapping_network.pth')
    


if __name__ == "__main__":

    dataset_path = '/content/drive/My Drive/msvd'
    
    # Percorsi dei file
    video_folder = os.path.join(dataset_path, 'YouTubeClips')
    data = os.path.join(dataset_path, 'annotations.txt')

    # Instantiate the dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    sd_tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    sd_text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device)

    # Instantiate dataset
    mapping_dataset = MappingDataset(
        annotations_file=data,
        video_dir=video_folder,
        clip_model=clip_model,
        clip_processor=clip_processor,
        sd_tokenizer=sd_tokenizer,
        sd_text_encoder=sd_text_encoder,
    )

    # Create DataLoader
    mapping_dataloader = DataLoader(mapping_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=lambda x: x)
    training(mapping_dataloader)
