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
    def __init__(self, annotations_file, video_dir, target_size=(512, 320)):
        self.video_dir = video_dir
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

        # Estrarre un frame centrale
        mid_frame = frames[len(frames) // 2]
        mid_frame_np = np.array(mid_frame)

        # Converti il frame in formato PIL per il preprocessamento
        mid_frame_pil = Image.fromarray(cv2.cvtColor(mid_frame_np, cv2.COLOR_BGR2RGB))

        # Ottieni le descrizioni del video
        video_id = os.path.splitext(video_file)[0]
        description = self.video_descriptions.get(video_id, "")

        return mid_frame_pil, description
    


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
    
    
def training(mapping_dataloader, clip_model, clip_processor, sd_tokenizer, sd_text_encoder, device):
    # Instanzia la rete di mapping
    mapping_network = MappingNetwork().to(device)

    # Definisci l'ottimizzatore e la funzione di perdita
    optimizer = optim.Adam(mapping_network.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 10  # Regola secondo necessità

    for epoch in range(num_epochs):
        mapping_network.train()
        epoch_loss = 0
        for batch in mapping_dataloader:
            # batch è una lista di tuple (mid_frame_pil, description)
            images, descriptions = zip(*batch)

            # Preprocessa le immagini
            image_inputs = clip_processor(images=list(images), return_tensors="pt").pixel_values.to(device)

            # Tokenizza e codifica le descrizioni
            text_inputs = sd_tokenizer(
                list(descriptions),
                max_length=sd_tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                text_embeddings = sd_text_encoder(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask,
                ).last_hidden_state  # Shape: [batch_size, max_length, 768]

                print(f"text_embeddings shape: {text_embeddings.shape}, dtype: {text_embeddings.dtype}")

                image_embeddings = clip_model.vision_model(
                    pixel_values=image_inputs
                ).last_hidden_state  # Shape: [batch_size, num_patches, 1024]

                print(f"image_embeddings shape: {image_embeddings.shape}, dtype: {image_embeddings.dtype}")

            # Mappa le embedding delle immagini
            mapped_image_embeddings = mapping_network(image_embeddings)  # [batch_size, 257, 768]

            print(f"mapped_image_embeddings shape: {mapped_image_embeddings.shape}, dtype: {mapped_image_embeddings.dtype}")

            # Aggrega le embedding per campione (es. media)
            mapped_image_embeddings_pooled = mapped_image_embeddings.mean(dim=1)  # [batch_size, 768]
            text_embeddings_pooled = text_embeddings.mean(dim=1)  
            
            print(f"mapped_image_embeddings_pooled shape: {mapped_image_embeddings_pooled.shape}, dtype: {mapped_image_embeddings_pooled.dtype}")
            print(f"text_embeddings_pooled shape: {text_embeddings_pooled.shape}, dtype: {text_embeddings_pooled.dtype}")

            # Calcola la perdita
            loss = criterion(mapped_image_embeddings_pooled, text_embeddings_pooled)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(mapping_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Salva la rete di mapping addestrata
    torch.save(mapping_network.state_dict(), '/content/drive/My Drive/mapping_network.pth')


if __name__ == "__main__":
    dataset_path = '/content/drive/My Drive/msvd'

    # Percorsi dei file
    video_folder = os.path.join(dataset_path, 'YouTubeClips')
    data = os.path.join(dataset_path, 'annotations.txt')

    # Imposta il dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carica i modelli
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    sd_tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    sd_text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device)

    # Imposta i modelli in modalità eval
    clip_model.eval()
    sd_text_encoder.eval()

    # Instanzia il dataset
    mapping_dataset = MappingDataset(
        annotations_file=data,
        video_dir=video_folder,
    )

    # Crea il DataLoader con num_workers=0
    mapping_dataloader = DataLoader(
        mapping_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Impostato a 0 per evitare problemi con CUDA nei worker
        collate_fn=lambda x: x
    )

    # Avvia il training
    training(mapping_dataloader, clip_model, clip_processor, sd_tokenizer, sd_text_encoder, device)
