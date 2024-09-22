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
import random


class MappingDataset(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Leggi caption.txt e memorizza le descrizioni in un dizionario
        self.image_descriptions = {}
        with open(annotations_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        image_filename, caption = parts
                        if image_filename in self.image_descriptions:
                            self.image_descriptions[image_filename].append(caption)
                        else:
                            self.image_descriptions[image_filename] = [caption]

        # Ottieni la lista dei file immagine nella cartella Images
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)

        # Carica l'immagine utilizzando PIL
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Ottieni le descrizioni dell'immagine
        descriptions = self.image_descriptions.get(image_file, [])
        if descriptions:
            # Seleziona una didascalia casualmente
            description = random.choice(descriptions)
        else:
            description = ""

        return image, description


class CrossAttentionNetwork(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super(CrossAttentionNetwork, self).__init__()
        self.attention_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, text_features, mapped_image_features):
        # text_features: [batch_size, seq_len_text, embed_dim]
        # mapped_image_features: [batch_size, seq_len_img, embed_dim]

        # Transpose for multihead attention
        text_features_t = text_features.transpose(0, 1)  # Shape: [seq_len_text, batch_size, embed_dim]
        mapped_image_features_t = mapped_image_features.transpose(0, 1)  # Shape: [seq_len_img, batch_size, embed_dim]

        # Apply cross-attention
        attention_output, attention_weights = self.attention_layer(
            query=text_features_t,
            key=mapped_image_features_t,
            value=mapped_image_features_t
        )

        # Transpose back
        attention_output = attention_output.transpose(0, 1)  # Shape: [batch_size, seq_len_text, embed_dim]

        # Apply feedforward network
        output = self.feedforward(attention_output)  # Shape: [batch_size, seq_len_text, embed_dim]

        return output, attention_weights
    

def training_cross_attention(mapping_dataloader, mapping_network, clip_model, clip_processor, sd_tokenizer, sd_text_encoder, device):
    
    # mapping_network is pre-trained, set to eval mode
    mapping_network.eval()
    
    cross_attention_network = CrossAttentionNetwork(embed_dim=768, num_heads=8).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cross_attention_network.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    num_epochs = 10  # Adjust as needed
    
    for epoch in range(num_epochs):
        cross_attention_network.train()
        epoch_loss = 0.0
        for images, descriptions in mapping_dataloader:
            if not images or not descriptions:
                continue

            # Preprocess images
            image_inputs = clip_processor(images=list(images), return_tensors="pt").pixel_values.to(device)

            # Tokenize and encode descriptions
            text_inputs = sd_tokenizer(
                list(descriptions),
                max_length=sd_tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)


            text_input_ids = text_inputs.input_ids

            if hasattr(sd_text_encoder.config, "use_attention_mask") and sd_text_encoder.config.use_attention_mask:
                print("usa attention mask")
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            with torch.no_grad():
                text_embeddings = sd_text_encoder(
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                )
                text_embeddings = text_embeddings[0]
                text_embeddings.to(dtype=sd_text_encoder.dtype, device=device)

                print(f"text_embeddings shape: {text_embeddings.shape}, dtype: {text_embeddings.dtype}")

                image_embeddings = clip_model.vision_model(
                    pixel_values=image_inputs
                )
                image_embeddings = image_embeddings[0]

                mapped_image_embeddings = mapping_network(image_embeddings)

                print(f"image_embeddings shape: {image_embeddings.shape}, dtype: {image_embeddings.dtype}")

            # Forward pass through the cross-attention network
            output, attention_weights = cross_attention_network(text_embeddings, mapped_image_embeddings)  # [batch_size, seq_len_text, 768]

            # Compute loss between output and text_embeddings
            loss = criterion(output, text_embeddings)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cross_attention_network.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()

        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(mapping_dataloader):.4f}')
        
        # Optionally save the model
        torch.save(cross_attention_network.state_dict(), '/content/drive/My Drive/checkpoints/cross_attention_network.pth')

    
class MappingNetwork_(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768, hidden_dim=512):
        super(MappingNetwork_, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mapping(x)
    
class MappingNetwork(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768, hidden_dims=[512, 256, 256]):
        super(MappingNetwork, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.1))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [batch_size, num_patches, 1024]
        batch_size, num_patches, _ = x.size()
        x = x.view(batch_size * num_patches, -1)  # [batch_size * num_patches, 1024]
        x = self.mapping(x)  # [batch_size * num_patches, 768]
        x = x.view(batch_size, num_patches, -1)  # [batch_size, num_patches, 768]
        return x
    
    
def training_mapping(mapping_dataloader, clip_model, clip_processor, sd_tokenizer, sd_text_encoder, device):
    
    mapping_network = MappingNetwork().to(device)

    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(mapping_network.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    num_epochs = 15  # Regola secondo necessità

    for epoch in range(num_epochs):
        mapping_network.train()
        epoch_loss = 0.0
        epoch_cosine_sim = 0.0  # Per monitorare la similarità in questo epoch
        for images, descriptions in mapping_dataloader:
            if not images or not descriptions:
                continue

            # Preprocessa le immagini
            image_inputs = clip_processor(images=list(images), return_tensors="pt").pixel_values.to(device)

            # Tokenizza e codifica le descrizioni
            text_inputs = sd_tokenizer(
                list(descriptions),
                max_length=sd_tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            text_input_ids = text_inputs.input_ids

            if hasattr(sd_text_encoder.config, "use_attention_mask") and sd_text_encoder.config.use_attention_mask:
                print("usa attention mask")
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            #print(f"attention mask: {attention_mask}")


            with torch.no_grad():
                text_embeddings = sd_text_encoder(
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                )
                text_embeddings = text_embeddings[0]
                text_embeddings.to(dtype=sd_text_encoder.dtype, device=device)

                #print(f"text_embeddings shape: {text_embeddings.shape}, dtype: {text_embeddings.dtype}")

                image_embeddings = clip_model.vision_model(
                    pixel_values=image_inputs,
                )
                image_embeddings = image_embeddings.last_hidden_state

                #print(f"image_embeddings shape: {image_embeddings.shape}, dtype: {image_embeddings.dtype}")

            # Mappa le embedding delle immagini
            mapped_image_embeddings = mapping_network(image_embeddings)  # [batch_size, 257, 768]

            #print(f"mapped_image_embeddings shape: {mapped_image_embeddings.shape}, dtype: {mapped_image_embeddings.dtype}")

            # Aggrega le embedding per campione (es. media)
            mapped_image_embeddings_pooled = mapped_image_embeddings.mean(dim=1)  # [batch_size, 768]
            text_embeddings_pooled = text_embeddings.mean(dim=1)  
            
            #print(f"mapped_image_embeddings_pooled shape: {mapped_image_embeddings_pooled.shape}, dtype: {mapped_image_embeddings_pooled.dtype}")
            #print(f"text_embeddings_pooled shape: {text_embeddings_pooled.shape}, dtype: {text_embeddings_pooled.dtype}")

            # Normalizzazione
            mapped_image_embeddings_pooled = F.normalize(mapped_image_embeddings_pooled, dim=-1)
            text_embeddings_pooled = F.normalize(text_embeddings_pooled, dim=-1)

            # Calcolo della loss
            target = torch.ones(text_embeddings_pooled.size(0)).to(device)
            loss = criterion(mapped_image_embeddings_pooled, text_embeddings_pooled, target)

            cosine_sim = F.cosine_similarity(text_embeddings_pooled, mapped_image_embeddings_pooled)
            mean_cosine_sim = cosine_sim.mean().item()
            epoch_cosine_sim += mean_cosine_sim

            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapping_network.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()

        scheduler.step()
        
        # Calcolo della similarità media per l'epoch
        avg_epoch_cosine_sim = epoch_cosine_sim / len(mapping_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(mapping_dataloader):.4f},'
          f' Mean Cosine Similarity: {avg_epoch_cosine_sim:.4f}')
        
        if(epoch >= 5):
            torch.save(mapping_network.state_dict(), '/content/drive/My Drive/checkpoints/mapping_network.pth')


def custom_collate(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] != "", batch))
    images, descriptions = zip(*batch)
    return images, descriptions


if __name__ == "__main__":
    dataset_path = '/content/drive/My Drive/flickr'
    train_cross_attention = False
    # Percorsi dei file
    image_folder = os.path.join(dataset_path, 'Images')
    annotations_file = os.path.join(dataset_path, 'captions.txt')

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

    transform = transforms.Compose([
        transforms.Resize((320, 512)),
        transforms.ToTensor(),
    ])

    # Passa il transform al dataset
    mapping_dataset = MappingDataset(
        annotations_file=annotations_file,
        image_dir=image_folder,
        transform=transform,
    )

    # Crea il DataLoader con num_workers=0
    mapping_dataloader = DataLoader(
        mapping_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=custom_collate
    )

    if(train_cross_attention):
        # Load pre-trained mapping network
        mapping_network = MappingNetwork().to(device)
        mapping_network.load_state_dict(torch.load('/content/drive/My Drive/checkpoints/mapping_network.pth', map_location=device))
        mapping_network.eval()

        training_cross_attention(mapping_dataloader, mapping_network, clip_model, clip_processor, sd_tokenizer, sd_text_encoder, device)
    else:
        training_mapping(mapping_dataloader, clip_model, clip_processor, sd_tokenizer, sd_text_encoder, device)


