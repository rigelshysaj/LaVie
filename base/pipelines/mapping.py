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
    
    
def training_mapping(mapping_dataloader, clip_model, clip_processor, tokenizer, text_encoder, device):
    
    mapping_network = MappingNetwork().to(device)

    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(mapping_network.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    num_epochs = 10  # Regola secondo necessità

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
            text_inputs = tokenizer(
                list(descriptions),
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                text_embeddings = text_encoder(
                    input_ids=text_inputs.input_ids,
                ).last_hidden_state

                #print(f"text_embeddings shape: {text_embeddings.shape}, dtype: {text_embeddings.dtype}")

                image_embeddings = clip_model.vision_model(
                    pixel_values=image_inputs,
                ).last_hidden_state

                #print(f"image_embeddings shape: {image_embeddings.shape}, dtype: {image_embeddings.dtype}")

            '''
            # Verifica dei token speciali nel tokenizer
            special_tokens = tokenizer.special_tokens_map
            print("Token speciali del tokenizer:")
            print(special_tokens)

            # Accedi al numero di patch dal modello di visione
            num_patches = clip_model.vision_model.embeddings.num_patches
            print(f"Numero di patch: {num_patches}")

            # Ottieni la dimensione dell'immagine e delle patch
            image_size = clip_model.vision_model.config.image_size  # Dimensione dell'immagine (es. 224)
            patch_size = clip_model.vision_model.config.patch_size  # Dimensione della patch (es. 14 o 16)

            # Calcola il numero di patch per lato
            num_patches_per_side = image_size // patch_size

            # Calcola il numero totale di patch
            num_patches = num_patches_per_side ** 2

            print(f"Numero di patch calcolato: {num_patches}")
            '''


            # Mappa le embedding delle immagini
            mapped_image_embeddings = mapping_network(image_embeddings)  # [batch_size, 257, 768]

            #print(f"mapped_image_embeddings shape: {mapped_image_embeddings.shape}, dtype: {mapped_image_embeddings.dtype}")

            # Use the [CLS] token embeddings
            mapped_image_embeddings_pooled = mapped_image_embeddings[:, 0, :]  # [batch_size, 768]
            text_embeddings_pooled = text_embeddings[:, 0, :]
            
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

            print(f'Epoch {epoch+1}/{num_epochs},'
                  f'Loss: {loss.item():.4f}, Mean Cosine Similarity: {mean_cosine_sim:.4f}')
            
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
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    # Imposta i modelli in modalità eval
    clip_model.eval()
    text_encoder.eval()

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

    
    training_mapping(mapping_dataloader, clip_model, clip_processor, tokenizer, text_encoder, device)


