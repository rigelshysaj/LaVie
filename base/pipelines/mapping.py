import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from transformers import CLIPTextModel
from PIL import Image
import os
import torch.nn as nn
from torchvision import transforms
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


    
class MappingNetwork(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768, num_layers=6, num_heads=8, seq_len_in=257, seq_len_out=77):
        super(MappingNetwork, self).__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.seq_proj = nn.Linear(seq_len_in, seq_len_out)
    
    def forward(self, x):
        # x: [batch_size, seq_len_in, input_dim]
        x = self.input_proj(x)  # [batch_size, seq_len_in, output_dim]
        x = x.permute(1, 0, 2)  # [seq_len_in, batch_size, output_dim]
        x = self.transformer_encoder(x)  # [seq_len_in, batch_size, output_dim]
        x = x.permute(1, 0, 2)  # [batch_size, seq_len_in, output_dim]
        x = self.seq_proj(x.transpose(1, 2)).transpose(1, 2)  # [batch_size, seq_len_out, output_dim]
        return x
    
    
def training_mapping(train_dataloader, val_dataloader, clip_model, clip_processor, tokenizer, text_encoder, device):
    mapping_network = MappingNetwork().to(device)

    criterion = nn.CosineEmbeddingLoss()
    #optimizer = optim.Adam(mapping_network.parameters(), lr=1e-4)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    optimizer = optim.AdamW(mapping_network.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 20  # Puoi regolare secondo necessità
    patience = 5  # Numero di epoche da attendere prima di fermare l'addestramento
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        mapping_network.train()
        epoch_loss = 0.0
        epoch_cosine_sim = 0.0  # Per monitorare la similarità in questo epoch

        # Ciclo di addestramento
        for images, descriptions in train_dataloader:
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

                image_embeddings = clip_model.vision_model(
                    pixel_values=image_inputs,
                ).last_hidden_state

            #print(f"image_embeddings shape: {image_embeddings.shape}, dtype: {image_embeddings.dtype}")
            # Mappa le embedding delle immagini
            mapped_image_embeddings = mapping_network(image_embeddings)  # [batch_size, 257, 768]
            #print(f"mapped_image_embeddings shape: {mapped_image_embeddings.shape}, dtype: {mapped_image_embeddings.dtype}")
            
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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapping_network.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        
        # Calcolo della loss e della similarità media di training
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        avg_epoch_cosine_sim = epoch_cosine_sim / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_epoch_loss:.4f},'
              f' Training Mean Cosine Similarity: {avg_epoch_cosine_sim:.4f}')

        # Fase di validazione
        mapping_network.eval()
        val_loss = 0.0
        val_cosine_sim = 0.0
        with torch.no_grad():
            for images, descriptions in val_dataloader:
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

                text_embeddings = text_encoder(
                    input_ids=text_inputs.input_ids,
                ).last_hidden_state

                image_embeddings = clip_model.vision_model(
                    pixel_values=image_inputs,
                ).last_hidden_state

                # Mappa le embedding delle immagini
                mapped_image_embeddings = mapping_network(image_embeddings)

                mapped_image_embeddings_pooled = mapped_image_embeddings.mean(dim=1)
                text_embeddings_pooled = text_embeddings.mean(dim=1)

                # Normalizzazione
                mapped_image_embeddings_pooled = F.normalize(mapped_image_embeddings_pooled, dim=-1)
                text_embeddings_pooled = F.normalize(text_embeddings_pooled, dim=-1)

                # Calcolo della loss
                target = torch.ones(text_embeddings_pooled.size(0)).to(device)
                loss = criterion(mapped_image_embeddings_pooled, text_embeddings_pooled, target)

                cosine_sim = F.cosine_similarity(text_embeddings_pooled, mapped_image_embeddings_pooled)
                mean_cosine_sim = cosine_sim.mean().item()
                val_cosine_sim += mean_cosine_sim

                val_loss += loss.item()

        # Calcolo della loss e della similarità media di validazione
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_cosine_sim = val_cosine_sim / len(val_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f},'
              f' Validation Mean Cosine Similarity: {avg_val_cosine_sim:.4f}')

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Salva il miglior modello
            torch.save(mapping_network.state_dict(), '/content/drive/My Drive/checkpoints/mapping_network_best.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping on epoch {epoch+1}')
                break



def custom_collate(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] != "", batch))
    images, descriptions = zip(*batch)
    return images, descriptions


if __name__ == "__main__":
    dataset_path = '/content/drive/My Drive/flickr'
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

    # Definisci le trasformazioni
    transform = transforms.Compose([
        transforms.Resize((320, 512)),
        transforms.ToTensor(),
    ])

    # Crea il dataset completo
    full_dataset = MappingDataset(
        annotations_file=annotations_file,
        image_dir=image_folder,
        transform=transform,
    )

    # Dividi il dataset
    from torch.utils.data import random_split

    train_ratio = 0.8
    val_ratio = 0.2
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Crea DataLoader per training e validazione
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=custom_collate
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=custom_collate
    )

    # Inizia l'addestramento
    training_mapping(train_dataloader, val_dataloader, clip_model, clip_processor, tokenizer, text_encoder, device)
