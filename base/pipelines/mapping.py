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
    def __init__(self, input_dim=1024, output_dim=768, num_layers=12, num_heads=12, seq_len_in=257, seq_len_out=77):
        super(MappingNetwork, self).__init__()

        # Project dimensions
        self.image_proj = nn.Linear(input_dim, output_dim)
        self.text_proj = nn.Linear(output_dim, output_dim)

        # Positional embeddings
        self.image_pos_embedding = nn.Parameter(torch.randn(1, seq_len_in, output_dim))
        self.text_pos_embedding = nn.Parameter(torch.randn(1, seq_len_out, output_dim))

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=output_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, image_embeds, text_embeds):
        # image_embeds: [batch_size, seq_len_in, input_dim]
        # text_embeds: [batch_size, seq_len_out, output_dim]

        image_embeds = image_embeds.to(torch.float32)
        text_embeds = text_embeds.to(torch.float32)

        # Project image embeddings
        image_embeds = self.image_proj(image_embeds) + self.image_pos_embedding  # [batch_size, seq_len_in, output_dim]

        # Add positional embeddings to text embeddings
        text_embeds = text_embeds + self.text_pos_embedding

        # Transpose for Transformer [seq_len, batch_size, output_dim]
        image_embeds = image_embeds.permute(1, 0, 2)
        text_embeds = text_embeds.permute(1, 0, 2)

        # Apply Transformer Decoder
        output = self.transformer_decoder(tgt=text_embeds, memory=image_embeds)  # [seq_len_out, batch_size, output_dim]

        # Transpose back
        output = output.permute(1, 0, 2)  # [batch_size, seq_len_out, output_dim]

        return output

    
    
def training_mapping(train_dataloader, val_dataloader, clip_model, clip_processor, tokenizer, text_encoder, device):
    mapping_network = MappingNetwork(
        input_dim=1024,
        output_dim=768,
        num_layers=12,  # Aumentato
        num_heads=12,   # Aumentato
        seq_len_in=257,
        seq_len_out=77
    ).to(device)

    criterion = nn.CosineEmbeddingLoss()
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
                # Get text embeddings
                text_embeddings = text_encoder(
                    input_ids=text_inputs.input_ids,
                ).last_hidden_state  # [batch_size, 77, 768]

                # Get image embeddings
                image_outputs = clip_model.vision_model(
                    pixel_values=image_inputs,
                )
                image_embeddings = image_outputs.last_hidden_state

            # Map image embeddings to text embedding space
            mapped_image_embeddings = mapping_network(image_embeddings, text_embeddings)

            # Compute loss
            # Reshape to [batch_size * seq_len, embedding_dim]
            mapped_image_embeddings_flat = mapped_image_embeddings.reshape(-1, 768)
            text_embeddings_flat = text_embeddings.reshape(-1, 768)

            # Normalize embeddings
            mapped_image_embeddings_flat = F.normalize(mapped_image_embeddings_flat, p=2, dim=1)
            text_embeddings_flat = F.normalize(text_embeddings_flat, p=2, dim=1)

            target = torch.ones(mapped_image_embeddings_flat.size(0)).to(device)  # [batch_size * seq_len]
            loss = criterion(mapped_image_embeddings_flat, text_embeddings_flat, target)

            # Calculate mean cosine similarity
            cosine_sim = F.cosine_similarity(mapped_image_embeddings_flat, text_embeddings_flat)
            mean_cosine_sim = cosine_sim.mean().item()
            epoch_cosine_sim += mean_cosine_sim

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
                ).last_hidden_state  # [batch_size, 77, 768]

                # Get image embeddings
                image_outputs = clip_model.vision_model(
                    pixel_values=image_inputs,
                )
                image_embeddings = image_outputs.last_hidden_state

                # Mappa le embedding delle immagini
                mapped_image_embeddings = mapping_network(image_embeddings, text_embeddings)

                mapped_image_embeddings_flat = mapped_image_embeddings.reshape(-1, 768)
                text_embeddings_flat = text_embeddings.reshape(-1, 768)

                # Normalize embeddings
                mapped_image_embeddings_flat = F.normalize(mapped_image_embeddings_flat, p=2, dim=1)
                text_embeddings_flat = F.normalize(text_embeddings_flat, p=2, dim=1)

                target = torch.ones(mapped_image_embeddings_flat.size(0)).to(device)  # [batch_size * seq_len]
                loss = criterion(mapped_image_embeddings_flat, text_embeddings_flat, target)

                cosine_sim = F.cosine_similarity(mapped_image_embeddings_flat, text_embeddings_flat)
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
        batch_size=32,
        shuffle=True,
        collate_fn=custom_collate
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate
    )

    # Inizia l'addestramento
    training_mapping(train_dataloader, val_dataloader, clip_model, clip_processor, tokenizer, text_encoder, device)
