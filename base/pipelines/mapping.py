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
    
def contrastive_loss(mapped_image_embeddings, text_embeddings, temperature=0.07):
    # Normalizza le embeddings
    mapped_image_embeddings = F.normalize(mapped_image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    # Calcola la similarità coseno
    logits = torch.matmul(mapped_image_embeddings, text_embeddings.transpose(-1, -2)) / temperature
    batch_size = logits.size(0)
    
    # Creare etichette per matching corretto
    labels = torch.arange(batch_size).to(logits.device)
    
    # Calcolo della loss per image-to-text e text-to-image
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.transpose(0, 1), labels)
    
    loss = (loss_i2t + loss_t2i) / 2
    return loss

    
class MappingNetwork(nn.Module):
    def __init__(self, input_dim=768, output_dim=768, num_layers=6, num_heads=8, seq_len_in=50, seq_len_out=77):
        super(MappingNetwork, self).__init__()
        # Layer di proiezione opzionale (puoi rimuoverla se input_dim == output_dim)
        # self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer per proiettare la sequenza da seq_len_in a seq_len_out
        self.seq_proj = nn.Linear(seq_len_in, seq_len_out)
    
    def forward(self, x):
        # x: [batch_size, seq_len_in, input_dim]
        # Se input_dim == output_dim, puoi saltare la proiezione iniziale
        # x = self.input_proj(x)  # [batch_size, seq_len_in, output_dim]
        
        # Trasponi per adattare alla forma richiesta dal Transformer
        x = x.permute(1, 0, 2)  # [seq_len_in, batch_size, output_dim]
        
        # Applica il Transformer Encoder
        x = self.transformer_encoder(x)  # [seq_len_in, batch_size, output_dim]
        
        # Ritorna alla forma originale
        x = x.permute(1, 0, 2)  # [batch_size, seq_len_in, output_dim]
        
        # Proietta la sequenza da seq_len_in a seq_len_out
        x = x.transpose(1, 2)  # [batch_size, output_dim, seq_len_in]
        x = self.seq_proj(x)   # [batch_size, output_dim, seq_len_out]
        x = x.transpose(1, 2)  # [batch_size, seq_len_out, output_dim]
        
        return x
    
    
def training_mapping(train_dataloader, val_dataloader, clip_model, clip_processor, tokenizer, text_encoder, device):
    mapping_network = MappingNetwork(
        input_dim=768,
        output_dim=768,
        num_layers=6,
        num_heads=8,
        seq_len_in=50,
        seq_len_out=77
    ).to(device)

    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.AdamW(mapping_network.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 20
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    linear_layer = nn.Linear(1024, 768).to(device)

    for epoch in range(num_epochs):
        mapping_network.train()
        epoch_loss = 0.0
        epoch_cosine_sim = 0.0

        for images, descriptions in train_dataloader:
            if not images or not descriptions:
                continue

            image_inputs = clip_processor(images=list(images), return_tensors="pt").pixel_values.to(device)

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
                ).pooler_output  # [batch_size, 77, 768]

                image_embeddings = clip_model.vision_model(
                    pixel_values=image_inputs,
                ).pooler_output  # [batch_size, 50, 768]

            image_embeddings = linear_layer(image_embeddings)

            print(f"text_embeddings shape: {text_embeddings.shape}, dtype: {text_embeddings.dtype}")
            print(f"image_embeddings shape: {image_embeddings.shape}, dtype: {image_embeddings.dtype}")

            cosine_similarities = F.cosine_similarity(text_embeddings, image_embeddings, dim=1)
            print(cosine_similarities)
        



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
