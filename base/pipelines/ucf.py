import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from torchvision.io import read_video
import numpy as np
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ConvertImageDtype, Normalize
from scipy.linalg import sqrtm
from torch import nn
from tqdm import tqdm
import torchvision.models.video as models_video


class UCF101Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, num_frames=16):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.num_frames = num_frames

        # Construct the full path to the CSV file
        csv_file_path = os.path.join(self.root_dir, self.csv_file)

        # Load annotations from the CSV file
        self.annotations = pd.read_csv(csv_file_path, sep=',')

        print("Columns in self.annotations:", self.annotations.columns)

        # Create a mapping from class labels to indices
        self.classes = sorted(self.annotations['label'].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}

        self.class_to_indices = {
            class_name: self.annotations[self.annotations['label'] == class_name].index.tolist()
            for class_name in self.classes
        }

        self.frame_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # Da [H, W, C] a [C, H, W]
            transforms.ToPILImage(),
            transforms.Resize((320, 512)),  # Ridimensiona a (altezza, larghezza)
            transforms.ToTensor(),  # Converte in tensor con valori in [0, 1]
            transforms.Lambda(lambda x: (x * 255).byte()),  # Scala a [0, 255] e converte in uint8
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Da [C, H, W] a [H, W, C]
        ])


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_info = self.annotations.iloc[idx]
        clip_path = video_info['clip_path'].lstrip('/')
        video_path = os.path.join(self.root_dir, clip_path)
        label = self.class_to_idx[video_info['label']]

        # Carica i frame del video
        video_frames, _, _ = read_video(video_path, pts_unit='sec')

        #print(f"video_frames shape: {video_frames.shape}, dtype: {video_frames.dtype}") #torch.Size([126, 240, 320, 3]), dtype: torch.uint8

        single_frame = video_frames[1]

        single_frame = self.frame_transform(single_frame)

        #print(f"single_frame1 shape: {single_frame.shape}, dtype: {single_frame.dtype}") #torch.Size([240, 320, 3]), dtype: torch.uint8

        # Assicurati che il video abbia il numero desiderato di frame
        frames = self.process_frames(video_frames)

        #print(f"frames shape: {frames.shape}, dtype: {frames.dtype}") #torch.Size([3, 16, 240, 320]), dtype: torch.uint8

        # frames shape: [C, T, H, W]
        # Permute to [T, C, H, W] per applicare le trasformazioni
        frames = frames.permute(1, 0, 2, 3)

        #print(f"frames1 shape: {frames.shape}, dtype: {frames.dtype}") #torch.Size([16, 3, 240, 320]), dtype: torch.uint8

        if self.transform:
            # Applica la trasformazione a ciascun frame
            frames = torch.stack([self.transform(frame) for frame in frames])
        else:
            # Normalizza i frame se nessuna trasformazione è fornita
            frames = frames.float() / 255.0

        #print(f"frames2 shape: {frames.shape}, dtype: {frames.dtype}") #torch.Size([16, 3, 224, 224]), dtype: torch.float32

        # Permute di nuovo a [C, T, H, W]
        frames = frames.permute(1, 0, 2, 3)

        #print(f"frames3 shape: {frames.shape}, dtype: {frames.dtype}") #torch.Size([3, 16, 224, 224]), dtype: torch.float32


        sample = {
            'frames': frames,
            'label': label,
            'frame': single_frame
        }

        return sample


    def process_frames(self, video_frames):
        num_video_frames = video_frames.shape[0]

        if num_video_frames > self.num_frames:
            indices = np.linspace(0, num_video_frames - 1, num=self.num_frames, dtype=int)
            frames = video_frames[indices]
        elif num_video_frames < self.num_frames:
            last_frame = video_frames[-1].unsqueeze(0)
            repeat_times = self.num_frames - num_video_frames
            padding = last_frame.repeat(repeat_times, 1, 1, 1)
            frames = torch.cat((video_frames, padding), dim=0)
        else:
            frames = video_frames

        frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]
        return frames
    

# Funzione di preprocessamento per i video generati
def preprocess_generated_video(video_tensor):
    """
    Preprocessa un video generato per adattarsi ai requisiti dell'I3D.
    :param video_tensor: Tensor di forma [16, 320, 512, 3], dtype torch.uint8
    :return: Tensor di forma [3, 16, 224, 224]
    """
    # Converti a [C, T, H, W] e normalizza
    video = video_tensor.permute(3, 0, 1, 2).float() / 255.0  # [3, 16, 320, 512]

    # Definisci le trasformazioni: center crop a 270x270 e resize a 224x224
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(270),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    ])

    frames = []
    for t in range(video.shape[1]):
        frame = video[:, t, :, :]  # [3, H, W]
        frame = transform(frame)
        frames.append(frame)

    # Stack dei frame: [T, C, H, W]
    video = torch.stack(frames)  # [16, 3, 224, 224]

    # Permuta a [C, T, H, W]
    video = video.permute(1, 0, 2, 3)  # [3, 16, 224, 224]

    return video

# Funzione per estrarre le feature usando I3D
def extract_i3d_features(videos, i3d_model, device):
    """
    Estrae le feature dei video utilizzando il modello I3D.
    :param videos: Tensor di forma [N, C, T, H, W]
    :param i3d_model: Modello I3D pre-addestrato
    :param device: Dispositivo (CPU o GPU)
    :return: Tensor delle feature di forma [N, feature_dim]
    """
    videos = videos.to(device)
    with torch.no_grad():
        features = i3d_model(videos)  # Supponendo che il modello restituisca [N, feature_dim]
    return features.cpu()

# Funzione per calcolare l'FVD
def compute_fvd(features_gen, features_real):
    """
    Calcola la Fréchet Video Distance (FVD) tra le feature generate e reali.
    :param features_gen: Numpy array delle feature generate [N_gen, feature_dim]
    :param features_real: Numpy array delle feature reali [N_real, feature_dim]
    :return: Valore FVD
    """
    # Calcola la media e la covarianza delle feature generate
    mu_gen = np.mean(features_gen, axis=0)
    sigma_gen = np.cov(features_gen, rowvar=False)

    # Calcola la media e la covarianza delle feature reali
    mu_real = np.mean(features_real, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)

    # Calcola la radice quadrata del prodotto delle matrici di covarianza
    covmean, _ = sqrtm(sigma_gen.dot(sigma_real), disp=False)

    # Gestione dei numeri complessi
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calcola l'FVD
    fvd = np.sum((mu_gen - mu_real) ** 2) + np.trace(sigma_gen + sigma_real - 2 * covmean)
    return fvd


if __name__ == "__main__":
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize frames to 224x224
        ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean and std
                            std=[0.229, 0.224, 0.225])
    ])

    # Create the dataset
    train_dataset = UCF101Dataset(
        csv_file='test.csv',
        root_dir='/content/drive/My Drive/UCF101',  # Replace with your actual path
        transform=transform,
        num_frames=16
    )

    # Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
