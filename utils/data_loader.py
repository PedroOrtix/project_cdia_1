from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class BeansDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['labels']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_datasets():
    """
    Carga y prepara los datasets de plantas de frijol de HuggingFace
    """
    # Configuración de transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Cargar el dataset
    ds = load_dataset("AI-Lab-Makerere/beans")
    
    # Crear datasets de PyTorch
    train_dataset = BeansDataset(ds['train'], transform=transform)
    val_dataset = BeansDataset(ds['validation'], transform=transform)
    
    return train_dataset, val_dataset
