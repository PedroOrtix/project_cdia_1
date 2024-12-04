from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class AnimalDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_datasets(val_split=0.1):
    """
    Carga y prepara los datasets de animales de HuggingFace
    """
    # Configuración de transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Cargar el dataset
    ds = load_dataset("Fr0styKn1ght/Animals")
    
    # Dividir en train y test
    train_test_split = ds['train'].train_test_split(test_size=val_split)
    
    # Crear datasets de PyTorch
    train_dataset = AnimalDataset(train_test_split['train'], transform=transform)
    val_dataset = AnimalDataset(train_test_split['test'], transform=transform)
    
    return train_dataset, val_dataset
