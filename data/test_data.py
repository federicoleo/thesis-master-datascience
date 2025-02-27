import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
    
class SimpleDataset(Dataset):
    def __init__(self, num_samples=10):
        self.data = torch.randn(num_samples, 3, 256, 256)
        self.labels = torch.zeros(num_samples)  # All normal samples
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        # Preprocessing
        img = F.center_crop(img, 224)

        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return img, label