#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import torch

import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset
from data.ksdd2 import KolektorSDD2


class KolektorSDD2Crops(Dataset):
    """Dataset for pre-cropped defect regions from KSDD2."""
    
    # Normalization as KolektorSDD2
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    def __init__(self,
                 crop_root,
                 transform=None,
                 add_augmented=False,
                 augmented_crop_root=None):
        """
        Args:
            crop_root (str): Directory containing pre-cropped defect images
            transform (callable, optional): Optional transform to be applied
        """
        self.crop_root = crop_root
        self.add_augmented = add_augmented
        self.augmented_crop_root = augmented_crop_root

        # Load original cropped defects
        self.crop_files = []
        self.mask_files = []
        self.load_crops(crop_root, self.crop_files, self.mask_files)
        
        if add_augmented and os.path.exists(augmented_crop_root):
            # Load augmented cropped defects
            self.load_crops(augmented_crop_root, self.crop_files, self.mask_files)

        print(f"Loaded {len(self.crop_files)} toal defect crops")
                
        self.transform = transform or self.get_default_transform()
        self.normalize = T.Normalize(self.mean, self.std)
    
    def load_crops(self, root_dir, crop_files, mask_files):
        """Load crop and mask filenames from a directory."""
        for file in os.listdir(root_dir):
            if file.endswith('.png') and not file.endswith('_GT.png'):
                # For each crop file, find its corresponding mask
                mask_file = file.replace('.png', '_GT.png')
                if os.path.exists(os.path.join(root_dir, mask_file)):
                    # Store full paths for easier loading later
                    crop_files.append(os.path.join(root_dir, file))
                    mask_files.append(os.path.join(root_dir, mask_file))
    
    def __len__(self):
        return len(self.crop_files)
    
    def __getitem__(self, idx):
        # Load crop and mask
        crop_path = os.path.join(self.crop_root, self.crop_files[idx])
        mask_path = os.path.join(self.crop_root, self.mask_files[idx])
        
        crop = Image.open(crop_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Apply transformations
        if self.transform:
            crop = self.transform(crop)
            mask = self.transform(mask)
        
        # Apply normalization to the crop only
        crop = self.normalize(crop)
        
        # Always mark as defect (class=1)
        return crop, 1, mask, 0
    
    @staticmethod
    def get_default_transform():
        """Default transformation for crops (preserves original dimensions)"""
        return T.Compose([
            T.ToTensor()
        ])