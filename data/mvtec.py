#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset

def c2chw(x):
    return x.unsqueeze(1).unsqueeze(2)


def inverse_list(list_obj):
    """
    List to dict: index -> element
    """
    dict_obj = {}
    for idx, x in enumerate(list_obj):
        dict_obj[x] = idx
    return dict_obj


class MVTecAD(Dataset):
    """
    MVTec Anomaly Detection dataset

    Args:
        dataroot (string): path to the root directory of the dataset
        category (string): specific product category to use (e.g., 'bottle', 'cable')
                          if None, uses all categories
        split (string): data split ['train', 'test']
        img_size (tuple): resize image to this size
    """

    # All product categories in MVTec AD
    CATEGORIES = [
        "bottle", "cable", "capsule", "carpet", "grid", 
        "hazelnut", "leather", "metal_nut", "pill", "screw", 
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]
    
    # Class labels
    LABELS = ['normal', 'anomaly']

    # ImageNet normalization values
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
            self,
            dataroot='/path/to/dataset/mvtec_anomaly_detection',
            category=None,
            split='train',
            img_size=(224, 224),
            negative_only=False,
            add_augmented=False,
            num_augmented=0,
            zero_shot=False
        ):
        super(MVTecAD, self).__init__()

        self.dataroot = dataroot
        self.split = split
        self.img_size = img_size
        self.categories = [category] if category else self.CATEGORIES
        self.negative_only = negative_only,
        self.add_augmented = add_augmented,
        self.num_augmented = num_augmented,
        self.zero_shot = zero_shot

        if self.add_augmented:
            assert self.split == 'train', 'Augmented images are only for the training set!'
        
        if self.zero_shot:
            assert self.add_augmented, 'Zero-shot learning requires augmented images!'
            assert self.split == 'train', 'Zero-shot learning is only for the training set!'
        
        self.class_to_idx = inverse_list(self.LABELS)
        self.transform = self.get_transform(img_size)
        self.normalize = T.Normalize(self.MEAN, self.STD)
        
        # Load dataset
        self.samples = []  # Will store (image_path, mask_path, label)
        self.load_dataset()

        # Filter samples if negative_only is True
        if negative_only:
            self.samples = [(img, mask, label) for img, mask, label in self.samples if label == 0]

    
    def load_dataset(self):
        """Load image paths and corresponding labels/masks"""
        for category in self.categories:
            category_dir = os.path.join(self.dataroot, category)
            
            if self.split == 'train':
                # Training set only has normal images
                train_dir = os.path.join(category_dir, 'train')
                normal_dir = os.path.join(train_dir, 'good')
                
                if os.path.exists(normal_dir):
                    for img_name in os.listdir(normal_dir):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(normal_dir, img_name)
                            self.samples.append((img_path, None, 0))  # Normal = 0, no mask
                
                # In MVTec, training set typically only has normal images,
                # but add this logic for completeness
                for defect_type in os.listdir(train_dir):
                    if defect_type == 'good':
                        continue  # Already processed
                    
                    defect_dir = os.path.join(train_dir, defect_type)
                    if not os.path.isdir(defect_dir):
                        continue
                    
                    for img_name in os.listdir(defect_dir):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(defect_dir, img_name)
                            # Anomalous sample
                            self.samples.append((img_path, None, 1))  # Anomaly = 1


            elif self.split == 'test':
                # Test set has both normal and anomalous images
                test_dir = os.path.join(category_dir, 'test')
                gt_dir = os.path.join(category_dir, 'ground_truth')
                
                # Collect normal images
                normal_dir = os.path.join(test_dir, 'good')
                if os.path.exists(normal_dir):
                    for img_name in os.listdir(normal_dir):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(normal_dir, img_name)
                            self.samples.append((img_path, None, 0))  # Normal = 0, no mask
                
                # Collect anomalous images from defect subfolders
                for defect_type in os.listdir(test_dir):
                    if defect_type == 'good':
                        continue  # Already processed normal images
                    
                    defect_dir = os.path.join(test_dir, defect_type)
                    if not os.path.isdir(defect_dir):
                        continue
                    
                     # The corresponding mask directory for this defect type
                    mask_dir = os.path.join(gt_dir, defect_type)
                    
                    for img_name in os.listdir(defect_dir):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(defect_dir, img_name)
                            
                            # Find corresponding mask - the name format is always 
                            # filename_mask.png regardless of the original extension
                            base_name = os.path.splitext(img_name)[0]  # Get filename without extension
                            mask_name = f"{base_name}_mask.png"
                            mask_path = os.path.join(mask_dir, mask_name)
                            
                            if os.path.exists(mask_path):
                                self.samples.append((img_path, mask_path, 1))  # Anomaly = 1
    
    def process_samples(self):
        """Process samples based on parameters (negative_only, zero_shot)"""
        # Handle negative_only parameter - filter out positive samples
        if self.negative_only:
            self.samples = [(img, mask, label) for img, mask, label in self.samples if label == 0]
        
        # Handle zero_shot parameter - filter out real positive samples
        if self.zero_shot and self.split == 'train':
            self.samples = [(img, mask, label) for img, mask, label in self.samples if label == 0]
        
        # Handle augmented data
        if self.add_augmented and self.split == 'train':
            # Placeholder for augmented data loading
            # This will be implemented in the future extension
            augmented_dir = os.path.join(self.dataroot, 'augmented')
            if self.num_augmented > 0:
                augmented_dir = os.path.join(self.dataroot, f'augmented_{self.num_augmented}')
            
            # Once you implement augmentation, you'll need to add logic here
            # to load and incorporate the augmented images into self.samples
            pass
    

    def __getitem__(self, index):
        img_path, mask_path, label = self.samples[index]
        
        # Load and transform the image
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        # Load and transform the mask (if available)
        if mask_path is not None:
            mask = Image.open(mask_path).convert('L')
            mask = self.transform(mask) > 0  # Convert to binary mask
        else:
            mask = torch.zeros(1, *self.img_size)
        
        # Apply normalization to image
        if self.normalize is not None:
            img = self.normalize(img)
        
        return img, mask, label
    
    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def get_transform(output_size):
        transform = [
            T.Resize(output_size),
            T.ToTensor()
        ]
        transform = T.Compose(transform)
        return transform
    
    @staticmethod
    def denorm(x):
        return x * c2chw(torch.Tensor(MVTecAD.STD)) + c2chw(torch.Tensor(MVTecAD.MEAN))