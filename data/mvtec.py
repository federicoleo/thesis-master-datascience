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
    dict_obj = {}
    for idx, x in enumerate(list_obj):
        dict_obj[x] = idx
    return dict_obj

class MVTecAD(Dataset):
    CATEGORIES = [
        "bottle", "cable", "capsule", "carpet", "grid", 
        "hazelnut", "leather", "metal_nut", "pill", "screw", 
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]
    LABELS = ['normal', 'anomaly']
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
            self,
            dataroot='data/mvtec_preprocessed',
            split='train',
            category=None,
            negative_only=False,
            add_augmented=False,
            num_augmented=0,
            zero_shot=False,
            memory_bank_type='negative',  # 'negative', 'positive', 'test'
            transform=None
        ):
        super(MVTecAD, self).__init__()
        self.dataroot = dataroot
        self.split = split
        self.categories = [category] if category else self.CATEGORIES
        # self.output_size = (256, 256)
        self.negative_only = negative_only
        self.add_augmented = add_augmented
        self.num_augmented = num_augmented
        self.zero_shot = zero_shot
        self.memory_bank_type = memory_bank_type

        # Validation

        if self.memory_bank_type == 'positive':
            assert self.add_augmented, 'Positive memory bank requires augmented images!'
        if self.split == 'test':
            if self.memory_bank_type != 'test':
                raise ValueError("Test split must use memory_bank_type='test'")
            if self.add_augmented:
                print("Warning: add_augmented=True is ignored for test split; using original test data only")
                self.add_augmented = False  # Enforce no augmentation for test

        self.class_to_idx = inverse_list(self.LABELS)
        self.classes = self.LABELS
        # self.transform = self.get_transform(self.output_size)
        self.transform = transform
        self.normalize = T.Normalize(self.MEAN, self.STD)
        
        self.samples = None
        self.masks = None
        self.product_ids = []
        self.load_imgs()

    def load_imgs(self):
        if self.split not in ['train', 'test']:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'test'.")
        
        split_dir = os.path.join(self.dataroot, self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"{split_dir} does not exist")
        
        N = 0
        augmented_paths = {}
        
        if self.memory_bank_type == 'negative':
            print(f"Loading negative images for {self.split} split")
            for category in self.categories:
                category_dir = os.path.join(split_dir, category)
                if not os.path.exists(category_dir):
                    continue
                for defect_type in os.listdir(category_dir):
                    defect_dir = os.path.join(category_dir, defect_type)
                    if not os.path.isdir(defect_dir):
                        continue
                    if defect_type == 'good' or not self.negative_only:
                        img_files = [f for f in os.listdir(defect_dir) if f.endswith(".png") and not f.endswith("_GT.png")]
                        N += len(img_files)
        
        elif self.memory_bank_type == 'positive':
            print(f"Loading positive images for {self.split} split")
            for category in self.categories:
                if self.num_augmented > 0:
                    aug_imgs_path = os.path.join(self.dataroot, 'augmented', category, f'augmented_{self.num_augmented}', 'imgs')
                    aug_masks_path = os.path.join(self.dataroot, 'augmented', category, f'augmented_{self.num_augmented}', 'masks')
                else:
                    aug_imgs_path = os.path.join(self.dataroot, 'augmented', category, 'augmented', 'imgs')
                    aug_masks_path = os.path.join(self.dataroot, 'augmented', category, 'augmented', 'masks')
                if os.path.exists(aug_imgs_path):
                    aug_img_files = [f for f in os.listdir(aug_imgs_path) if f.endswith(".png") and not f.endswith("_GT.png")]
                    N += len(aug_img_files)
                    augmented_paths[category] = (aug_imgs_path, aug_masks_path)
                else:
                    print(f"Warning: {aug_imgs_path} does not exist, skipping augmented images for {category}")
        
        elif self.memory_bank_type == 'test':
            print(f"Loading test images for {self.split} split")
            for category in self.categories:
                category_dir = os.path.join(split_dir, category)
                if not os.path.exists(category_dir):
                    continue
                for defect_type in os.listdir(category_dir):
                    defect_dir = os.path.join(category_dir, defect_type)
                    if not os.path.isdir(defect_dir):
                        continue
                    img_files = [f for f in os.listdir(defect_dir) if f.endswith(".png") and not f.endswith("_GT.png")]
                    N += len(img_files)

        self.samples = torch.zeros(N, 3, 224, 224)
        self.masks = torch.zeros(N, 224, 224, dtype=torch.long)
        self.product_ids = []
        idx = 0

        if self.memory_bank_type == 'negative':
            for category in self.categories:
                category_dir = os.path.join(split_dir, category)
                if not os.path.exists(category_dir):
                    continue
                for defect_type in os.listdir(category_dir):
                    defect_dir = os.path.join(category_dir, defect_type)
                    if not os.path.isdir(defect_dir):
                        continue
                    if defect_type == 'good' or not self.negative_only:
                        img_files = [f for f in os.listdir(defect_dir) if f.endswith(".png") and not f.endswith("_GT.png")]
                        for img_file in img_files:
                            product_id = img_file[:-4]
                            img_path = os.path.join(defect_dir, img_file)
                            mask_path = os.path.join(defect_dir, f"{product_id}_GT.png")
                            img = self.transform(Image.open(img_path))
                            mask = self.transform(Image.open(mask_path).convert('L'))
                            mask = (mask * 255).long().squeeze(0) # [224, 224]
                            if not self.negative_only or mask.sum() == 0:
                                self.samples[idx] = img
                                self.masks[idx] = mask
                                self.product_ids.append(product_id)
                                idx += 1

        elif self.memory_bank_type == 'positive':
            for category in self.categories:
                if category in augmented_paths:
                    aug_imgs_path, aug_masks_path = augmented_paths[category]
                    aug_img_files = [f for f in os.listdir(aug_imgs_path) if f.endswith(".png") and not f.endswith("_GT.png")]
                    for img_file in aug_img_files:
                        product_id = img_file[:-4]
                        img_path = os.path.join(aug_imgs_path, img_file)
                        mask_path = os.path.join(aug_masks_path, f"{product_id}_GT.png")
                        img = self.transform(Image.open(img_path))
                        mask = self.transform(Image.open(mask_path).convert('L'))
                        mask = (mask * 255).long().squeeze(0)
                        self.samples[idx] = img
                        self.masks[idx] = mask
                        self.product_ids.append(product_id)
                        idx += 1

        elif self.memory_bank_type == 'test':
            for category in self.categories:
                category_dir = os.path.join(split_dir, category)
                if not os.path.exists(category_dir):
                    continue
                for defect_type in os.listdir(category_dir):
                    defect_dir = os.path.join(category_dir, defect_type)
                    if not os.path.isdir(defect_dir):
                        continue
                    img_files = [f for f in os.listdir(defect_dir) if f.endswith(".png") and not f.endswith("_GT.png")]
                    for img_file in img_files:
                        product_id = img_file[:-4]
                        img_path = os.path.join(defect_dir, img_file)
                        mask_path = os.path.join(defect_dir, f"{product_id}_GT.png")
                        img = self.transform(Image.open(img_path))
                        mask = self.transform(Image.open(mask_path).convert('L'))
                        mask = (mask * 255).long().squeeze(0)
                        self.samples[idx] = img
                        self.masks[idx] = mask
                        self.product_ids.append(product_id)
                        idx += 1

        self.samples = self.samples[:idx]
        self.masks = self.masks[:idx]
        self.N = idx
        print(f"Loaded {self.N} images for {self.split} split (memory_bank_type={self.memory_bank_type}, negative_only={self.negative_only}, zero_shot={self.zero_shot})")

    def __getitem__(self, index):
        img = self.samples[index]
        mask = self.masks[index]
        label = self.class_to_idx['anomaly'] if mask.sum() > 0 else self.class_to_idx['normal']
        
        if self.normalize is not None:
            img = self.normalize(img)
        
        return img, mask, label

    def __len__(self):
        return self.N

    @staticmethod
    def denorm(x):
        return x * c2chw(torch.Tensor(MVTecAD.STD)) + c2chw(torch.Tensor(MVTecAD.MEAN))