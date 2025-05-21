#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import torch

import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset


def c2chw(x):
    return x.unsqueeze(1).unsqueeze(2)


def inverse_list(list):
    """
    List to dict: index -> element
    """
    dict = {}

    for idx, x in enumerate(list):
        dict[x] = idx

    return dict


class KolektorSDD2(Dataset):
    """"
    Kolektor Surface-Defect 2 dataset

        Args:
            dataroot (string): path to the root directory of the dataset
            split    (string): data split ['train', 'test']
            scale    (string): input image scale
            debug    (bool)  : debug mode
    """

    labels = ['ok', 'defect']

    # ImageNet.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self,
                 dataroot='/path/to/dataset/'
                          'KolektorSDD2',
                 split='train', negative_only=False, positive_only=False,
                 add_augmented=False, num_augmented=100, zero_shot=False):
        super(KolektorSDD2, self).__init__()

        self.fold = None
        self.dataroot = dataroot

        self.split_path = None
        self.split = 'train' if 'val' == split else split

        self.output_size = (632//2, 224//2)
        self.negative_only = negative_only
        self.positive_only = positive_only
        self.add_augmented = add_augmented
        self.num_augmented = num_augmented
        if self.add_augmented:
            assert self.split == 'train', 'Augmented images are only for the training set!'
        self.num_augmented = num_augmented
        self.zero_shot = zero_shot
        if self.zero_shot:
            assert self.add_augmented, 'Zero-shot learning requires augmented images!'
            assert self.split == 'train', 'Zero-shot learning is only for the training set!'

        self.class_to_idx = inverse_list(self.labels)
        self.classes = self.labels
        self.transform = KolektorSDD2.get_transform(output_size=self.output_size)
        self.normalize = T.Normalize(KolektorSDD2.mean, KolektorSDD2.std)

        self.load_imgs()
        if negative_only:
            m = self.masks.sum(-1).sum(-1) == 0
            self.samples = self.samples[m]
            self.masks = self.masks[m]
            self.product_ids = [pid for flag, pid in zip(m, self.product_ids)
                                    if flag]
        
        if positive_only:
            m = self.masks.sum(-1).sum(-1) > 0  # Mask sum > 0 means it has defects
            self.samples = self.samples[m]
            self.masks = self.masks[m]
            self.product_ids = [pid for flag, pid in zip(m, self.product_ids)
                                    if flag]


    def load_imgs(self):
        # Please remove this duplicated files in the official dataset:
        #   -- 10301_GT (copy).png
        #   -- 10301 (copy).png
        if self.num_augmented > 0:
            augmented_imgs_path = os.path.join(self.dataroot, f'augmented_{self.num_augmented}', 'imgs')
            augmented_masks_path = os.path.join(self.dataroot, f'augmented_{self.num_augmented}', 'masks')
        else:
            augmented_imgs_path = os.path.join(self.dataroot, f'augmented', 'imgs')
            augmented_masks_path = os.path.join(self.dataroot, f'augmented', 'masks')

        if self.split == 'test':
            N = 1004
        elif self.split == 'train' and self.negative_only:
            # only augmented positives and original negatives
            N = 2085 # number of original negatives
        elif self.split == 'train' and self.positive_only:
            N = 246
        else:
            # all original data + augmented
            N = 2331 # number of original negatives and positives
        
        if self.add_augmented:
            N += len(os.listdir(augmented_imgs_path))
            if self.num_augmented > 0:
                assert len(os.listdir(augmented_imgs_path)) == self.num_augmented, f'Number of augmented images requested ({self.num_augmented}) does not match with number found ({len(os.listdir(augmented_imgs_path))})!'

        self.samples = torch.Tensor(N, 3, *self.output_size).zero_()
        self.masks = torch.LongTensor(N, *self.output_size).zero_()
        self.product_ids = []

        cnt = 0
        path = "%s/%s/" % (self.dataroot, self.split)
        image_list = [f for f in os.listdir(path)
                      if re.search(r'[0-9]+\.png$', f)]
        assert 0 < len(image_list), self.dataroot

        for img_name in image_list:
            product_id = img_name[:-4]
            img = self.transform(Image.open(path + img_name))
            lab = self.transform(
                Image.open(path + product_id + '_GT.png').convert('L'))
            if self.negative_only:
                # check that the mask is negative
                if lab.sum() == 0:
                    self.samples[cnt] = img
                    self.masks[cnt] = lab
                    self.product_ids.append(product_id)
                    cnt += 1
            elif self.positive_only:
                if lab.sum() > 0:
                    self.samples[cnt] = img
                    self.masks[cnt] = lab
                    self.product_ids.append(product_id)
                    cnt += 1
            else:
                # default
                self.samples[cnt] = img
                self.masks[cnt] = lab
                self.product_ids.append(product_id)
                cnt += 1

        # Add the augmented images.
        if self.add_augmented:
            if 'train' == self.split:
                image_list = os.listdir(augmented_imgs_path)
                
                for img_name in image_list:
                    product_id = img_name[:-4]
                    img = self.transform(Image.open(os.path.join(augmented_imgs_path, img_name)))
                    mask_name = product_id + "_GT.png"
                    lab = self.transform(
                        Image.open(os.path.join(augmented_masks_path, mask_name)).convert('L'))
                    self.samples[cnt] = img
                    self.masks[cnt] = lab
                    self.product_ids.append(product_id)
                    cnt += 1

        assert N == cnt, '{} should be {}!'.format(cnt, N)


    def __getitem__(self, index):
        x = self.samples[index]
        a = self.masks[index] > 0
        if self.normalize is not None:
            x = self.normalize(x)

        if 0 == a.sum():
            y = self.class_to_idx['ok']
        else:
            y = self.class_to_idx['defect']

        return x, y, a, 0


    def __len__(self):
        return self.samples.size(0)


    @staticmethod
    def get_transform(output_size=(632//2, 224//2)):
        transform = [
            T.Resize(output_size),
            T.ToTensor()
        ]
        transform = T.Compose(transform)
        return transform


    @staticmethod
    def denorm(x):
        return x * c2chw(torch.Tensor(KolektorSDD2.std)) + c2chw(torch.Tensor(KolektorSDD2.mean))


if __name__ == "__main__":
    dataroot = "data/ksdd2_preprocessed"  # Change this to your dataset path

    # Create negative-only dataset
    # neg_dataset = KolektorSDD2(dataroot=dataroot, split='train', negative_only=True, add_augmented=False)
    # print(f"Negative-only dataset shape: ({len(neg_dataset)}, 3, {neg_dataset.output_size[0]}, {neg_dataset.output_size[1]})")

    # Create positive-only dataset with augmented images
    pos_dataset = KolektorSDD2(dataroot=dataroot, split='train', positive_only=True, add_augmented=False)
    print(f"Positive-only dataset shape: ({len(pos_dataset)}, 3, {pos_dataset.output_size[0]}, {pos_dataset.output_size[1]})")

    sample, label, mask, _ = pos_dataset[0]

    print(len(mask.shape))
    print(mask.unsqueeze(0).unsqueeze(0).shape) 