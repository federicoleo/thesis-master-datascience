#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import wandb
import argparse
from tqdm import tqdm

import numpy as np
import torch

import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader

from DiversitySampling.src.coreset import CoresetSampler
from data.mvtec import MVTecAD

from torchvision import transforms
from PIL import ImageFilter
from torch import tensor
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger(__name__)

class PatchCoreSingleBank(torch.nn.Module):
    def __init__(self, device='mps', image_size=224):
        super(PatchCoreSingleBank, self).__init__()
        self.k_nearest = 3
        self.image_size = image_size
        self.device = device
        self.memory_bank = None
        self.extracted_features = []

        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
        
        def hook(module, input, output):
            self.extracted_features.append(output)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, sample):
        self.extracted_features = []
        _ = self.model(sample)
        return self.extracted_features

    def fit(self, dataloader: DataLoader):
        memory_items = []
        for sample, _, _ in tqdm(dataloader, desc="Building Memory Bank"):
            sample = sample.to(self.device)
            features = self.process_features(sample)
            memory_items.extend(features)
        self.memory_bank = torch.cat(memory_items, dim=0)
        N, C, H, W = self.memory_bank.shape
        print(f"Memory Bank: {N} samples of shape {C}x{H}x{W}")

        share = 0.01
        target = max(1000, int(N * H * W * share))
        self.memory_bank, indices = coreset_subsampling(
            self.memory_bank, target, epsilon=0.1, device=self.device
        )
        print(f"Memory Bank reduced to {len(indices)} patch embeddings")

    def process_features(self, sample):
        features = self(sample)
        layer2 = local_neighborhood_aggregation(features[0], p=3)
        layer3 = local_neighborhood_aggregation(features[1], p=3)
        layer3 = bilinear_upsample(layer3, target_size=layer2.shape[2:])
        combined = torch.cat([layer2, layer3], dim=1)
        resized = bilinear_upsample(combined, target_size=(28, 28))
        return [resized[b].unsqueeze(0) for b in range(resized.shape[0])]

    def evaluate(self, test_dataloader: DataLoader):
        image_preds = []
        image_labels = []
        pixel_preds = []
        pixel_labels = []

        for sample, mask, label in tqdm(test_dataloader, desc="Evaluating"):
            sample, mask = sample.to(self.device), mask.to(self.device)
            
            image_labels.append(label.numpy())
            pixel_labels.extend(mask.flatten().cpu().numpy())
            score, segm_map = self.predict(sample)
            image_preds.append(score.cpu().numpy())
            pixel_preds.extend(segm_map.flatten().cpu().numpy())

        image_labels = np.concatenate(image_labels)
        image_preds = np.array(image_preds)
        image_auc = roc_auc_score(image_labels, image_preds)
        pixel_auc = roc_auc_score(pixel_labels, pixel_preds)
        return image_auc, pixel_auc
    
    def predict(self, sample):
        feature_maps = self(sample)
        feature_maps = [local_neighborhood_aggregation(fm, p=3) for fm in feature_maps]
        feature_maps[1] = bilinear_upsample(feature_maps[1], target_size=feature_maps[0].shape[2:])
        patch_collection = torch.cat(feature_maps, dim=1)
        patch_collection = patch_collection.reshape(patch_collection.shape[1], -1).T

        distances = torch.cdist(patch_collection, self.memory_bank, p=2.0)
        dist_score, dist_score_idx = torch.min(distances, dim=1)
        
        s_idx = torch.argmax(dist_score)
        s_star = dist_score[s_idx]
        m_test_star = patch_collection[s_idx]
        
        m_star = self.memory_bank[dist_score_idx[s_idx]].unsqueeze(0)
        knn_dists = torch.cdist(m_star, self.memory_bank, p=2.0)
        _, nn_idxs = knn_dists.topk(k=self.k_nearest, largest=False)
        m_neighborhood = self.memory_bank[nn_idxs[0, 1:]]
        w_denominator = torch.linalg.norm(m_test_star - m_neighborhood, dim=1)
        norm = torch.sqrt(torch.tensor(patch_collection.shape[1], device=self.device))
        w = 1 - (torch.exp(s_star / norm) / torch.sum(torch.exp(w_denominator / norm)))
        
        s_final = w * s_star
        
        fmap_size = feature_maps[0].shape[-2:]
        segm_map = dist_score.view(1, 1, *fmap_size)
        segm_map = bilinear_upsample(segm_map, (self.image_size, self.image_size))
        segm_map = gaussian_blur(segm_map)
        
        return s_final, segm_map

# Helper Functions (unchanged)
def local_neighborhood_aggregation(feature_map, p=3):
    B, C, H, W = feature_map.shape
    offset = p // 2
    padded = F.pad(feature_map, (offset, offset, offset, offset), mode='reflect')
    neighborhoods = F.unfold(padded, kernel_size=p, stride=1)
    neighborhoods = neighborhoods.view(B, C, p*p, H*W).permute(0, 3, 1, 2).reshape(B*H*W, C, p, p)
    pooled = F.adaptive_avg_pool2d(neighborhoods, (1, 1))
    return pooled.reshape(B, H*W, C, 1).squeeze(-1).permute(0, 2, 1).reshape(B, C, H, W)

def bilinear_upsample(lower_spatial_block, target_size):
    if lower_spatial_block.shape[2:] == target_size:
        return lower_spatial_block
    return F.interpolate(lower_spatial_block, size=target_size, mode='bilinear', align_corners=False)

def random_projection(embeddings, target_dim, epsilon=0.1, seed=0):
    N, C = embeddings.shape
    torch.manual_seed(seed)
    projection_matrix = torch.randn(C, target_dim, device=embeddings.device)
    projection_matrix /= torch.sqrt(torch.sum(projection_matrix**2, dim=0, keepdim=True))
    return torch.matmul(embeddings, projection_matrix)

def coreset_subsampling(embeddings, target_samples, epsilon=0.1, device=None, use_projection=True):
    if device is None:
        device = embeddings.device
    original_shape = embeddings.shape
    if len(original_shape) > 2:
        N, C, H, W = embeddings.shape
        reshaped_embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, C)
    else:
        reshaped_embeddings = embeddings
    n_samples, C = reshaped_embeddings.shape
    if use_projection and C > 10:
        d_star = 128
        if d_star < C:
            print(f"Projecting from {C} to {d_star} dimensions")
            embeddings_for_sampling = random_projection(reshaped_embeddings, d_star, epsilon)
        else:
            embeddings_for_sampling = reshaped_embeddings
    else:
        embeddings_for_sampling = reshaped_embeddings
    target_samples = min(target_samples, n_samples)
    sampler = CoresetSampler(n_samples=target_samples, device=str(device), tqdm_disable=False, verbose=1)
    selected_indices = sampler.sample(embeddings_for_sampling.cpu().numpy())
    return reshaped_embeddings[selected_indices], selected_indices

def gaussian_blur(img):
    tensor_to_pil = transforms.ToPILImage()
    pil_to_tensor = transforms.ToTensor()
    max_value = img.max()
    blurred_pil = tensor_to_pil(img[0] / max_value).filter(ImageFilter.GaussianBlur(radius=4))
    return pil_to_tensor(blurred_pil) * max_value

def main(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging = args.logging
    run_name = f'PatchCoreSingleBank-bs_{args.batch_size}-two_categories'
    if logging:
        wandb.init(name=run_name, config=args, tags=['single_bank'])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Limit to two categories for testing
    categories = ["bottle", "cable"]
    image_aucs = {}
    pixel_aucs = {}

    for category in categories:
        print(f"\nProcessing category: {category}")
        train_data = MVTecAD(
            dataroot=args.dataset_path,
            split='train',
            category=category,
            negative_only=True,  # Only normal samples
            memory_bank_type='negative',
            transform=transform
        )
        test_data = MVTecAD(
            dataroot=args.dataset_path,
            split='test',
            category=category,
            memory_bank_type='test',
            transform=transform
        )
        print(f"{category} - Train: {len(train_data)}, Test: {len(test_data)}")

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

        model = PatchCoreSingleBank(device=device, image_size=224)
        print(f"Building memory bank for {category} on {device} [...]")
        model.fit(train_loader)
        print(f"Evaluating {category}...")
        image_auc, pixel_auc = model.evaluate(test_loader)
        image_aucs[category] = image_auc
        pixel_aucs[category] = pixel_auc
        print(f"{category} - Image AUC: {image_auc:.4f}, Pixel AUC: {pixel_auc:.4f}")

    avg_image_auc = np.mean(list(image_aucs.values()))
    avg_pixel_auc = np.mean(list(pixel_aucs.values()))
    print(f"\nFinal Results:")
    print(f"Average Image AUC: {avg_image_auc:.4f}")
    print(f"Average Pixel AUC: {avg_pixel_auc:.4f}")

    if logging:
        wandb.log({'avg_image_auc': avg_image_auc, 'avg_pixel_auc': avg_pixel_auc})
        for category in categories:
            wandb.log({f'{category}_image_auc': image_aucs[category], f'{category}_pixel_auc': pixel_aucs[category]})
        wandb.finish()

    print("PatchCore Single Bank evaluation finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PatchCore with Single Memory Bank for MVTec')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoaders')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoaders')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to MVTec dataset')
    parser.add_argument('--logging', action='store_true', help='Log stats to wandb')
    args = parser.parse_args()
    main(args)