#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import torch
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, Wide_ResNet50_2_Weights
from torch.utils.data import DataLoader

from DiversitySampling.src.coreset import CoresetSampler
from torchvision import transforms
from PIL import ImageFilter
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger(__name__)

class PatchCoreSingle(torch.nn.Module):
    """
    PatchCore implementation adapted for KSDD2 dataset with the ability to work with 
    arbitrary image dimensions without upsampling to fixed sizes.
    """
    def __init__(self, device='cuda', backbone='resnet50', memory_type='negative', subsampling_share=0.01):
        """
        Initialize the PatchCore model for KSDD2.
        
        Args:
            device (str): Device to use ('cuda' or 'cpu')
            backbone (str): Backbone network to use ('resnet50' or 'wide_resnet50_2')
            memory_type (str): Type of memory bank ('negative' or 'positive')
            subsampling_share (float): Share of memory bank to keep after coreset subsampling
        """
        super(PatchCoreSingle, self).__init__()
        self.k_nearest = 3
        self.device = device
        self.memory_bank = None
        self.extracted_features = []
        self.memory_type = memory_type
        self.subsampling_share = subsampling_share

        # Load the appropriate backbone
        if backbone == 'resnet50':
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
        elif backbone == 'wide_resnet50_2':
            self.model = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Register hooks to extract features
        def hook(module, input, output):
            self.extracted_features.append(output)
            
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
        
        # Set model to evaluation mode and freeze parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, sample):
        """
        Extract features from the input image.
        
        Args:
            sample (torch.Tensor): Input image tensor of shape [B, C, H, W]
            
        Returns:
            list: List of feature maps from different layers
        """
        self.extracted_features = []
        _ = self.model(sample)
        return self.extracted_features

    def fit(self, dataloader: DataLoader):
        """
        Build memory bank from normal samples.
        
        Args:
            dataloader (DataLoader): DataLoader containing normal samples
        """
        memory_items = []
        
        for sample, _, _, _ in tqdm(dataloader, desc=f"Building {self.memory_type.capitalize()} Memory Bank"):
            sample = sample.to(self.device)
            features = self.process_features(sample)
            memory_items.extend(features)
            
        self.memory_bank = torch.cat(memory_items, dim=0)
        N, C, H, W = self.memory_bank.shape
        print(f"Memory Bank: {N} samples of shape {C}x{H}x{W}")

        # Apply coreset subsampling to reduce memory bank size
        target = max(1000, int(N * H * W * self.subsampling_share))
        
        self.memory_bank, indices = self.coreset_subsampling(
            self.memory_bank, target, epsilon=0.1, device=self.device
        )
        print(f"Memory Bank reduced to {len(indices)} patch embeddings")

    def process_features(self, sample):
        """
        Process features to create patch embeddings.
        
        Args:
            sample (torch.Tensor): Input image tensor
            
        Returns:
            list: List of processed feature tensors for each image in the batch
        """
        features = self(sample)
        
        # Apply local neighborhood aggregation to features
        layer2 = self.local_neighborhood_aggregation(features[0], p=3)
        layer3 = self.local_neighborhood_aggregation(features[1], p=3)
        
        # Upsample layer3 to match layer2 dimensions
        layer3 = self.bilinear_upsample(layer3, target_size=layer2.shape[2:])
        
        # Concatenate features from both layers
        combined = torch.cat([layer2, layer3], dim=1)
        
        # Return feature tensors for each image in batch
        return [combined[b].unsqueeze(0) for b in range(combined.shape[0])]

    
    def get_anomaly_score(self, sample):
        """
        Predict if a sample is anomalous and generate an anomaly map.
        
        Args:
            sample (torch.Tensor): Input image tensor
            
        Returns:
            float: Anomaly score
            torch.Tensor: Anomaly segmentation map
        """
        # Extract features
        feature_maps = self.process_features(sample)
        
        # Apply local neighborhood aggregation
        feature_maps = [self.local_neighborhood_aggregation(fm, p=3) for fm in feature_maps]
        
        # Ensure consistent dimensions
        feature_maps[1] = self.bilinear_upsample(feature_maps[1], target_size=feature_maps[0].shape[2:])
        
        # Concatenate features
        patch_collection = torch.cat(feature_maps, dim=1)

        # Store original dimensions
        C, H, W = patch_collection.shape[1:]
        
        # Reshape for distance calculation
        patch_collection = patch_collection.reshape(patch_collection.shape[1], -1).T

        # Calculate distances to memory bank
        distances = torch.cdist(patch_collection, self.memory_bank, p=2.0)
        dist_score, dist_score_idx = torch.min(distances, dim=1)
        
        # Find the patch with maximum distance (most anomalous)
        s_idx = torch.argmax(dist_score)
        s_star = dist_score[s_idx]
        m_test_star = patch_collection[s_idx]
        
        # Calculate neighborhood-based weight
        m_star = self.memory_bank[dist_score_idx[s_idx]].unsqueeze(0)
        knn_dists = torch.cdist(m_star, self.memory_bank, p=2.0)
        _, nn_idxs = knn_dists.topk(k=self.k_nearest, largest=False)
        m_neighborhood = self.memory_bank[nn_idxs[0, 1:]]
        
        w_denominator = torch.linalg.norm(m_test_star - m_neighborhood, dim=1)
        norm = torch.sqrt(torch.tensor(patch_collection.shape[1], device=self.device))
        
        # For negative bank, we want high weight when far from normal
        # For positive bank, we want high weight when close to defects (invert the formula)
        if self.memory_type == 'negative':
            w = 1 - (torch.exp(s_star / norm) / torch.sum(torch.exp(w_denominator / norm)))
        else:  # positive
            w = torch.exp(s_star / norm) / torch.sum(torch.exp(w_denominator / norm))
            
        # Final anomaly score
        s_final = w * s_star
        
        # Create distance map
        dist_map = dist_score.view(1, 1, H, W)
        
        return s_final, dist_map

    def predict(self, sample):
        """
        Predict if a sample is anomalous and generate an anomaly map.
        """
        # Get anomaly score and distance map
        score, dist_map = self.get_anomaly_score(sample)
        
        # Upsample to match input image size
        original_size = (sample.shape[2], sample.shape[3])
        segm_map = self.bilinear_upsample(dist_map, original_size)
        
        # Apply Gaussian blur for smoother visualization
        segm_map = self.gaussian_blur(segm_map)
        
        return score, segm_map

    # Helper methods
    def local_neighborhood_aggregation(self, feature_map, p=3):
        """
        Perform local neighborhood aggregation on feature maps.
        
        Args:
            feature_map (torch.Tensor): Feature map of shape [B, C, H, W]
            p (int): Size of neighborhood patch
            
        Returns:
            torch.Tensor: Aggregated feature map of same shape
        """
        B, C, H, W = feature_map.shape
        offset = p // 2
        
        # Pad feature map
        padded = F.pad(feature_map, (offset, offset, offset, offset), mode='reflect')
        
        # Extract neighborhoods
        neighborhoods = F.unfold(padded, kernel_size=p, stride=1)
        
        # Reshape for aggregation
        neighborhoods = neighborhoods.view(B, C, p*p, H*W).permute(0, 3, 1, 2).reshape(B*H*W, C, p, p)
        
        # Apply average pooling
        pooled = F.adaptive_avg_pool2d(neighborhoods, (1, 1))
        
        # Reshape back to original dimensions
        result = pooled.reshape(B, H*W, C, 1).squeeze(-1).permute(0, 2, 1).reshape(B, C, H, W)
        
        return result

    def bilinear_upsample(self, lower_spatial_block, target_size):
        """
        Upsample feature map to target size using bilinear interpolation.
        
        Args:
            lower_spatial_block (torch.Tensor): Feature map to upsample
            target_size (tuple): Target size (H, W)
            
        Returns:
            torch.Tensor: Upsampled feature map
        """
        if lower_spatial_block.shape[2:] == target_size:
            return lower_spatial_block
            
        return F.interpolate(lower_spatial_block, size=target_size, mode='bilinear', align_corners=False)

    def coreset_subsampling(self, embeddings, target_samples, epsilon=0.1, device=None, use_projection=True):
        """
        Perform coreset subsampling to reduce memory bank size.
        
        Args:
            embeddings (torch.Tensor): Embeddings to subsample
            target_samples (int): Number of samples to select
            epsilon (float): Error tolerance for random projection
            device (str): Device to use for computation
            use_projection (bool): Whether to use random projection
            
        Returns:
            torch.Tensor: Subsampled embeddings
            torch.Tensor: Indices of selected samples
        """
        if device is None:
            device = embeddings.device
            
        # Reshape embeddings if needed
        original_shape = embeddings.shape
        if len(original_shape) > 2:
            N, C, H, W = embeddings.shape
            reshaped_embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, C)
        else:
            reshaped_embeddings = embeddings
            
        n_samples, C = reshaped_embeddings.shape
        
        # Apply random projection if beneficial
        if use_projection and C > 10:
            d_star = 128
            if d_star < C:
                print(f"Projecting from {C} to {d_star} dimensions")
                embeddings_for_sampling = self.random_projection(reshaped_embeddings, d_star, epsilon)
            else:
                embeddings_for_sampling = reshaped_embeddings
        else:
            embeddings_for_sampling = reshaped_embeddings
            
        # Ensure we don't request more samples than available
        target_samples = min(target_samples, n_samples)
        
        # Use CoresetSampler for efficient subsampling
        sampler = CoresetSampler(n_samples=target_samples, device=str(device), tqdm_disable=False, verbose=1)
        selected_indices = sampler.sample(embeddings_for_sampling.cpu().numpy())
        
        return reshaped_embeddings[selected_indices], selected_indices

    def random_projection(self, embeddings, target_dim, epsilon=0.1, seed=0):
        """
        Apply random projection to reduce embedding dimensionality.
        
        Args:
            embeddings (torch.Tensor): Embeddings to project
            target_dim (int): Target dimensionality
            epsilon (float): Error tolerance
            seed (int): Random seed
            
        Returns:
            torch.Tensor: Projected embeddings
        """
        N, C = embeddings.shape
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Create random projection matrix
        projection_matrix = torch.randn(C, target_dim, device=embeddings.device)
        
        # Normalize projection matrix
        projection_matrix /= torch.sqrt(torch.sum(projection_matrix**2, dim=0, keepdim=True))
        
        # Apply projection
        return torch.matmul(embeddings, projection_matrix)

    def gaussian_blur(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur to a tensor.
        
        Args:
            img (torch.Tensor): Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Blurred tensor
        """
        # Create PIL transformations
        blur_kernel = ImageFilter.GaussianBlur(radius=4)
        tensor_to_pil = transforms.ToPILImage()
        pil_to_tensor = transforms.ToTensor()
        
        # Get maximum value for normalization
        max_value = img.max()
        
        # Convert to PIL, apply blur, convert back to tensor
        blurred_pil = tensor_to_pil(img[0] / max_value).filter(blur_kernel)
        blurred_map = pil_to_tensor(blurred_pil).to(img.device)
        
        return blurred_map * max_value
    

class PatchCoreDual:
    """
    Dual PatchCore model that combines scores from negative and positive memory banks.
    The final anomaly score is the ratio of negative score to positive score.
    """
    def __init__(self, device='cuda', backbone='wide_resnet50_2', negative_subsampling = 0.01, positive_subsampling = 0.10):
        self.negative_model = PatchCoreSingle(device, backbone, memory_type='negative', subsampling_share=negative_subsampling)
        self.positive_model = PatchCoreSingle(device, backbone, memory_type='positive', subsampling_share=positive_subsampling)
        self.device = device
    
    def fit(self, negative_dataloader, positive_dataloader):
        """Build both negative and positive memory banks."""
        print("Training negative model...")
        self.negative_model.fit(negative_dataloader)
        
        print("Training positive model...")
        self.positive_model.fit(positive_dataloader)
    
    def predict(self, sample):
        """
        Predict anomaly score and map for a sample using both memory banks.
        The final score is the ratio of negative to positive score (s- / s+).
        """
        # Get scores from both models
        neg_score, neg_map = self.negative_model.get_anomaly_score(sample)
        pos_score, pos_map = self.positive_model.get_anomaly_score(sample)
        
        # Calculate ratio (with small epsilon to avoid division by zero)
        epsilon = 1e-6
        
        # Higher ratio means more anomalous:
        # - Higher neg_score = farther from normal samples
        # - Lower pos_score = closer to defective samples
        ratio_score = neg_score / (pos_score + epsilon)
        
        # Create ratio map for segmentation
        H, W = neg_map.shape[2:]
        neg_map_flat = neg_map.view(-1)
        pos_map_flat = pos_map.view(-1)
        ratio_map_flat = neg_map_flat / (pos_map_flat + epsilon)
        ratio_map = ratio_map_flat.view(1, 1, H, W)
        
        # Upsample to original image size
        original_size = (sample.shape[2], sample.shape[3])
        segm_map = self.negative_model.bilinear_upsample(ratio_map, original_size)
        
        # Apply Gaussian blur
        segm_map = self.negative_model.gaussian_blur(segm_map)
        
        return ratio_score, segm_map
    
    def evaluate(self, test_dataloader):
        """
        Evaluate the model on a test dataset.
        
        Returns:
            float: Image-level AUROC
            float: Pixel-level AUROC
        """
        image_preds = []
        image_labels = []
        pixel_preds = []
        pixel_labels = []
        
        for sample, label, mask, _ in tqdm(test_dataloader, desc="Evaluating PatchCoreDual"):
            sample = sample.to(self.device)
            mask = mask.to(self.device)
            
            # Store ground truth
            image_labels.append(label.item())
            pixel_labels.extend((mask.flatten().cpu().numpy() > 0).astype(np.uint8))
            
            # Get predictions
            score, segm_map = self.predict(sample)
            
            # Store predictions
            image_preds.append(score.cpu().numpy())
            pixel_preds.extend(segm_map.flatten().cpu().numpy())
        
        # Calculate metrics
        image_auc = roc_auc_score(image_labels, image_preds)
        pixel_auc = roc_auc_score(pixel_labels, pixel_preds)
        
        return image_auc, pixel_auc

# FURTHER WORK:
# add set subsampling rates method