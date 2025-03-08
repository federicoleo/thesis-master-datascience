# 1 Pre - Trained ReNet50 Model
# 2 PatchCore Model
# 3 Train PatchCore Model
# 4 Evaluate PatchCore Model

import logging
import wandb
import argparse
from tqdm import tqdm

import numpy as np
import torch

import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, Subset

from DiversitySampling.src.coreset import CoresetSampler
# from data.ksdd2 import KolektorSDD2
from data.mvtec import MVTecAD

from torchvision import transforms
from PIL import ImageFilter
from torch import tensor
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger(__name__)

class PatchCore(torch.nn.Module):
    def __init__(self, device='cuda', image_size=224):
        super(PatchCore, self).__init__()
        
        self.k_nearest = 3
        self.image_size = image_size
        self.device = device
        self.alpha = 0.7 # weight for negative distance (far from normal)
        self.beta = 0.3 # weight for positive distance (close to anomalous)
        
        self.neg_memory_bank = None
        self.pos_memory_bank = None
        self.extracted_features = []
        
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)

        def hook(module, input, output): # module: layer, input: input to the layer, output: output of the layer
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

    def fit(self, neg_dataloader : DataLoader, pos_dataloader : DataLoader):
        """Build Negative and Positive Memory Banks from provided data."""
        # Negative Memory Bank
        neg_memory_items = []
        for sample, _, _ in tqdm(neg_dataloader, desc="Building Negative Memory Bank"):
            sample = sample.to(self.device)
            features = self.process_features(sample)
            neg_memory_items.extend(features)
        self.neg_memory_bank = torch.cat(neg_memory_items, dim=0)
        N, C, H, W = self.neg_memory_bank.shape
        print(f"Negative Memory Bank: {N} samples of shape {C}x{H}x{W}")

        # Positive Memory Bank
        pos_memory_items = []
        for sample, _, _ in tqdm(pos_dataloader, desc="Building Positive Memory Bank"):
            sample = sample.to(self.device)
            features = self.process_features(sample)
            pos_memory_items.extend(features)

        self.pos_memory_bank = torch.cat(pos_memory_items, dim=0)
        N, C, H, W = self.pos_memory_bank.shape
        print(f"Positive Memory Bank: {N} samples of shape {C}x{H}x{W}")

        # Coreset Subsampling
        share = 0.01
        neg_target = max(1000, int(N * H * W * share))
        pos_target = max(1000, int(N * H * W * share))

        self.neg_memory_bank, neg_indices = coreset_subsampling(
            self.neg_memory_bank, neg_target, epsilon=0.1, device=self.device
        )
        self.pos_memory_bank, pos_indices = coreset_subsampling(
            self.pos_memory_bank, pos_target, epsilon=0.1, device=self.device
        )
        print(f"Negative Memory Bank reduced to {len(neg_indices)} patch embeddings")
        print(f"Positive Memory Bank reduced to {len(pos_indices)} patch embeddings")

    def process_features (self, sample):
        features = self(sample)
        layer2 = local_neighborhood_aggregation(features[0], p=3)
        layer3 = local_neighborhood_aggregation(features[1], p=3)
        layer3 = bilinear_upsample(layer3, target_size=layer2.shape[2:])
        combined = torch.cat([layer2, layer3], dim=1)  # [B, 1536, H, W]
        resized = bilinear_upsample(combined, target_size=(28, 28))  # [B, 1536, 28, 28]
        return [resized[b].unsqueeze(0) for b in range(resized.shape[0])]

    def evaluate(self, test_dataloader: DataLoader):
        """
            Compute anomaly detection score and relative segmentation map for
            each test sample. Returns the ROC AUC computed from predictions scores.

            Returns:
            - image-level ROC-AUC score
            - pixel-level ROC-AUC score
        """

        image_preds = []
        image_labels = []
        pixel_preds = []
        pixel_labels = []

        for sample, mask, label in tqdm(test_dataloader, desc="Evaluating"):
            sample = sample.to(self.device)
            mask = mask.to(self.device)

            image_labels.append(label.numpy())
            pixel_labels.extend((mask.flatten().cpu().numpy() > 0).astype(np.uint8))

            score, segm_map = self.predict(sample)  # Anomaly Detection
            image_preds.append(score.cpu().numpy())
            pixel_preds.extend(segm_map.flatten().cpu().numpy())

        image_labels = np.array(image_labels)
        image_preds = np.array(image_preds)
        # Compute ROC AUC for prediction scores
        image_level_rocauc = roc_auc_score(image_labels, image_preds)
        pixel_level_rocauc = roc_auc_score(pixel_labels, pixel_preds)

        return image_level_rocauc, pixel_level_rocauc

    def predict(self, sample):
        """Compute anomaly score using d_neg / d_pos."""
        # Get features through forward method
        feature_maps = self(sample)
        # Local Neighborhood Aggregation
        feature_maps = [local_neighborhood_aggregation(fm, p=3) for fm in feature_maps]
        # Matching Feature Dimensions
        feature_maps[1] = bilinear_upsample(feature_maps[1], target_size=feature_maps[0].shape[2:])
        # Concatenation and Resizing
        patch_collection = torch.cat(feature_maps, dim=1) # [B, 1536, H, W]
        patch_collection = patch_collection.reshape(patch_collection.shape[1], -1).T # [H*W, 1536]

        # Calculate distances to both memory banks
        neg_distances = torch.cdist(patch_collection, self.neg_memory_bank, p=2.0)
        neg_dist_score, neg_dist_score_idx = torch.min(neg_distances, dim=1)
        
        pos_distances = torch.cdist(patch_collection, self.pos_memory_bank, p=2.0)
        pos_dist_score, pos_dist_score_idx = torch.min(pos_distances, dim=1)
        
        # Calculate initial ratio score (with epsilon to avoid division by zero)
        epsilon = 1e-6
        ratio_score = neg_dist_score / (pos_dist_score + epsilon)
        
        # Find patch with highest ratio score
        s_idx = torch.argmax(ratio_score)
        s_star = ratio_score[s_idx]
        
        # Extract the most anomalous test patch
        m_test_star = patch_collection[s_idx]
        
        # NEGATIVE MEMORY BANK WEIGHTING
        # For negative weighting: HIGHER is better (far from normal)
        m_neg_star = self.neg_memory_bank[neg_dist_score_idx[s_idx]].unsqueeze(0)
        
        # Find k-nearest neighbors in negative memory bank
        knn_neg_dists = torch.cdist(m_neg_star, self.neg_memory_bank, p=2.0)
        _, nn_neg_idxs = knn_neg_dists.topk(k=self.k_nearest, largest=False)
        
        m_neg_neighborhood = self.neg_memory_bank[nn_neg_idxs[0, 1:]]
        w_neg_denominator = torch.linalg.norm(m_test_star - m_neg_neighborhood, dim=1)
        
        # Normalization factor
        norm = torch.sqrt(torch.tensor(patch_collection.shape[1]))
        
        # Compute negative weight - KEEP as high distance is good
        w_neg = 1 - (torch.exp(neg_dist_score[s_idx] / norm) / 
                    torch.sum(torch.exp(w_neg_denominator / norm)))
        
        # POSITIVE MEMORY BANK WEIGHTING
        # For positive weighting: LOWER is better (close to anomalies)
        m_pos_star = self.pos_memory_bank[pos_dist_score_idx[s_idx]].unsqueeze(0)
        
        # Find k-nearest neighbors in positive memory bank
        knn_pos_dists = torch.cdist(m_pos_star, self.pos_memory_bank, p=2.0)
        _, nn_pos_idxs = knn_pos_dists.topk(k=self.k_nearest, largest=False)
        
        m_pos_neighborhood = self.pos_memory_bank[nn_pos_idxs[0, 1:]]
        w_pos_denominator = torch.linalg.norm(m_test_star - m_pos_neighborhood, dim=1)
        
        # Compute positive weight - INVERT the formula since LOW distance is good
        # The original formula gives low weight for low distance, we want the opposite
        w_pos = (torch.exp(pos_dist_score[s_idx] / norm) / 
                torch.sum(torch.exp(w_pos_denominator / norm)))
        
        # Final weighted anomaly score:
        # Higher w_neg (far from normal) * Higher w_pos (close to anomalous) * Higher ratio
        s_final = (self.alpha * w_neg + self.beta * w_pos) * s_star
        
        # Create segmentation map using the ratio scores
        fmap_size = feature_maps[0].shape[-2:]
        segm_map = ratio_score.view(1, 1, *fmap_size)
        segm_map = bilinear_upsample(segm_map, (self.image_size, self.image_size))
        segm_map = gaussian_blur(segm_map)
        
        return s_final, segm_map


def local_neighborhood_aggregation(feature_map, p=3):
    """
    Performs local neighborhood aggregation by:
    1. Expanding each position to its pxp neighborhood
    2. Applying adaptive average pooling to each neighborhood
    3. Using the pooled result as the new feature at that position
    
    Args:
        feature_map (torch.Tensor): Input tensor of shape (B, C, H, W)
        p (int): Size of the neighborhood patch (default: 3 for 3x3)
    
    Returns:
        torch.Tensor: Reconstructed feature map with same shape as input
    """
    B, C, H, W = feature_map.shape
    offset = p // 2
    
    # Create output tensor
    reconstructed = torch.zeros_like(feature_map)
    
    # Pad input to handle border cases - this makes extraction easier
    padded = F.pad(feature_map, (offset, offset, offset, offset), mode='reflect')
    
    # Extract all neighborhoods at once using unfold
    # This gives us tensor of shape [B, C*p*p, H*W]
    neighborhoods = F.unfold(padded, kernel_size=p, stride=1)
    
    # Reshape to [B*H*W, C, p, p] to prepare for adaptive_avg_pool2d
    # First reshape to [B, C, p*p, H*W]
    neighborhoods = neighborhoods.view(B, C, p*p, H*W)
    # Then transpose and reshape to get [B*H*W, C, p, p]
    neighborhoods = neighborhoods.permute(0, 3, 1, 2).reshape(B*H*W, C, p, p)
    
    # Apply adaptive_avg_pool2d to get [B*H*W, C, 1, 1]
    pooled = F.adaptive_avg_pool2d(neighborhoods, (1, 1))
    
    # Reshape back to [B, H, W, C]
    pooled = pooled.reshape(B, H*W, C, 1).squeeze(-1).permute(0, 2, 1)
    
    # Finally reshape to [B, C, H, W]
    reconstructed = pooled.reshape(B, C, H, W)
    
    return reconstructed


def bilinear_upsample(lower_spatial_block, target_size):
    """
    Upsamples the given feature map to the target spatial size using bilinear interpolation.

    Args:
        lower_spatial_block (torch.Tensor): Input tensor of shape (B, C, H, W)
        target_size (tuple): Target spatial size (H_out, W_out)

    Returns:
        torch.Tensor: Upsampled feature map with shape (B, C, H_out, W_out)
    """
    if lower_spatial_block.shape[2:] == target_size:
        return lower_spatial_block
    else:
        return F.interpolate(lower_spatial_block, size=target_size, mode='bilinear', align_corners=False)


def random_projection(embeddings, target_dim, epsilon=0.1, seed=0):
    """
    Applies random projection to reduce embedding dimensionality using Johnson-Lindenstrauss lemma.
    
    Args:
        embeddings (torch.Tensor): Input embeddings of shape [N, C]
        target_dim (int): Target dimension to project to
        epsilon (float): Error tolerance for distance preservation
        seed (int): Random seed for reproducibility
        
    Returns:
        torch.Tensor: Projected embeddings of shape [N, target_dim]
    """
    # Get original dimensions
    N, C = embeddings.shape
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Create random projection matrix
    projection_matrix = torch.randn(C, target_dim, device=embeddings.device)
    
    
    # Normalize columns to ensure distance preservation properties
    # Normalize the projection matrix (l2 norm with the sqrt)
    projection_matrix = projection_matrix / torch.sqrt(torch.sum(projection_matrix**2, dim=0, keepdim=True))
    
    # Apply projection
    # we multiply our patch embeddings with the projection matrix
    # relative distances between points are approximately preserved (that's the key benefit of Johnson-Lindenstrauss projection)
    projected = torch.matmul(embeddings, projection_matrix)
    
    return projected


def calculate_projection_dim(n_samples, original_dim, epsilon=0.1):
    """
    Calculate the minimum dimension needed for projection according to Johnson-Lindenstrauss lemma.
    
    Args:
        n_samples (int): Number of samples in the dataset
        original_dim (int): Original feature dimension
        epsilon (float): Desired error bound (typically 0.1-0.3)
        
    Returns:
        int: Target dimension for projection
    """
    # JL lemma formula for the minimum dimension
    # Calculate reduced dimension d* based on Johnson-Lindenstrauss lemma
    # The formula is derived from ensuring epsilon-distortion with high probability
    jl_dim = int(4 * np.log(n_samples) / (epsilon**2/2 - epsilon**3/3))
    
    # Can't project to higher dimension than original
    return min(original_dim, jl_dim)


def coreset_subsampling(embeddings, target_samples, epsilon=0.1, device=None, use_projection=True):
    """
    Applies coreset subsampling to select representative embeddings.
    
    Args:
        embeddings (torch.Tensor): Input embeddings of shape [N, C] or [N, C, H, W]
        target_samples (int): Number of samples to select
        epsilon (float): Error tolerance for random projection
        device (str): Device to use for computation
        use_projection (bool): Whether to apply random projection
        
    Returns:
        torch.Tensor: Selected embeddings
        list: Indices of selected embeddings
    """
    
    # Determine device
    if device is None:
        device = embeddings.device
    
    # Handle different input shapes
    original_shape = embeddings.shape
    if len(original_shape) > 2:
        # For feature maps [N, C, H, W], reshape to [N*H*W, C]
        N, C, H, W = embeddings.shape
        reshaped_embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, C)
    else:
        # Already in correct shape [N, C]
        reshaped_embeddings = embeddings
    
    n_samples, C = reshaped_embeddings.shape
    
    # Step 1: Apply random projection if requested and beneficial
    if use_projection and C > 10:  # Only project if dimension is substantial
        # Calculate target projection dimension
        # d_star = calculate_projection_dim(n_samples, C, epsilon)
        d_star = 128
        
        # Apply projection if it reduces dimension
        if d_star < C:
            print(f"Projecting from {C} to {d_star} dimensions")
            embeddings_for_sampling = random_projection(reshaped_embeddings, d_star, epsilon)
        else:
            print(f"Skipping projection as calculated dimension {d_star} ≥ original {C}")
            embeddings_for_sampling = reshaped_embeddings
    else:
        # Skip projection
        embeddings_for_sampling = reshaped_embeddings
    
    # Step 2: Apply coreset sampling
    # Ensure we don't try to select more samples than available
    target_samples = min(target_samples, n_samples)
    
    # Initialize and run sampler
    sampler = CoresetSampler(
        n_samples=target_samples,
        device=str(device),
        tqdm_disable=False,
        verbose=1
    )
    
    # Get indices of selected samples
    selected_indices = sampler.sample(embeddings_for_sampling.cpu().numpy())
    
    # Step 3: Return selected embeddings in original space
    selected_embeddings = reshaped_embeddings[selected_indices]
    
    print(f"Reduced from {n_samples} to {len(selected_indices)} samples")
    
    return selected_embeddings, selected_indices

def gaussian_blur(img: tensor) -> tensor:
    """
        Apply a gaussian smoothing with sigma = 4 over the input image.
    """
    # Setup
    blur_kernel = ImageFilter.GaussianBlur(radius=4)
    tensor_to_pil = transforms.ToPILImage()
    pil_to_tensor = transforms.ToTensor()
    # Smoothing
    max_value = img.max()   # Maximum value of all elements in the image tensor
    blurred_pil = tensor_to_pil(img[0] / max_value).filter(blur_kernel)
    blurred_map = pil_to_tensor(blurred_pil).to(img.device)

    return blurred_map * max_value

def main(args):
    # Set the device.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set the seed for reproducibility.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    add_augmented = args.add_augmented
    num_augmented = args.num_augmented
    logging = args.logging

    run_name = f'PatchCore-add_augmented_{add_augmented}-num_augmented_{num_augmented}-bs_{args.batch_size}'
    tags = [f'{num_augmented}augmented']

    if add_augmented:
        tags.append('augmented')
    else:
        tags.append('not_augmented')
    
    if logging:
        # Start a new wandb run to track this script.
        wandb.init(
            name=run_name,
            config=args,
            tags=tags
        )

    # Transform to match PatchCore official preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    categories = MVTecAD.CATEGORIES
    image_aucs = {}
    pixel_aucs = {}

    for category in categories:
        print(f"\nProcessing category: {category}")
        neg_train_data = MVTecAD(
            dataroot=args.dataset_path,
            split='train',
            category=category,
            negative_only=True,
            memory_bank_type='negative',
            transform=transform
        )
        pos_train_data = MVTecAD(
            dataroot=args.dataset_path,
            split='train',
            category=category,
            add_augmented=add_augmented,
            num_augmented=num_augmented,
            memory_bank_type='positive',
            transform=transform
        )
        test_data = MVTecAD(
            dataroot=args.dataset_path,
            split='test',
            category=category,
            memory_bank_type='test',
            transform=transform
        )
        print(f"{category} - Neg: {len(neg_train_data)}, Pos: {len(pos_train_data)}, Test: {len(test_data)}")

        neg_loader = DataLoader(neg_train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        pos_loader = DataLoader(pos_train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

        model = PatchCore(device=device, image_size=224)
        print(f"Building memory banks for {category} on {device} [...]")
        model.fit(neg_loader, pos_loader)
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

    print("PatchCore evaluation finished.")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PatchCore with Dual Memory Banks for MVTec')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoaders')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoaders')
    parser.add_argument('--dataset', type=str, choices=['mvtec'], default='mvtec', help='Dataset to use (only mvtec supported)')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to MVTec dataset')
    parser.add_argument('--add_augmented', action='store_true', help='Add augmented images to Positive bank')
    parser.add_argument('--num_augmented', type=int, default=150, help='Number of augmented images per category')
    parser.add_argument('--logging', action='store_true', help='Log stats to wandb')

    args = parser.parse_args()
    if args.dataset != 'mvtec':
        raise ValueError("This implementation supports only 'mvtec' dataset for now.")
    main(args)