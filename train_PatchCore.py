# 1 Pre - Trained ReNet50 Model
# 2 PatchCore Model
# 3 Train PatchCore Model
# 4 Evaluate PatchCore Model

import logging
import os
import pickle # to be used with mvtec images
import tqdm

import numpy as np
import torch
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader

from sklearn.random_projection import GaussianRandomProjection
from DiversitySampling.src.coreset import CoresetSampler

LOGGER = logging.getLogger(__name__)

class PatchCore(torch.nn.Module):
    def __init__(self):
        super(PatchCore, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extracted_features = []
        self.memory_bank = []
        

        def hook(module, input, output): # module: layer, input: input to the layer, output: output of the layer
            self.extracted_features.append(output)
    
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

        # Disable gradient computation
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, sample):

            _ = self.model(sample)

            return self.extracted_features


    def fit(self, train_dataloader : DataLoader):
        
        memory_items = []

        for sample, _ in tqdm(train_dataloader, total=len(train_dataloader)):
            # Extract features for the current batch
            self.extracted_features = []
            _ = self.model(sample)

            layer2_features = self.extracted_features[0] # Shape: [B, 512, H2, W2]
            layer3_features = self.extracted_features[1] # Shape: [B, 1024, H3, W3]

            layer2_processed = local_neighborhood_aggregation(layer2_features, p=3)
            layer3_processed = local_neighborhood_aggregation(layer3_features, p=3)

            # Upsample layer3 to match layer2's spatial dimensions
            layer3_upsampled = bilinear_upsample(layer3_processed, 
                                            target_size=layer2_processed.shape[2:])
            
            # Process each image in the batch
            B = layer2_processed.shape[0]
            for b in range(B):
                # Concatenate features from both layers along channel dimension
                combined = torch.cat([
                    layer2_processed[b],  # [512, H2, W2]
                    layer3_upsampled[b]   # [1024, H2, W2]
                ], dim=0)  # Result: [1536, H2, W2]
                
                # Resize to target dimension
                resized = bilinear_upsample(combined.unsqueeze(0), target_size=(128, 128)).squeeze(0)
                
                # Add to memory items collection
                memory_items.append(resized.unsqueeze(0))


        self.memory_bank = torch.cat(memory_items, dim=0)
        # [num_samples, 1536, 128, 128]
        
        N, C, H, W = self.memory_bank.shape
        print(f"Memory bank created with {N} samples of shape {C}x{H}x{W}")
        
        # Apply coreset sampling
        share = 0.01  # Keep 1% of all patches (adjust based on your needs)
        target_samples = max(1000, int(N * H * W * share))

        selected_embeddings, selected_indices = coreset_subsampling(
             embeddings=self.memory_bank,
             target_samples=target_samples,
             epsilon=0.1,
             device=self.device,
             use_projection=True
        )

        self.memory_bank = selected_embeddings

        print(f"Memory bank reduced from {N*H*W} to {len(selected_indices)} patch embeddings via coreset subsampling")

        # Save memory bank to disk
        # with open("memory_bank.pkl", "wb") as f:
        #     pickle.dump(self.memory_bank, f)

    def evaluate(self, test_dataloader : DataLoader):
        "anomaly detection"
        pass

    def predict(self, sample):
        pass


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
        d_star = calculate_projection_dim(n_samples, C, epsilon)
        # d_star = 128
        
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

if __name__ == '__main__':
    print("Testing PatchCore model")





# 1. we take a point (h,w) in the slice of the feature map
# 2. we extract the neighborhood of the point
# we average pool it to make it an aggregation of the neighborhood
# This acts like local smoothing, combining the features within a patch (or neighborhood)
# into one single vector of a fixed, predefined dimensionality d

# as a result the overall resolution of the feature map is preserved
# but resulting in a LOCALLY AWARE PATCH-FEATURE COLLECTION




