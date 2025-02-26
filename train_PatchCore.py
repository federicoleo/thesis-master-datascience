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
        patch_embeddings = self.memory_bank.permute(0, 2, 3, 1).reshape(-1, C)  # [N*H*W, C]
    
        # Apply random projection for dimensionality reduction (Johnson-Lindenstrauss)
        n_samples = patch_embeddings.shape[0]
        epsilon = 0.1  # Error tolerance
        
        # Calculate reduced dimension d* based on Johnson-Lindenstrauss lemma
        # The formula is derived from ensuring epsilon-distortion with high probability
        d_star = min(
            C,  # Can't project to higher dimension
            int(4 * np.log(n_samples) / (epsilon**2/2 - epsilon**3/3))  # ~O(log N)
        )
        # d_star = 128  # Set a fixed value for testing
        print(f"Projecting from {C} to {d_star} dimensions")

        # Create random projection matrix
        torch.manual_seed(0)  # For reproducibility
        
        projection_matrix = torch.randn(C, d_star).to(patch_embeddings.device)
        #  matrix [C, d_star] with random values from a standard normal distribution
        
        projection_matrix = projection_matrix / torch.sqrt(torch.sum(projection_matrix**2, dim=0, keepdim=True))
        # Normalize the projection matrix (l2 norm with the sqrt)

        # Apply projection
        # We multiply our patch embeddings with the projection matrix
        projected_embeddings = torch.matmul(patch_embeddings, projection_matrix)  # [N*H*W, d_star]
        # we multiply our patch embeddings with the projection matrix
        # relative distances between points are approximately preserved (that's the key benefit of Johnson-Lindenstrauss projection)
        
        # Apply coreset sampling
        share = 0.01  # Keep 1% of all patches (adjust based on your needs)
        target_samples = max(1000, int(N * H * W * share))
        
        # Initialize and run CoresetSampler
        sampler = CoresetSampler(
            n_samples=target_samples,
            device=patch_embeddings.device,
            tqdm_disable=False,
            verbose=1
        )
        
        # Sample the most representative patches
        coreset_idx = sampler.sample(projected_embeddings.cpu().numpy())
        
        # Keep only the sampled patch embeddings from the original space (not the projected space)
        self.memory_bank = patch_embeddings[coreset_idx]  # [target_samples, C]
        
        print(f"Memory bank reduced from {N*H*W} to {len(coreset_idx)} patch embeddings")

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

def random_linear_projections(memory_bank: torch.Tensor, d_star: int) -> torch.Tensor:
    """
    Applies JL projection to feature dimension while preserving spatial structure.
    Input: [N, 1536, H, W]
    Output: [N, d_star, H, W]
    """
    device = memory_bank.device
    N, C, H, W = memory_bank.shape
    
    # Generate projection matrix once (shared across all spatial positions)
    projection_matrix = torch.randn(C, d_star, device=device) / np.sqrt(d_star)
    
    # Project features at every spatial position
    projected = torch.einsum('nchw,cd->ndhw', memory_bank, projection_matrix)
    return projected  # Shape: [N, d_star, H, W]



def coreset_subsampling(memory_bank, coreset_target_size):
    # greedy approach
    B, C, H, W = memory_bank.shape
    
    features = memory_bank.view(C, H * W).T

    features_np = features.cpu().detach().numpy()

    N, _ = features_np.shape
    num_samples = coreset_target_size*N

    selected_indices = [0]  # Start with the first point
    distances = np.full(N, np.inf)

    # Greedily select points that maximize the minimum distance to the current centers
    for _ in range(1, num_samples):
        last_center = features_np[selected_indices[-1]]
        # Compute Euclidean distances from the last center to all points
        dist_to_last = np.linalg.norm(features_np - last_center, axis=1)
        # Update each point's distance to its closest center so far
        distances = np.minimum(distances, dist_to_last)
        # Select the point with the maximum distance to its nearest center
        next_index = np.argmax(distances)
        selected_indices.append(next_index)

    # Retrieve the selected feature vectors and convert back to a torch tensor
    centers_np = features_np[selected_indices]
    centers = torch.tensor(centers_np, device=memory_bank.device, dtype=memory_bank.dtype)
    return centers




if __name__ == '__main__':
    print("Testing PatchCore model")





# 1. we take a point (h,w) in the slice of the feature map
# 2. we extract the neighborhood of the point
# we average pool it to make it an aggregation of the neighborhood
# This acts like local smoothing, combining the features within a patch (or neighborhood)
# into one single vector of a fixed, predefined dimensionality d

# as a result the overall resolution of the feature map is preserved
# but resulting in a LOCALLY AWARE PATCH-FEATURE COLLECTION




