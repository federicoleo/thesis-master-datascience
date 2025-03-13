import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import random

from patchcore import PatchCoreSingle, PatchCoreDual
from data.ksdd2 import KolektorSDD2
from data.ksdd2_crops import KolektorSDD2Crops  # New class for pre-cropped defects


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="PatchCore Dual Experiments")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to KSDD2 dataset")
    parser.add_argument("--crops_path", type=str, required=True, help="Path to pre-cropped defects")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for normal training")
    parser.add_argument("--backbone", type=str, default="resnet50", 
                       choices=["resnet50", "wide_resnet50_2"], help="Backbone network")
    parser.add_argument("--scoring_method", type=str, default="ratio",
                       choices=["ratio", "weighted_linear", "weighted_ratio"], 
                       help="Method to combine scores")
    parser.add_argument("--neg_weight", type=float, default=0.7, help="Weight for negative score")
    parser.add_argument("--pos_weight", type=float, default=0.3, help="Weight for positive score")
    parser.add_argument("--add_augmented", default=False, action="store_true", help="Use augmented defects")
    parser.add_argument("--augmented_path", type=str, default=None, help="Path to augmented images")
    parser.add_argument("--neg_subsampling", type=float, default=0.01, help="Negative memory bank subsampling rate")
    parser.add_argument("--pos_subsampling", type=float, default=0.10, help="Positive memory bank subsampling rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    
    # Normal dataset (defect-free samples)
    normal_dataset = KolektorSDD2(
        dataroot=args.dataset_path,
        split='train',
        negative_only=True
    )
    dataset_size = len(normal_dataset)
    subset_size = int(0.6 * dataset_size)
    indices = random.sample(range(dataset_size), subset_size)

    normal_subset = Subset(normal_dataset, indices)

    # Defect crops dataset (pre-cropped)
    crops_dataset = KolektorSDD2Crops(
        crop_root=args.crops_path,
        add_augmented=args.add_augmented,
        augmented_crop_root=args.augmented_path
    )
    
    # Test dataset
    test_dataset = KolektorSDD2(
        dataroot=args.dataset_path,
        split='test'
    )
    
    print(f"Normal samples: {len(normal_dataset)}")
    print(f"Defect crops: {len(crops_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    normal_loader = DataLoader(
        normal_subset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    # Use batch_size=1 for crops to handle variable sizes
    crops_loader = DataLoader(
        crops_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # Run experiments
    results = {}
    # Experiment 1: Standard PatchCore (negative only)
    print("\n===== Experiment 1: Standard PatchCore (Negative Only) =====")
    negative_model = PatchCoreSingle(
        device=device, 
        backbone=args.backbone, 
        memory_type='negative',
        subsampling_share=args.neg_subsampling
    )
    negative_model.fit(normal_loader)
    
    # Evaluate standard model
    neg_image_preds = []
    neg_pixel_preds = []
    image_labels = []
    pixel_labels = []
    
    for sample, label, mask, _ in tqdm(test_loader, desc="Evaluating Standard PatchCore"):
        sample = sample.to(device)
        mask = mask.to(device)
        
        image_labels.append(label.item())
        pixel_labels.extend((mask.flatten().cpu().numpy() > 0).astype(np.uint8))
        
        neg_score, neg_map = negative_model.predict(sample)
        neg_image_preds.append(neg_score.cpu().numpy())
        neg_pixel_preds.extend(neg_map.flatten().cpu().numpy())
    
    neg_image_auc = roc_auc_score(image_labels, neg_image_preds)
    neg_pixel_auc = roc_auc_score(pixel_labels, neg_pixel_preds)

    neg_image_ap = average_precision_score(image_labels, neg_image_preds)
    neg_pixel_ap = average_precision_score(pixel_labels, neg_pixel_preds)
    
    print(f"Standard PatchCore Results:")
    print(f"Image-level AUROC: {neg_image_auc:.4f}")
    print(f"Pixel-level AUROC: {neg_pixel_auc:.4f}")
    print(f"Image-level AP: {neg_image_ap:.4f}")
    print(f"Pixel-level AP: {neg_pixel_ap:.4f}")
    
    results["standard_patchcore"] = {
        "image_auc": neg_image_auc,
        "pixel_auc": neg_pixel_auc,
        "image_ap": neg_image_auc,
        "pixel_ap": neg_pixel_ap
    }
    
    # Experiment 2: Dual PatchCore with pre-cropped defects
    print("\n===== Experiment 2: Dual PatchCore with Pre-Cropped Defects =====")
    dual_model = PatchCoreDual(
        device=device, 
        backbone=args.backbone,
        negative_subsampling=args.neg_subsampling,
        positive_subsampling=args.pos_subsampling
    )
    dual_model.fit(normal_loader, crops_loader)
    
    dual_image_auc, dual_pixel_auc, dual_image_ap, dual_pixel_ap, dual_f1_scores, dual_best_idx, dual_best_threshold = dual_model.evaluate(test_loader)
    
    print(f"Dual PatchCore Results:")
    print(f"Image-level AUROC: {dual_image_auc:.4f}")
    print(f"Pixel-level AUROC: {dual_pixel_auc:.4f}")
    print(f"Image-level AP: {dual_image_ap:.4f}")
    print(f"Pixel-level AP: {dual_pixel_ap:.4f}")
    print(f"Best threshold (F1): {dual_best_threshold:.4f}")
    print(f"Best F1 score: {dual_f1_scores[dual_best_idx]:.4f}")
    
    results["dual_patchcore"] = {
        "image_auc": dual_image_auc,
        "pixel_auc": dual_pixel_auc,
        "image_ap": dual_image_ap,
        "pixel_ap": dual_pixel_ap,
        "best_threshold": dual_best_threshold,
        "best_f1": dual_f1_scores[dual_best_idx]
    }
    
    # Save results
    np.save(os.path.join(args.output_dir, "results.npy"), results)
    print(f"Results saved to {os.path.join(args.output_dir, 'results.npy')}")
    
    # Optional: Visualization of a few test samples for qualitative comparison
    # fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # # Select a few test samples
    # vis_indices = [0, 10, 20]  # Adjust as needed
    
    # for i, idx in enumerate(vis_indices):
    #     sample, label, mask, _ = test_dataset[idx]
    #     sample = sample.unsqueeze(0).to(device)
        
    #     # Get predictions from both models
    #     _, neg_map = negative_model.predict(sample)
    #     _, dual_map = dual_model.predict(sample)
        
    #     # Denormalize sample for visualization
    #     denorm_sample = KolektorSDD2.denorm(sample.cpu())[0].permute(1, 2, 0).numpy()
        
    #     # Plot original image
    #     axes[i, 0].imshow(denorm_sample)
    #     axes[i, 0].set_title(f"Original (Label: {label})")
    #     axes[i, 0].axis('off')
        
    #     # Plot ground truth mask
    #     axes[i, 1].imshow(mask.squeeze().numpy(), cmap='jet')
    #     axes[i, 1].set_title("Ground Truth")
    #     axes[i, 1].axis('off')
        
    #     # Plot standard PatchCore heatmap
    #     axes[i, 2].imshow(neg_map.squeeze().cpu().numpy(), cmap='jet')
    #     axes[i, 2].set_title("Standard PatchCore")
    #     axes[i, 2].axis('off')
        
    #     # Plot dual PatchCore heatmap
    #     axes[i, 2].imshow(dual_map.squeeze().cpu().numpy(), cmap='jet')
    #     axes[i, 2].set_title("Dual PatchCore")
    #     axes[i, 2].axis('off')
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(args.output_dir, "visualization.png"))
    # print(f"Visualization saved to {os.path.join(args.output_dir, 'visualization.png')}")


if __name__ == "__main__":
    main()