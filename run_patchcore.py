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

    # Defect crops dataset (pre-cropped)
    anomalous_dataset = KolektorSDD2Crops(
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
    print(f"Anomalous samples (Defect Cropped): {len(anomalous_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    # Use batch_size=1 for crops to handle variable sizes
    positive_loader = DataLoader(
        anomalous_dataset,
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
    # print("\n===== Experiment 1: Standard PatchCore (Negative Only) =====")
    # negative_model = PatchCoreSingle(
    #     device=device, 
    #     backbone=args.backbone, 
    #     memory_type='negative',
    #     subsampling_share=args.neg_subsampling
    # )
    # negative_model.fit(normal_loader)
    
    
    # neg_image_auc, neg_pixel_auc = negative_model.evaluate_single(test_loader)
    
    # print(f"Standard PatchCore Results:")
    # print(f"Image-level AUROC: {neg_image_auc:.4f}")
    # print(f"Pixel-level AUROC: {neg_pixel_auc:.4f}")
    
    # results["standard_patchcore"] = {
    #     "image_auc": neg_image_auc,
    #     "pixel_auc": neg_pixel_auc
    # }

    # Experiment Dual PatchCore with pre-cropped defects
    print("\n===== Dual PatchCore with Pre-Cropped Defects =====")
    dual_model = PatchCoreDual(
        device=device, 
        backbone=args.backbone,
        negative_subsampling=args.neg_subsampling,
        positive_subsampling=args.pos_subsampling
    )
    dual_model.fit(negative_dataloader=normal_loader, positive_dataloader=positive_loader)
    
    dual_image_auc, dual_pixel_auc = dual_model.evaluate(test_loader)
    
    print(f"Dual PatchCore Results:")
    print(f"Image-level AUROC: {dual_image_auc:.4f}")
    print(f"Pixel-level AUROC: {dual_pixel_auc:.4f}")
    
    results["dual_patchcore"] = {
        "image_auc": dual_image_auc,
        "pixel_auc": dual_pixel_auc
    }
    
    # Save results
    np.save(os.path.join(args.output_dir, "results.npy"), results)
    print(f"Results saved to {os.path.join(args.output_dir, 'results.npy')}")


if __name__ == "__main__":
    main()