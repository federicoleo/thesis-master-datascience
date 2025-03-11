import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

from patchcore import PatchCoreSingle
from data.ksdd2 import KolektorSDD2

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Standard PatchCore Experiment")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to KSDD2 dataset")
    parser.add_argument("--output_dir", type=str, default="./results_standard", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--backbone", type=str, default="resnet50", 
                       choices=["resnet50", "wide_resnet50_2"], help="Backbone network")
    parser.add_argument("--subsampling", type=float, default=0.01, help="Memory bank subsampling rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select device
    device = "cuda" if torch.cuda.is_available() else "mps"
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    
    # Training dataset (defect-free samples)
    train_dataset = KolektorSDD2(
        dataroot=args.dataset_path,
        split='train',
        negative_only=True
    )
    
    # Test dataset
    test_dataset = KolektorSDD2(
        dataroot=args.dataset_path,
        split='test'
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize and train PatchCore
    print("Training Standard PatchCore...")
    model = PatchCoreSingle(
        device=device, 
        backbone=args.backbone, 
        memory_type='negative',
        subsampling_share=args.subsampling
    )
    model.fit(train_loader)
    
    # Evaluate on test set
    print("Evaluating...")
    image_preds = []
    image_labels = []
    pixel_preds = []
    pixel_labels = []
    
    for sample, label, mask, _ in tqdm(test_loader):
        sample = sample.to(device)
        mask = mask.to(device)
        
        # Store ground truth
        image_labels.append(label.item())
        pixel_labels.extend((mask.flatten().cpu().numpy() > 0).astype(np.uint8))
        
        # Get predictions
        score, segm_map = model.predict(sample)
        
        # Store predictions
        image_preds.append(score.cpu().numpy())
        pixel_preds.extend(segm_map.flatten().cpu().numpy())
    
    # Calculate metrics
    image_auc = roc_auc_score(image_labels, image_preds)
    pixel_auc = roc_auc_score(pixel_labels, pixel_preds)
    
    # Calculate Average Precision
    image_ap = average_precision_score(image_labels, image_preds)
    pixel_ap = average_precision_score(pixel_labels, pixel_preds)
    
    # Find optimal threshold based on F1 score
    precisions, recalls, thresholds = precision_recall_curve(image_labels, image_preds)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0
    
    # Print results
    print(f"\nStandard PatchCore Results:")
    print(f"Image-level AUROC: {image_auc:.4f}")
    print(f"Pixel-level AUROC: {pixel_auc:.4f}")
    print(f"Image-level AP: {image_ap:.4f}")
    print(f"Pixel-level AP: {pixel_ap:.4f}")
    print(f"Best threshold (F1): {best_threshold:.4f}")
    print(f"Best F1 score: {f1_scores[best_idx]:.4f}")
    
    # Save results
    results = {
        "image_auc": image_auc,
        "pixel_auc": pixel_auc,
        "image_ap": image_ap,
        "pixel_ap": pixel_ap,
        "best_threshold": best_threshold,
        "best_f1": f1_scores[best_idx]
    }
    np.save(os.path.join(args.output_dir, "results.npy"), results)
    print(f"Results saved to {os.path.join(args.output_dir, 'results.npy')}")

if __name__ == "__main__":
    main()