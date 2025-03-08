import os
import cv2
import numpy as np
from glob import glob
import pandas as pd
import argparse
from tqdm import tqdm
import shutil

mvtec_resolutions = {
    "hazelnut": (1024, 1024),
    "screw": (1024, 1024),
    "pill": (800, 800),
    "carpet": (1024, 1024),
    "zipper": (1024, 1024),
    "cable": (1024, 1024),
    "leather": (1024, 1024),
    "capsule": (1000, 1000),
    "tile": (840, 840),
    "grid": (1024, 1024),
    "metal_nut": (700, 700),
    "wood": (1024, 1024),
    "transistor": (1024, 1024),
    "bottle": (900, 900),
    "toothbrush": (1024, 1024)
    }


def process_mvtec(src_dir, dst_dir, categories=None):
    
    # Get all MVTec categories if not specified
    if categories is None:
        categories = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    for category in categories:
        print(f"Processing category: {category}")
        src_category_dir = os.path.join(src_dir, category)
        # dst_category_dir = os.path.join(dst_dir, category)
        # os.makedirs(dst_category_dir, exist_ok=True)

        splits = ['train', 'test', 'ground_truth']
        for split in splits:
            src_split_dir = os.path.join(src_category_dir, split)
            #dst_split_dir = os.path.join(dst_category_dir, "test") if split == "ground_truth" else os.path.join(dst_category_dir, split)
            dst_split_dir = os.path.join(dst_dir, "test" if split == "ground_truth" else split)
            os.makedirs(dst_split_dir, exist_ok=True)
            
            for defect_type in os.listdir(src_split_dir):
                src_defect_dir = os.path.join(src_split_dir, defect_type)
                dst_defect_dir = os.path.join(dst_split_dir, category, defect_type)
                os.makedirs(dst_defect_dir, exist_ok=True)
                all_imgs = [img for img in os.listdir(src_defect_dir) if img.endswith(".png") and not img.endswith("_GT.png")]
                for img in tqdm(all_imgs, desc=f"Reshaping {category} images for {split} split, {defect_type} defect", unit="file", total=len(all_imgs)):
                    if split == 'ground_truth':
                        src_path = os.path.join(src_defect_dir, img)
                        mask_name = img.replace("_mask.png", "_GT.png")  # e.g., 000_GT.png
                        dst_path = os.path.join(dst_defect_dir, mask_name)
                        img_data = cv2.imread(src_path)
                        img_resized = cv2.resize(img_data, mvtec_resolutions[category], interpolation=cv2.INTER_NEAREST)
                        cv2.imwrite(dst_path, img_resized)
                        print(f"Moved mask: {src_path} -> {dst_path}")
                    else:
                        img_path = os.path.join(src_defect_dir, img)
                        img_out_path = os.path.join(dst_defect_dir, img)
                        img_data = cv2.imread(img_path)
                        img_resized = cv2.resize(img_data, mvtec_resolutions[category])
                        cv2.imwrite(img_out_path, img_resized)

                        # Create a negative mask for "good" (normal) samples
                        if split == "train" or (split == "test" and defect_type == "good"):
                            # Create black mask (all zeros) with the same resolution
                            mask_name = os.path.splitext(img)[0] + "_GT.png"
                            mask_path = os.path.join(dst_defect_dir, mask_name)
                            height, width = mvtec_resolutions[category]
                            mask = np.zeros((height, width), dtype=np.uint8)
                            cv2.imwrite(mask_path, mask)
                            print(f"Created mask: {mask_path}")
                            
        print(f"Processed category: {category}")


def make_csv(dst_dir):
    splits = ['train', 'test']
    for split in splits:
        imgs_dict = {"path": [], "label": []}
        split_dir = os.path.join(dst_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping {split}")
            continue
        
        for category in os.listdir(split_dir):
            category_split_dir = os.path.join(split_dir, category)
            if not os.path.isdir(category_split_dir):
                continue
            
            for defect_type in os.listdir(category_split_dir):
                img_dir = os.path.join(category_split_dir, defect_type)
                if not os.path.isdir(img_dir):
                    continue
                all_imgs = [img for img in os.listdir(img_dir) if img.endswith(".png") and not img.endswith("_GT.png")]
                for img in tqdm(all_imgs, desc=f"Processing {split}/{category}/{defect_type}", unit="file", total=len(all_imgs)):
                    rel_path = os.path.join(category, defect_type, img)  # e.g., bottle/good/000.png
                    mask_name = img.replace(".png", "_GT.png")
                    mask_path = os.path.join(img_dir, mask_name)
                    
                    imgs_dict["path"].append(rel_path)
                    if split == "train":
                        imgs_dict["label"].append("negative")  # All train images are negative
                    elif split == "test":
                        if os.path.exists(mask_path):
                            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            if np.any(mask > 0):
                                imgs_dict["label"].append("positive")
                            else:
                                imgs_dict["label"].append("negative")
                        else:
                            print(f"Warning: Mask not found for {rel_path}")
                            imgs_dict["label"].append("negative")  # Default if mask missing
        
        # Save CSV for this split
        df = pd.DataFrame(imgs_dict)
        csv_path = os.path.join(dst_dir, f"{split}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {split}.csv with {len(df)} entries")

def main(args):
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    categories = args.categories.split(",") if args.categories else None
    

    # Create destination directory
    print(f"Processing MVTec dataset from {src_dir} to {dst_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    
    # Reshape images
    process_mvtec(src_dir, dst_dir, categories)
    
    # Create CSV files
    make_csv(dst_dir)

    print(f"\nPreprocessing complete! Processed data is in {dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MVTec Anomaly Detection dataset")
    parser.add_argument("--src_dir", type=str, required=True, help="Path to the MVTec dataset root")
    parser.add_argument("--dst_dir", type=str, required=True, help="Path to the destination directory")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated list of categories to process (default: all)")
    args = parser.parse_args()
    main(args)