#!/usr/bin/env python
# -*- coding: utf-8 -*-

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
import torch
import os
import random
import numpy as np
import argparse
from glob import glob
from PIL import Image

# Category-specific prompts for MVTec defects
CATEGORY_PROMPTS = {
    "bottle": ["cracks on glass surface", "dents on bottle", "scratches on cap"],
    "cable": ["scratches on copper wire", "frayed insulation", "burn marks"],
    "capsule": ["punctures on pill surface", "discoloration spots", "cracks on coating"],
    "carpet": ["stains on fabric", "burn marks", "tears in weave"],
    "grid": ["bent metal bars", "rust spots", "broken grid lines"],
    "hazelnut": ["cracks on nut shell", "dark mold spots", "scratches on surface"],
    "leather": ["scratches on leather", "stains on surface", "tears in material"],
    "metal_nut": ["scratches on metal", "dents on nut", "rust patches"],
    "pill": ["discoloration on pill", "cracks on surface", "puncture marks"],
    "screw": ["scratches on metal", "bent screw thread", "rust on surface"],
    "tile": ["cracks on ceramic", "stains on tile", "chipped edges"],
    "toothbrush": ["bent bristles", "stains on handle", "broken bristle tips"],
    "transistor": ["bent metal pins", "burn marks on chip", "scratches on surface"],
    "wood": ["scratches on wood grain", "dark knots", "cracks in surface"],
    "zipper": ["broken teeth", "scratches on metal", "frayed fabric edges"]
}

# Generic negative prompt (can be made category-specific if needed)
NEGATIVE_PROMPT = "smooth intact surface, clean, plain, uniform"

def get_normal_images(category_dir):
    """Get list of normal (good) images for a category."""
    return glob(os.path.join(category_dir, "good", "*.png"))

def get_anomaly_masks(category_dir):
    """Get list of anomaly masks from test split, excluding 'good'."""
    mask_paths = []
    for defect_type in os.listdir(category_dir):
        if defect_type != "good":  # Exclude normal samples
            mask_paths.extend(glob(os.path.join(category_dir, defect_type, "*_GT.png")))
    return mask_paths

def threshold_mask(mask):
    """Ensure mask is binary (0 or 255) after resizing"""
    mask_np = np.array(mask)
    mask_np = (mask_np > 127).astype(np.uint8) * 255
    return Image.fromarray(mask_np)

def main(args):
    ### ARGUMENTS
    src_dir = args.src_dir  # Root directory of MVTec dataset (e.g., "data/mvtec_preprocessed")
    imgs_per_prompt = args.imgs_per_prompt  # Number of images per prompt per category
    seed = args.seed

    # Hyperparameters
    num_inference_steps = 30
    guidance_scale = 20.0
    strength = 0.99
    padding_mask_crop = 2
    RES = (256, 256)  # Stored as 256x256 for MVTecAD compatibility, cropped to 224x224 later
    TARGET = (1024, 1024)  # Inpainting resolution for SDXL quality

    ### MAIN
    # Device setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # MPS may not work with diffusers
    print("Running on device: ", device)

    # Seed everything for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    # Load SDXL Inpainting model
    model = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    print(f"Loading model {model}")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model, torch_dtype=torch.float16, variant="fp16"
    ).to(device)

    # Process each category
    categories = sorted(os.listdir(os.path.join(src_dir, "train")))
    for category in categories:
        print(f"\nProcessing category: {category}")
        
        # Define paths
        train_category_dir = os.path.join(src_dir, "train", category)
        test_category_dir = os.path.join(src_dir, "test", category)
        if not os.path.exists(train_category_dir) or not os.path.exists(test_category_dir):
            print(f"Skipping {category}: train or test directory missing")
            continue
        
        # Get normal images and anomaly masks
        normal_imgs = get_normal_images(train_category_dir)
        anomaly_masks = get_anomaly_masks(test_category_dir)
        if not normal_imgs or not anomaly_masks:
            print(f"Skipping {category}: No normal images or anomaly masks found")
            continue
        
        print(f"Num normal images: {len(normal_imgs)}")
        print(f"Num anomaly masks: {len(anomaly_masks)}")
        
        # Define output directory
        prompts = CATEGORY_PROMPTS.get(category, ["generic defect"])  # Fallback if category not in prompts
        num_images = imgs_per_prompt * len(prompts)
        dst_dir = os.path.join(src_dir, "augmented", category, f"augmented_{num_images}")
        os.makedirs(os.path.join(dst_dir, "imgs"), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, "masks"), exist_ok=True)

        # Generate augmented images
        img_idx = 0
        for prompt in prompts:
            print(f"Generating images for prompt: '{prompt}'")
            for _ in range(imgs_per_prompt):
                # Sample normal image and mask
                neg_img_path = random.choice(normal_imgs)
                mask_path = random.choice(anomaly_masks)
                
                # Load and resize
                neg_img = load_image(neg_img_path).resize(TARGET)
                mask = load_image(mask_path).resize(TARGET)

                # Inpaint
                out_image = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=neg_img,
                    mask_image=mask,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    generator=generator,
                    height=TARGET[1],
                    width=TARGET[0],
                    original_size=TARGET,
                    target_size=TARGET,
                    padding_mask_crop=padding_mask_crop
                ).images[0]

                # Resize to MVTecAD resolution
                out_image = out_image.resize(RES)
                mask = mask.resize(RES)
                mask = threshold_mask(mask) # Ensure 0/255 binary mask

                # Save with MVTecAD-compatible naming
                out_img_path = os.path.join(dst_dir, "imgs", f"{img_idx:05d}.png")
                out_mask_path = os.path.join(dst_dir, "masks", f"{img_idx:05d}_GT.png")
                out_image.save(out_img_path)
                mask.save(out_mask_path)
                img_idx += 1
        
        print(f"Generated {img_idx} synthetic positive samples for {category}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic positive samples for MVTec using SDXL Inpainting")
    parser.add_argument("--src_dir", type=str, required=True, help="Root directory of MVTec dataset (e.g., 'data/mvtec_preprocessed')")
    parser.add_argument("--imgs_per_prompt", type=int, default=50, help="Number of images to generate per prompt per category")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random generation")
    args = parser.parse_args()

    main(args)