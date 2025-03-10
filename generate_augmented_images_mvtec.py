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
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import time
import re

# Define defect-specific prompts for each category and defect type
DEFECT_TYPE_PROMPTS = {
    "bottle": {
        "broken_large": ["large crack in glass", "broken glass rim", "shattered bottle edge"],
        "broken_small": ["small glass chip", "tiny crack on surface", "minor glass damage"],
        "contamination": ["contamination spot", "dirt on surface", "foreign material"]
    },
    "cable": {
        "bent_wire": ["bent copper wire", "deformed cable", "warped wire"],
        "cable_swap": ["misaligned wires", "swapped cable colors", "incorrect wire arrangement"],
        "combined": ["damaged cable insulation with exposed wire", "multiple cable defects", "complex cable damage"],
        "cut_inner_insulation": ["cut in inner insulation", "damaged internal cable layer", "exposed inner wire"],
        "cut_outer_insulation": ["damaged outer cable coating", "cut in external insulation", "outer layer damage"],
        "missing_cable": ["missing wire segment", "incomplete cable", "gap in wire"],
        "missing_wire": ["missing internal wire", "wire strand absence", "incomplete internal conductor"],
        "poke_insulation": ["punctured insulation", "small hole in cable coating", "pinpoint damage to cable cover"]
    },
    # Add other categories with their specific defect types...
}

# Generic prompts for when specific defect types aren't defined
CATEGORY_PROMPTS = {
    "bottle": ["small crack on rim", "glass chip", "contamination spot", "glass fragment"],
    "cable": ["exposed wire", "bent cable", "missing wire", "damaged insulation"],
    "capsule": ["crack on capsule", "deformed shape", "scratch on surface", "discoloration"],
    "carpet": ["stain on fabric", "thread pull", "small hole", "color irregularity"],
    "grid": ["bent grid section", "broken connection", "deformed pattern", "misalignment"],
    "hazelnut": ["crack in shell", "mold spot", "puncture", "dark discoloration"],
    "leather": ["scratch mark", "fold line", "small puncture", "stain patch"],
    "metal_nut": ["surface scratch", "bent edge", "discoloration", "deformed shape"],
    "pill": ["surface crack", "color spot", "imperfect imprint", "contamination"],
    "screw": ["scratch on head", "damaged thread", "deformed shaft", "rough edge"],
    "tile": ["hairline crack", "stain mark", "rough spot", "edge chip"],
    "toothbrush": ["misaligned bristles", "damaged head", "bent bristles", "discoloration"],
    "transistor": ["bent pin", "damaged casing", "misalignment", "broken lead"],
    "wood": ["knot in grain", "scratch line", "discoloration", "small hole"],
    "zipper": ["broken tooth", "misaligned teeth", "fabric damage", "bent track"]
}

# Category-specific negative prompts
NEGATIVE_PROMPTS = {
    "bottle": "smooth unbroken rim, clean glass, intact edge, uniform appearance",
    "cable": "intact insulation, properly aligned, complete wires, undamaged coating",
    "capsule": "smooth surface, perfect shape, uniform color, clear imprint",
    "carpet": "uniform texture, even color, intact fibers, consistent pattern",
    "grid": "straight lines, intact connections, uniform pattern, undamaged structure",
    "hazelnut": "smooth shell, even color, intact surface, natural appearance",
    "leather": "smooth texture, even color, unblemished surface, uniform appearance",
    "metal_nut": "smooth surface, straight edges, uniform color, perfect shape",
    "pill": "smooth surface, clear imprint, uniform color, perfect shape",
    "screw": "smooth threads, unblemished head, straight shaft, uniform finish",
    "tile": "smooth surface, even color, perfect edges, uniform texture",
    "toothbrush": "aligned bristles, uniform head, complete pattern, symmetrical shape",
    "transistor": "straight pins, intact case, correct alignment, undamaged leads",
    "wood": "smooth grain, even color, natural texture, unblemished surface",
    "zipper": "aligned teeth, intact track, smooth edge, uniform spacing"
}

def get_normal_images(category_dir):
    """Get list of normal (good) images for a category."""
    return glob(os.path.join(category_dir, "good", "*.png"))

def get_anomaly_masks_by_defect(category_dir):
    """Get dictionary of anomaly masks organized by defect type."""
    defect_masks = {}
    for defect_type in os.listdir(category_dir):
        if defect_type != "good" and os.path.isdir(os.path.join(category_dir, defect_type)):
            masks = glob(os.path.join(category_dir, defect_type, "*_GT.png"))
            if masks:
                defect_masks[defect_type] = masks
    return defect_masks

def enhance_mask(mask_img, dilation_pixels=5):
    """Slightly enhance mask by dilating it to ensure full defect coverage."""
    import cv2
    mask_np = np.array(mask_img)
    if len(mask_np.shape) == 3:
        mask_np = mask_np[:,:,0]  # Use first channel if RGB
    
    # Ensure mask is binary
    binary_mask = (mask_np > 127).astype(np.uint8)
    
    # Dilate mask to slightly expand defect area
    kernel = np.ones((dilation_pixels, dilation_pixels), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    
    # Convert back to PIL with values 0 and 255
    return Image.fromarray(dilated_mask * 255)

def visualize_images(original, mask, inpainted, save_path):
    """Create a visualization showing original, mask, and inpainted result."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(np.array(original))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(np.array(mask), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Inpainted Result")
    plt.imshow(np.array(inpainted))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(args):
    ### ARGUMENTS
    src_dir = args.src_dir
    imgs_per_prompt = args.imgs_per_prompt
    seed = args.seed
    categories_to_augment = args.categories
    debug_mode = args.debug
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create timestamp for unique debug folder
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Load SD-XL Inpainting model
    model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    print(f"Loading model {model_id}...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        variant="fp16" if device.type == "cuda" else None
    ).to(device)
    
    # Process each category
    all_categories = sorted([d for d in os.listdir(os.path.join(src_dir, "train")) 
                            if os.path.isdir(os.path.join(src_dir, "train", d))])
    
    if categories_to_augment:
        categories = [c for c in categories_to_augment if c in all_categories]
    else:
        categories = all_categories
    
    print(f"Categories to process: {categories}")
    
    # Create a debug directory if in debug mode
    if debug_mode:
        debug_dir = os.path.join(src_dir, f"debug_inpainting_{timestamp}")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug output will be saved to {debug_dir}")
    
    for category in categories:
        print(f"\nProcessing category: {category}")
        
        # Define paths
        train_category_dir = os.path.join(src_dir, "train", category)
        test_category_dir = os.path.join(src_dir, "test", category)
        
        if not os.path.exists(train_category_dir):
            print(f"ERROR: Train category directory missing: {train_category_dir}")
            continue
            
        if not os.path.exists(test_category_dir):
            print(f"ERROR: Test category directory missing: {test_category_dir}")
            continue
        
        # Get normal images
        normal_imgs = get_normal_images(train_category_dir)
        if not normal_imgs:
            print(f"ERROR: No normal images found in {train_category_dir}/good/")
            continue
        print(f"Found {len(normal_imgs)} normal images")
        
        # Get defect masks organized by defect type
        defect_masks_by_type = get_anomaly_masks_by_defect(test_category_dir)
        if not defect_masks_by_type:
            print(f"ERROR: No defect masks found in {test_category_dir}")
            continue
        
        print(f"Found defect types: {list(defect_masks_by_type.keys())}")
        for defect_type, masks in defect_masks_by_type.items():
            print(f"  {defect_type}: {len(masks)} masks")
        
        # Create output directories
        total_prompts = 0
        for defect_type in defect_masks_by_type:
            if defect_type in DEFECT_TYPE_PROMPTS.get(category, {}):
                total_prompts += len(DEFECT_TYPE_PROMPTS[category][defect_type])
            else:
                total_prompts += 1  # Using a generic prompt
        
        num_images = imgs_per_prompt * total_prompts
        dst_dir = os.path.join(src_dir, "augmented", category, f"augmented_{num_images}")
        os.makedirs(os.path.join(dst_dir, "imgs"), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, "masks"), exist_ok=True)
        
        # Keep track of generated images
        img_idx = 0
        
        # Process each defect type with dedicated prompts
        for defect_type, masks in defect_masks_by_type.items():
            if len(masks) == 0:
                continue
                
            print(f"\nProcessing defect type: {defect_type}")
            
            # Get prompts for this specific defect type, or use generic ones
            if category in DEFECT_TYPE_PROMPTS and defect_type in DEFECT_TYPE_PROMPTS[category]:
                prompts = DEFECT_TYPE_PROMPTS[category][defect_type]
                print(f"Using defect-specific prompts: {prompts}")
            else:
                # Fall back to category-level prompts
                prompts = CATEGORY_PROMPTS.get(category, ["generic defect"])
                print(f"Using generic prompts: {prompts}")
            
            # Get negative prompt
            negative_prompt = NEGATIVE_PROMPTS.get(category, "perfect condition, no defects")
            
            # Generate images for each prompt
            for prompt in prompts:
                print(f"Generating {imgs_per_prompt} images for prompt: '{prompt}'")
                
                for i in range(imgs_per_prompt):
                    # Select normal image and mask from this defect type
                    normal_img_path = random.choice(normal_imgs)
                    mask_path = random.choice(masks)
                    
                    # Extract information from file paths
                    defect_name = os.path.basename(os.path.dirname(mask_path))
                    mask_filename = os.path.basename(mask_path)
                    normal_filename = os.path.basename(normal_img_path)
                    
                    print(f"  [{img_idx}] Creating defect '{defect_type}' using:")
                    print(f"      Normal: {normal_filename}")
                    print(f"      Mask: {mask_filename}")
                    
                    try:
                        # Load images
                        normal_img = load_image(normal_img_path)
                        mask_img = load_image(mask_path)
                        
                        # Get dimensions
                        original_size = normal_img.size
                        
                        # Resize to SDXL-friendly resolution (1024x1024)
                        normal_img_sdxl = normal_img.resize((1024, 1024))
                        mask_img_sdxl = mask_img.resize((1024, 1024))
                        
                        # Process mask: enhance it slightly to ensure complete coverage
                        mask_img_sdxl = enhance_mask(mask_img_sdxl, dilation_pixels=3)
                        
                        # Create instance-specific seed
                        instance_seed = seed + img_idx
                        generator = torch.Generator(device=device).manual_seed(instance_seed)
                        
                        # Crucial parameters for proper inpainting
                        # strength=0.99 will replace the masked area entirely
                        # strength=0.0 would keep the original image
                        # We use a lower strength to ensure we're keeping most of the normal image
                        inpaint_strength = 0.6
                        
                        # Debug info
                        if debug_mode:
                            print(f"  Prompt: '{prompt}'")
                            print(f"  Negative prompt: '{negative_prompt}'")
                            print(f"  Inpainting strength: {inpaint_strength}")
                            print(f"  Seed: {instance_seed}")
                        
                        # Generate inpainted image
                        start_time = time.time()
                        result = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=normal_img_sdxl,
                            mask_image=mask_img_sdxl,
                            guidance_scale=args.guidance_scale,
                            num_inference_steps=args.steps,
                            strength=inpaint_strength,
                            generator=generator,
                            height=1024,
                            width=1024
                        )
                        
                        inpainted_img = result.images[0]
                        print(f"  Generation took {time.time() - start_time:.2f}s")
                        
                        # Resize back to original resolution
                        inpainted_img = inpainted_img.resize(original_size)
                        
                        # Save the result
                        img_filename = f"{img_idx:05d}.png"
                        mask_filename = f"{img_idx:05d}_GT.png" 
                        
                        # Save inpainted image
                        inpainted_path = os.path.join(dst_dir, "imgs", img_filename)
                        inpainted_img.save(inpainted_path)
                        
                        # Save the corresponding mask
                        mask_out_path = os.path.join(dst_dir, "masks", mask_filename)
                        mask_img.resize(original_size).save(mask_out_path)
                        
                        # Create visualization if in debug mode
                        if debug_mode:
                            vis_path = os.path.join(debug_dir, f"{category}_{defect_type}_{img_idx:05d}.png")
                            visualize_images(normal_img, mask_img, inpainted_img, vis_path)
                        
                        img_idx += 1
                        
                    except Exception as e:
                        print(f"  ERROR generating image: {str(e)}")
        
        print(f"Successfully generated {img_idx} augmented images for category '{category}'")
    
    print("\nAugmentation process completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic defect images for MVTec dataset")
    parser.add_argument("--src_dir", type=str, required=True, help="Directory containing the preprocessed dataset")
    parser.add_argument("--imgs_per_prompt", type=int, default=5, help="Number of images per prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--categories", nargs="+", default=None, help="List of categories to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with visualizations")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=12.0, help="Guidance scale (7.5-15.0 work well)")
    
    args = parser.parse_args()
    main(args)

