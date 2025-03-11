import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import glob

def reshape_image(image, target_size):
    """
    Reshape an image to the target size.
    
    Args:
        image: Image array to reshape
        target_size: Tuple of (height, width)
        
    Returns:
        Resized image
    """
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

def extract_anomalous_crops(train_dir: str, output_dir: str, csv_file: str = None, padding: int = 10, 
                             min_area: int = 16, reshape_size: tuple = None, mask_suffix: str = "_GT"):
    """
    Extract and save anomalous patches from samples based on their masks, with optional CSV filtering.

    Args:
        train_dir (str): Directory containing images and masks (e.g., '10301.png', '10301_mask.png').
        output_dir (str): Directory to save cropped images and masks.
        csv_file (str, optional): Path to CSV file with 'path,label' columns (e.g., '10301.png,positive').
                                  If None, process all images in train_dir as positive (default: None).
        padding (int): Extra pixels around each defect (default: 10).
        min_area (int): Minimum patch dimension (height or width) in pixels (default: 16).
        reshape_size (tuple, optional): Size to reshape output images/masks (height, width).

    Returns:
        int: Number of patches processed and saved.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0

    # Determine samples to process
    if csv_file:
        # Read CSV and filter for positive samples
        df = pd.read_csv(csv_file)
        samples = df[df['label'] == 'positive']['path'].tolist()
        if not samples:
            print("No positive samples found in CSV.")
            return 0
    else:
        # No CSV: Assume all .png files in train_dir are positive
        all_files = glob.glob(os.path.join(train_dir, "*.png"))
        samples = [
            os.path.basename(f) for f in all_files
            if not f.endswith(f"{mask_suffix}.png")
        ]
        for f in samples:
            assert os.path.exists(os.path.join(train_dir, f"{f[:-4]}_GT.png")), f"Mask not found for {f}"

        if not samples:
            print(f"No .png files found in {train_dir}.")
            return 0

    for img_filename in tqdm(samples, desc="Extracting anomaly patches"):
        # Construct full paths
        img_path = os.path.join(train_dir, img_filename)
        mask_filename = img_filename.replace('.png', '_GT.png')  # Assumes mask naming convention
        mask_path = os.path.join(train_dir, mask_filename)

        # Check if files exist
        if not os.path.exists(img_path):
            print(f"Skipping {img_filename}: Image not found.")
            continue
        if not os.path.exists(mask_path):
            print(f"Skipping {img_filename}: Mask not found.")
            continue

        # Load image and mask
        img_np = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # [H, W, C]
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # [H, W], 0 or 255

        if img_np is None or mask_np is None:
            print(f"Skipping {img_filename}: Failed to load image or mask.")
            continue

        # Convert mask to binary (0 or 1)
        mask_np = (mask_np > 0).astype(np.uint8)  # Threshold to 0/1
        mask_uint8 = mask_np * 255  # For saving as 0/255

        # Find contours in the mask
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fallback: If no contours but mask has positive pixels
        if not contours and mask_np.sum() > 0:
            y_coords, x_coords = np.where(mask_np > 0)
            if len(y_coords) > 0:
                x_min = max(0, np.min(x_coords) - padding)
                x_max = min(mask_np.shape[1], np.max(x_coords) + padding + 1)
                y_min = max(0, np.min(y_coords) - padding)
                y_max = min(mask_np.shape[0], np.max(y_coords) + padding + 1)

                if (y_max - y_min) >= min_area and (x_max - x_min) >= min_area:
                    # Crop patches
                    img_patch = img_np[y_min:y_max, x_min:x_max]
                    mask_patch = mask_uint8[y_min:y_max, x_min:x_max]
                    
                    # Reshape if specified
                    if reshape_size:
                        img_patch = reshape_image(img_patch, reshape_size)
                        mask_patch = reshape_image(mask_patch, reshape_size)

                    # Save patches
                    base_name = os.path.splitext(img_filename)[0]
                    patch_filename = f"{base_name}_anomaly_full.png"
                    mask_filename = f"{base_name}_anomaly_full_mask.png"
                    cv2.imwrite(os.path.join(output_dir, patch_filename), cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(output_dir, mask_filename), mask_patch)
                    processed_count += 1

        # Process each contour
        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Add padding
            x_min = max(0, x - padding)
            y_min = max(0, y - padding)
            x_max = min(img_np.shape[1], x + w + padding)
            y_max = min(img_np.shape[0], y + h + padding)

            # Skip small patches
            if (y_max - y_min) < min_area or (x_max - x_min) < min_area:
                continue

            # Crop patches
            img_patch = img_np[y_min:y_max, x_min:x_max]
            mask_patch = mask_uint8[y_min:y_max, x_min:x_max]
            
            # Reshape if specified
            if reshape_size:
                img_patch = reshape_image(img_patch, reshape_size)
                mask_patch = reshape_image(mask_patch, reshape_size)

            # Save patches
            base_name = os.path.splitext(img_filename)[0]
            patch_filename = f"{base_name}.png"
            mask_filename = f"{base_name}_GT.png"
            cv2.imwrite(os.path.join(output_dir, patch_filename), cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, mask_filename), mask_patch)
            processed_count += 1

    print(f"Extracted and saved {processed_count} anomalous patches to {output_dir}")
    return processed_count

def analyze_results(output_dir):
    """Analyze the extracted patches and provide statistics."""
    extracted_images = [f for f in os.listdir(output_dir) if not f.endswith('_GT.png')]
    if not extracted_images:
        print("No extracted patches found for analysis.")
        return
        
    sizes = []
    total_pixels = 0
    
    for img_name in extracted_images:
        img_path = os.path.join(output_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            sizes.append((h, w))
            total_pixels += h * w
    
    if sizes:
        avg_size = total_pixels / len(sizes)
        min_h = min(h for h, w in sizes)
        min_w = min(w for h, w in sizes)
        max_h = max(h for h, w in sizes)
        max_w = max(w for h, w in sizes)
        
        print("\nPatch Statistics:")
        print(f"Total patches: {len(sizes)}")
        print(f"Average patch area: {avg_size:.1f} pixels")
        print(f"Size range: ({min_h}×{min_w}) to ({max_h}×{max_w})")
        
        # Check if all patches have identical dimensions (likely from reshaping)
        unique_sizes = set(sizes)
        if len(unique_sizes) == 1:
            print(f"All patches have identical dimensions: {min_h}×{min_w} (reshaped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and save anomalous patches from images using their masks."
    )
    parser.add_argument("--train_dir", type=str, required=True, help="Directory containing the images and masks.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where extracted patches will be saved.")
    parser.add_argument("--csv_file", type=str, default=None, help="Path to CSV file with 'path,label' columns. If provided, only samples labeled as 'positive' will be processed.")
    parser.add_argument("--padding", type=int, default=10, help="Extra pixels around each defect (default: 10).")
    parser.add_argument("--min_area", type=int, default=16, help="Minimum patch dimension in pixels (default: 16).")
    parser.add_argument("--reshape", type=str, default=None, help="Reshape all output images to this size, format: 'HEIGHT,WIDTH' (e.g. '256,256')")
    parser.add_argument("--analyze", action="store_true", help="Analyze and display statistics about the extracted patches.")
    args = parser.parse_args()
    
    # Parse reshape dimensions if provided
    reshape_size = None
    if args.reshape:
        try:
            h, w = map(int, args.reshape.split(','))
            reshape_size = (h, w)
            print(f"- Reshaping output to: {h}×{w} pixels")
        except ValueError:
            print(f"Error: Invalid reshape format '{args.reshape}'. Expected format: 'HEIGHT,WIDTH'")
            exit(1)
    
    print(f"Starting anomaly patch extraction:")
    print(f"- Source directory: {args.train_dir}")
    print(f"- Output directory: {args.output_dir}")
    print(f"- CSV file: {args.csv_file if args.csv_file else 'None (processing all images)'}")
    print(f"- Padding: {args.padding} pixels")
    print(f"- Minimum area: {args.min_area} pixels")
    
    count = extract_anomalous_crops(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        csv_file=args.csv_file,
        padding=args.padding,
        min_area=args.min_area,
        reshape_size=reshape_size
    )

    if args.analyze and count > 0:
        analyze_results(args.output_dir)