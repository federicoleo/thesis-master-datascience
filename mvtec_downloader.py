#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import tarfile
import requests

def download_and_extract(category, out_path, keep_archive=False):
    """
    Download and extract MVTec Anomaly Detection dataset by category
    """
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    # Dictionary of available categories and their download URLs
    mvtec_categories = {
        'bottle': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz',
        'cable': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz',
        'capsule': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz',
        'carpet': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz',
        'grid': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz',
        'hazelnut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz',
        'leather': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz',
        'metal_nut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz',
        'pill': 'https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz',
        'screw': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz',
        'tile': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz',
        'toothbrush': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz',
        'transistor': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz',
        'wood': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz',
        'zipper': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz',
        'mvtec_anomaly_detection': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz'
    }
    
    if category not in mvtec_categories:
        print(f"Error: Category '{category}' not found. Available categories:")
        for cat in mvtec_categories.keys():
            print(f" - {cat}")
        return
    
    url = mvtec_categories[category]
    tar_filename = f"{category}.tar.xz"
    tar_path = os.path.join(out_path, tar_filename)
    
    print(f"Downloading MVTec Anomaly Detection dataset - {category}...")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Error downloading dataset: HTTP status code {response.status_code}")
        return
        
    # Get the total file size for progress reporting
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB chunks
    downloaded = 0
    
    with open(tar_path, 'wb') as f:
        for data in response.iter_content(block_size):
            downloaded += len(data)
            f.write(data)
            # Print progress
            done = int(50 * downloaded / total_size) if total_size > 0 else 0
            print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded//(1024*1024)}MB/{total_size//(1024*1024)}MB", end='')
    print("\nDownload complete!")
    
    print(f"Extracting {tar_filename}...")
    try:
        with tarfile.open(tar_path, "r:xz") as tar:
            tar.extractall(out_path)
        print(f"Extraction complete!")
        
        # Remove the tar file if requested
        if not keep_archive:
            os.remove(tar_path)
            print(f"Removed archive file: {tar_filename}")
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        print("Archive file was not deleted due to extraction error.")
        return
    
    print(f"Done! Dataset '{category}' saved in: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract MVTec Anomaly Detection dataset")
    parser.add_argument('--category', type=str, default='bottle', 
                        help='Dataset category to download (bottle, cable, carpet, etc. or "all" for entire dataset)')
    parser.add_argument('--out_path', type=str, default='.', help='Output directory')
    parser.add_argument('--keep_archive', action='store_true', 
                        help='Keep the archive file after extraction')
    
    args = parser.parse_args()
    
    # If "all" is specified, download the full dataset
    if args.category == 'all':
        download_and_extract('mvtec_anomaly_detection', args.out_path, args.keep_archive)
    else:
        download_and_extract(args.category, args.out_path, args.keep_archive)


