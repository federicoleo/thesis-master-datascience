import os
import shutil

# Define the source directory where your images are located
source_dir = "data/ksdd2_preprocessed/augmented_100/imgs"

# Define the destination directory for GT images
destination_dir = "data/ksdd2_preprocessed/augmented_100/masks"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Count how many files are moved
moved_count = 0

# Iterate through all files in the source directory
for filename in os.listdir(source_dir):
    # Check if the file ends with _GT.png
    if filename.endswith("_GT.png"):
        # Define the full paths
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)
        
        # Move the file
        shutil.move(source_path, destination_path)
        moved_count += 1
        print(f"Moved: {filename}")

print(f"Successfully moved {moved_count} mask files to {destination_dir}") 