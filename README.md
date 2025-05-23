# Diffusion-Based Data Augmentation for Industrial Anomaly Detection

This project implements and evaluates PatchCore, a state-of-the-art anomaly detection method, on the Kolektor Surface Defect Dataset 2 (KSDD2). It includes options for using single or dual memory banks and incorporating augmented defect data generated by diffusion models.

## Core Components

### 1. PatchCore Implementation (`patchcore.py`)

This file contains the core logic for the PatchCore algorithm.

-   **`PatchCoreSingle`**: Implements PatchCore with a single memory bank, typically trained on normal (defect-free) samples.
-   **`PatchCoreDual`**: Extends PatchCore to use two memory banks:
    -   A **negative memory bank** built from normal samples.
    -   A **positive memory bank** built from anomalous (defective) samples. This can include pre-cropped defects and optionally, synthetically generated defects.
-   **Feature Extraction**: Utilizes pre-trained ResNet50 or WideResNet50_2 models as backbones to extract patch-level features from input images.
-   **Coreset Subsampling**: Employs coreset subsampling to reduce the size of the memory bank(s), improving efficiency while retaining representative features.
-   **Anomaly Scoring**:
    -   For a given test sample, features are extracted and compared against the memory bank(s).
    -   An anomaly score is calculated based on the distance to the nearest neighbors in the feature space.
    -   For `PatchCoreDual`, the scores from both positive and negative banks are combined.

### 2. Experiment Runner (`run_patchcore.py`)

This script orchestrates the training, evaluation, and experimentation process.

-   **Data Handling**:
    -   Loads the KSDD2 dataset using custom `Dataset` classes (`data/ksdd2.py` and `data/ksdd2_crops.py`).
    -   `KolektorSDD2`: Handles loading of the standard KSDD2 dataset.
    -   `KolektorSDD2Crops`: Specifically loads pre-cropped defect images, which can be augmented.
-   **Model Configuration**:
    -   Allows selection of the backbone network (ResNet50, WideResNet50_2).
    -   Configures subsampling rates for the memory bank(s).
    -   Supports enabling/disabling the use of augmented defect data.
-   **Training**: Fits the selected PatchCore model (`PatchCoreSingle` or `PatchCoreDual`) using the specified training data.
-   **Evaluation**:
    -   Evaluates the trained model on a test set.
    -   Calculates image-level and pixel-level Area Under the Receiver Operating Characteristic curve (AUROC) scores.
-   **Results**: Saves evaluation metrics to a `.npy` file.

### 3. Data Acquisition and Preprocessing

-   **`ksdd2_downloader.py`**: Script to download the KSDD2 dataset.
-   **`ksdd2_preprocess.py`**: Script for preprocessing the KSDD2 dataset. (Further details on preprocessing steps would typically be found within this script's comments or documentation).

### 4. Data Augmentation (`generate_augmented_images.py`)

-   This script is responsible for generating synthetic defect images using diffusion models.
-   The augmented images can be incorporated into the positive memory bank of the `PatchCoreDual` model to potentially improve its ability to detect diverse defects.

## Dataset

The project utilizes the **Kolektor Surface Defect Dataset 2 (KSDD2)**, which contains images of industrial surfaces with and without defects.

-   **Normal Data**: Images without any defects.
-   **Anomalous Data**: Images with various types of surface defects.
    -   The project uses pre-cropped defect regions for building the positive memory bank.

## Usage

1.  **Setup Environment**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Download Data**:
    ```bash
    python src/utils/ksdd2_downloader.py --out_path <path_to_save_ksdd2_dataset>
    ```
    *(Example: `python src/utils/ksdd2_downloader.py --out_path ./data/KSDD2_original`)*
3.  **Preprocess Data**:
    ```bash
    python src/utils/ksdd2_preprocess.py --src_dir <path_to_raw_ksdd2_dataset> --dst_dir <path_to_save_preprocessed_data>
    ```
    *(Example: `python src/utils/ksdd2_preprocess.py --src_dir ./data/KSDD2_original --dst_dir ./data/ksdd2_preprocessed`)*
    *(Ensure the source directory contains the KSDD2 dataset as downloaded, and the destination directory is where the processed files will be stored.)*

4.  **(Optional) Generate Augmented Images**:
    ```bash
    python src/utils/generate_augmented_images.py --src_dir <path_to_preprocessed_data> --imgs_per_prompt <number>
    ```
    *(Example: `python src/utils/generate_augmented_images.py --src_dir ./data/ksdd2_preprocessed --imgs_per_prompt 50`)*
    *(Refer to the script for more detailed arguments.)*
5.  **(Optional) Extract Anomaly Crops (if not part of your preprocessing or for specific experiments)**:
    ```bash
    python src/utils/extract_anomalous_crops.py --train_dir <path_to_preprocessed_data/train> --output_dir <path_to_save_crops> --csv_file <path_to_preprocessed_data/train.csv>
    ```
    *(Example: `python src/utils/extract_anomalous_crops.py --train_dir ./data/ksdd2_preprocessed/train --output_dir ./data/defect_crops --csv_file ./data/ksdd2_preprocessed/train.csv`)* 

6.  **Run PatchCore Experiments**:
    ```bash
    python src/run_patchcore.py \
        --dataset_path <path_to_preprocessed_data> \
        --crops_path <path_to_extracted_or_augmented_defect_crops> \
        --output_dir ./results \
        --backbone resnet50 \ # or wide_resnet50_2
        # --add_augmented \ # Uncomment to use augmented images from the crops_path if they are there
        # --augmented_path <path_if_augmented_are_separate_from_crops> \ # Usually crops_path will contain both original and augmented if used together
        --neg_subsampling 0.01 \
        --pos_subsampling 0.10 \
        --seed 42
    ```
    Refer to `src/run_patchcore.py` for the full list of command-line arguments and their descriptions.
