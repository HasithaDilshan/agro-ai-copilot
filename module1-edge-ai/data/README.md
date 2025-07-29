# Module 1: Data Management for Disease Recognition

This directory (`module1-edge-ai/data/`) is intended for small data samples and configuration files.
**Large datasets are NOT committed to this Git repository.**

## Full Dataset Acquisition (PlantVillage)

The primary dataset for training the disease recognition model is the **PlantVillage dataset**.

**Where to get it:**
-   **Kaggle:** [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
    * You will need a Kaggle account to download the `plantvillage dataset.zip` file.
-   **Public Google Cloud Storage Bucket (common mirror):** `https://storage.googleapis.com/plantdata/PlantVillage.zip`
    * This URL is often used directly within Google Colab notebooks for download.

**How to use it in Colab:**
1.  Upload the `plantvillage dataset.zip` to your **Google Drive** (e.g., `MyDrive/`).
2.  In `module1-edge-ai/notebooks/mvp_data_prep.ipynb`, ensure the `PLANTVILLAGE_ZIP_PATH_DRIVE` variable points to the correct path in your Google Drive.
3.  Alternatively, the Colab notebook has a `wget` command that can download the dataset directly into the Colab runtime environment from the GCS URL.

## Subset Creation

The `module1-edge-ai/notebooks/mvp_data_prep.ipynb` notebook is used to:
1.  Extract the raw PlantVillage dataset.
2.  Create a stratified `train`, `val`, `test` split (e.g., 70%/15%/15%) into the `module1-edge-ai/data/PlantVillage_Subset/` directory (this directory is in the Colab runtime or Google Drive, and *ignored by Git*).

## Local Samples (`PlantVillage_Subset_Sample/`)

This directory may contain a *very small sample* of images from the PlantVillage dataset for local development and testing purposes (e.g., quickly testing data loaders or inference logic without needing the full dataset).

**This sample data is explicitly ignored by Git (via the root `.gitignore` file).**

## Class Names

After training, the list of class names will be saved to `module1-edge-ai/data/class_names.txt` by the training notebook. This file *should* be committed to Git as it's crucial metadata for interpreting model outputs.