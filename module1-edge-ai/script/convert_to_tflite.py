import tensorflow as tf
import numpy as np
import os
import sys # For manipulating sys.path
import shutil # For potential cleanup like deleting previous runs

# --- IMPORTANT: CLONE YOUR REPO OR SYNC VIA DRIVE ---
# This section ensures your project code (including src/ and scripts/) is accessible.
# Option A: Clone the repo directly into Colab's ephemeral environment (recommended for active development)
# Make sure to update the URL with your actual GitHub repo URL
REPO_URL = "https://github.com/HasithaDilshan/agro-ai-copilot" # <--- CUSTOMIZE THIS TO YOUR ACTUAL REPO URL
REPO_NAME = "agro-ai-copilot"

# Check if the repository is already cloned to avoid re-cloning on subsequent runs
if not os.path.exists(REPO_NAME):
    print(f"Cloning repository: {REPO_URL}")
    !git clone {REPO_URL}
    %cd {REPO_NAME}
    # If using private repo, Colab will ask for credentials or you can set up SSH keys.
else:
    print(f"Repository '{REPO_NAME}' already cloned. Pulling latest changes...")
    %cd {REPO_NAME}
    !git pull origin main # Assuming your development is on main for simplicity, adjust if on feature branch

# --- Navigate to the specific module directory ---
# This ensures that 'src/' and 'scripts/' are relative to the current working directory
%cd module1-edge-ai

# Add project root (module1-edge-ai) to Python path to import from src/
project_root_in_colab = os.getcwd() # This will be /content/agro-ai-copilot/module1-edge-ai
if project_root_in_colab not in sys.path:
    sys.path.insert(0, project_root_in_colab) # Add to the beginning of sys.path

# Install module-specific dependencies from requirements.txt
print(f"Installing requirements from: {os.path.join(project_root_in_colab, 'requirements.txt')}")
!pip install -r requirements.txt
!pip install opencv-python-headless # Often useful for image processing (if not already in requirements)

# --- IMPORTANT: MOUNT GOOGLE DRIVE ---
# This is crucial for accessing your persistently stored data and saving models.
from google.colab import drive
drive.mount('/content/drive')

# --- DEFINE GOOGLE DRIVE PROJECT DATA PATHS (MATCHING drive_setup_project_data_dirs.ipynb) ---
# Ensure GOOGLE_DRIVE_PROJECT_ROOT matches the path you set in the setup notebook.
GOOGLE_DRIVE_PROJECT_ROOT = '/content/drive/MyDrive/AgroAI_Project_Data' # <--- ENSURE THIS MATCHES YOUR SETUP!
MODULE1_DRIVE_DATA_DIR = os.path.join(GOOGLE_DRIVE_PROJECT_ROOT, 'module1_edge_ai', 'data')
MODULE1_DRIVE_MODELS_DIR = os.path.join(GOOGLE_DRIVE_PROJECT_ROOT, 'module1_edge_ai', 'trained_models')

# Verify Drive paths exist (they should if setup notebook was run)
if not os.path.exists(MODULE1_DRIVE_DATA_DIR):
    raise FileNotFoundError(f"Module 1 data directory not found in Drive: {MODULE1_DRIVE_DATA_DIR}. Run setup notebook first.")
if not os.path.exists(MODULE1_DRIVE_MODELS_DIR):
    raise FileNotFoundError(f"Module 1 models directory not found in Drive: {MODULE1_DRIVE_MODELS_DIR}. Run setup notebook first.")

print(f"Module 1 data will be accessed from: {MODULE1_DRIVE_DATA_DIR}")
print(f"Module 1 models will be saved to: {MODULE1_DRIVE_MODELS_DIR}")

# Import custom loss function needed for loading the model
# This import will now work because module1-edge-ai (which contains src/) is in sys.path
from src.loss_functions import WeightedFocalLoss

print("Environment setup and imports complete.")
print(f"TensorFlow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")


def convert_keras_to_tflite(keras_model_path, tflite_output_path, num_classes, representative_dataset_path=None):
    """
    Converts a Keras FP32 model to TensorFlow Lite (TFLite).

    Args:
        keras_model_path (str): Path to the saved Keras .h5 model file.
        tflite_output_path (str): Path to save the converted .tflite model.
        num_classes (int): Number of classes your model predicts.
        representative_dataset_path (str, optional): Path to a directory
                                                    containing representative images for full integer quantization.
                                                    If None, dynamic range quantization is applied.
    """
    print(f"Loading Keras model from: {keras_model_path}")
    try:
        # Load the Keras model, providing custom_objects for the WeightedFocalLoss
        model = tf.keras.models.load_model(
            keras_model_path,
            custom_objects={'WeightedFocalLoss': WeightedFocalLoss}
        )
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        print("Ensure WeightedFocalLoss class is correctly defined and imported, and model path is correct.")
        return

    print("Keras model loaded successfully.")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if representative_dataset_path:
        print("Representative dataset path provided. Attempting full integer quantization...")
        def representative_dataset_gen():
            # For this to work, ensure representative_dataset_path is a directory
            # and contains images. You might use data_utils.create_tf_dataset
            # to load these images into a TF dataset.
            # Example (assuming create_tf_dataset and preprocess_image are available)
            # from src.data_utils import create_tf_dataset, preprocess_image
            # dataset_raw = create_tf_dataset(representative_dataset_path, (224, 224), 1, shuffle=False)
            # for images, _ in dataset_raw.map(preprocess_image).take(100): # Take a few batches
            #     yield [images] # Yield a list of inputs
            # For MVP, a dummy generator if you don't have representative_dataset setup yet:
            for _ in range(100):
                yield [np.random.uniform(0, 255, size=(1, 224, 224, 3)).astype(np.float32)] # Dummy data for calibration

        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        print("Configured for full integer quantization (INT8).")
    else:
        print("No representative dataset path provided. Applying dynamic range quantization.")

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_output_path), exist_ok=True)
    with open(tflite_output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved to: {tflite_output_path}")

if __name__ == "__main__":
    # Ensure the current directory is module1-edge-ai
    if os.path.basename(os.getcwd()) != 'module1-edge-ai':
        # This block ensures if the script is run standalone, it navigates correctly
        # However, the top-level setup block handles this when run in notebook cells.
        print("Script not run from module1-edge-ai directory. Adjusting path...")
        current_script_path = os.path.abspath(__file__)
        module_root = os.path.dirname(current_script_path)
        sys.path.insert(0, module_root)
        os.chdir(module_root)

    # Paths for model and class names, now correctly referencing Google Drive
    keras_model_path = os.path.join(MODULE1_DRIVE_MODELS_DIR, 'fp32_mvp_best_final.h5') # This path comes from training notebook
    tflite_output_path = os.path.join(MODULE1_DRIVE_MODELS_DIR, 'fp32_mvp_model.tflite')

    # Get number of classes from the saved class_names.txt in Google Drive
    class_names_path = os.path.join(MODULE1_DRIVE_DATA_DIR, 'class_names.txt')
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            num_classes = len(f.readlines())
    else:
        print(f"Warning: {class_names_path} not found in Google Drive. Assuming 38 classes for PlantVillage MVP.")
        num_classes = 38 # Default for PlantVillage

    convert_keras_to_tflite(keras_model_path, tflite_output_path, num_classes)
    print("TFLite conversion script finished.")