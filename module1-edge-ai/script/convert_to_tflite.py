import tensorflow as tf
import os
import numpy as np

# Import custom loss function needed for loading the model
from src.loss_functions import WeightedFocalLoss

# --- DEFINE GOOGLE DRIVE PROJECT DATA PATHS (MATCHING drive_setup_project_data_dirs.ipynb) ---
# IMPORTANT: This script should be run from Colab, after mounting Drive.
# Ensure GOOGLE_DRIVE_PROJECT_ROOT matches the path you used in the setup notebook.
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
    # Paths for model and class names, now correctly referencing Google Drive
    keras_model_path = os.path.join(MODULE1_DRIVE_MODELS_DIR, 'fp32_mvp_best_model.h5')
    tflite_output_path = os.path.join(MODULE1_DRIVE_MODELS_DIR, 'fp32_mvp_model.tflite')

    # For full integer quantization (INT8), you'd use a path to a small subset of your training data
    # For MVP, we'll stick to dynamic range quantization first.
    # representative_data_dir = os.path.join(MODULE1_DRIVE_DATA_DIR, 'PlantVillage_Subset', 'val')

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