import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import (
    preprocess_input,
    decode_predictions,
)
import numpy as np
import os
from PIL import Image  # Pillow library

# --- Configuration ---
IMG_SIZE = (
    224  # EfficientNetV2-B0 expects 224x224 input for ImageNet pre-trained weights
)
NUM_MOCK_CLASSES = 38  # PlantVillage has 38 classes (approx.) or ImageNet's 1000

# Mock class labels (replace with your actual PlantVillage class labels later)
# For MVP, we'll just use dummy labels or ImageNet's labels for basic testing.
# If using ImageNet pretrained, the model will output 1000 classes.
# For a *mock* plant disease, we can map to a few common ones.
MOCK_PLANT_DISEASE_LABELS = [
    "Healthy",
    "Common Rust",
    "Blight",
    "Leaf Spot",
    "Powdery Mildew",
    # ... add more relevant mock labels if desired
]


def load_and_preprocess_image(img_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """Loads and preprocesses an image for EfficientNetV2-B0."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch dimension
    # EfficientNetV2 models typically expect input in [0, 255] range when include_preprocessing=True
    # and handle normalization internally. If not, normalize to [-1, 1] or [0, 1].
    # Keras's EfficientNetV2 preprocess_input function handles this.
    return preprocess_input(img_array)


def get_mock_diagnosis(predictions, class_names):
    """Generates a mock diagnosis based on top prediction."""
    # For a truly pre-trained ImageNet model, decode_predictions decodes to ImageNet classes (1000 categories).
    # We need to adapt this for a mock plant disease.
    # Let's just pick the highest probability and map it to a mock plant disease.
    decoded_preds = decode_predictions(predictions, top=1)[
        0
    ]  # Top 1 ImageNet prediction
    imagenet_label = decoded_preds[0][1]  # e.g., 'granny_smith'
    confidence = decoded_preds[0][2]

    print(f"ImageNet detected: {imagenet_label} with {confidence:.2f} confidence.")

    # Simple mock logic for MVP:
    # In a real scenario, after fine-tuning on PlantVillage, you'd use your actual PlantVillage class labels.
    if "apple" in imagenet_label.lower() or "granny" in imagenet_label.lower():
        if confidence > 0.8:
            return (
                "Mock Diagnosis: Healthy Apple Leaf (ImageNet Confidence: {:.2f})".format(
                    confidence
                ),
                "healthy_apple",
            )
        else:
            return (
                "Mock Diagnosis: Apple leaf, but low confidence. Needs further analysis (ImageNet Confidence: {:.2f})".format(
                    confidence
                ),
                "uncertain_apple",
            )
    elif "lemon" in imagenet_label.lower() or "orange" in imagenet_label.lower():
        return (
            "Mock Diagnosis: Citrus leaf (ImageNet Confidence: {:.2f})".format(
                confidence
            ),
            "citrus_leaf",
        )
    else:
        # Fallback to a generic plant or 'disease' based on random chance for demonstration
        if np.random.rand() > 0.5:
            return (
                "Mock Diagnosis: Appears to be a healthy plant leaf (ImageNet Confidence: {:.2f})".format(
                    confidence
                ),
                "healthy_general",
            )
        else:
            return (
                "Mock Diagnosis: Possible early stage disease, uncertain (ImageNet Confidence: {:.2f})".format(
                    confidence
                ),
                "early_disease",
            )


def run_mvp_inference(image_path):
    """Runs the MVP inference flow."""
    # 1. Load the pre-trained EfficientNetV2-B0 model (ImageNet weights)
    # include_top=False means we get the feature extractor.
    # If include_top=True, it will try to classify into 1000 ImageNet classes.
    # For this MVP, let's use include_top=True to demonstrate immediate classification
    # and then mock-interpret the ImageNet predictions.
    try:
        model = EfficientNetV2B0(weights="imagenet", include_top=True)
        print("EfficientNetV2-B0 model loaded successfully with ImageNet weights.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(
            "Please ensure you have an internet connection to download weights or have them cached."
        )
        return "Error: Could not load model."

    # 2. Load and preprocess the input image
    preprocessed_img = load_and_preprocess_image(image_path)

    # 3. Make a prediction
    predictions = model.predict(preprocessed_img)

    # 4. Get mock diagnosis
    diagnosis, diagnosis_code = get_mock_diagnosis(
        predictions, MOCK_PLANT_DISEASE_LABELS
    )

    return diagnosis, diagnosis_code


if __name__ == "__main__":
    # Example usage: point to one of your downloaded images
    # Replace this with an actual path to an image in your module1-edge-ai/data/PlantVillage/
    sample_image_path = os.path.join(
        "data",
        "PlantVillage",
        "Apple___Apple_scab",
        "002e1de9-408a-4074-b78f-8d45366af9b3___FREC_Scab 3335.JPG",
    )
    # Ensure this path exists and you have downloaded the data!

    if not os.path.exists(sample_image_path):
        print(f"Error: Sample image not found at {sample_image_path}")
        print(
            "Please download PlantVillage data and update 'sample_image_path' to a valid image."
        )
    else:
        print(f"\n--- Running MVP inference on: {sample_image_path} ---")
        mock_diagnosis, diagnosis_code = run_mvp_inference(sample_image_path)
        print(f"\nFinal Mock Diagnosis: {mock_diagnosis}")
        print(f"Diagnosis Code: {diagnosis_code}")
