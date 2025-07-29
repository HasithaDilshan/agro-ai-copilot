import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models

IMG_SIZE = 224  # Standard input size for EfficientNetV2-B0


def build_fp32_efficientnet_model(num_classes):
    """
    Builds the FP32 baseline EfficientNetV2-B0 model with a custom classification head.

    Args:
        num_classes (int): The number of output classes for classification.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    # Load EfficientNetV2-B0 as the base model, pre-trained on ImageNet
    # include_top=False: Exclude the ImageNet classification head
    # weights='imagenet': Use pre-trained weights
    # input_shape: Define the expected input size (channels_last for TF)
    base_model = EfficientNetV2B0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze the base model's weights to prevent them from being updated during training
    # This is common in transfer learning to keep learned features intact.
    base_model.trainable = False

    # Create a custom classification head
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(
        inputs, training=False
    )  # Important: set training=False when using a frozen base
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)  # A dense layer
    x = layers.Dropout(0.3)(x)  # Dropout for regularization
    outputs = layers.Dense(num_classes, activation="softmax")(
        x
    )  # Final classification layer

    model = models.Model(inputs, outputs)

    return model


# Add an empty __init__.py in src/ if not already present to make it a package
