import tensorflow as tf
import os


def create_tf_dataset(
    data_dir,
    img_size,
    batch_size,
    validation_split=None,
    subset=None,
    seed=42,
    shuffle=True,
):
    """
    Creates a tf.data.Dataset from image files in a directory.

    Args:
        data_dir (str): Path to the directory containing class subdirectories.
        img_size (tuple): Target size for the images (height, width).
        batch_size (int): Batch size for the dataset.
        validation_split (float, optional): Fraction of data to reserve for validation. Defaults to None.
        subset (str, optional): One of "training" or "validation". Required if validation_split is set.
        seed (int, optional): Seed for shuffling and transformations. Defaults to 42.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        tf.data.Dataset: A TensorFlow dataset of images and labels.
    """
    if validation_split and not subset:
        raise ValueError(
            "Must specify 'subset' ('training' or 'validation') when 'validation_split' is used."
        )

    # If data_dir structure is like 'data/PlantVillage_Subset/train' already
    # then validation_split is not needed here
    if validation_split is None:
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            labels="inferred",
            label_mode="int",
            image_size=img_size,
            interpolation="nearest",  # Or 'bicubic', 'bilinear'
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
    else:  # For initial splitting directly from a raw directory
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            labels="inferred",
            label_mode="int",
            image_size=img_size,
            interpolation="nearest",
            batch_size=batch_size,
            shuffle=shuffle,
            validation_split=validation_split,
            subset=subset,
            seed=seed,
        )
    return dataset


def get_class_names(data_dir):
    """
    Gets the class names (sub-directory names) from a dataset directory.

    Args:
        data_dir (str): Path to the dataset directory (e.g., train, val, test).

    Returns:
        list: A sorted list of class names.
    """
    return sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )


def apply_data_augmentation(image, label, img_height, img_width):
    """Applies data augmentation to an image."""
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    # Random zoom (TensorFlow Addons might be needed for advanced transforms, or implement manually)
    # This basic example sticks to built-in TF ops
    return image, label


def preprocess_image(image, label):
    """Normalizes image pixels to [0, 1] for EfficientNetV2."""
    # EfficientNetV2 models expect input in [0, 255] range when include_preprocessing=True
    # and handle normalization internally. If not, normalize explicitly.
    # Here we assume model will handle it or a separate layer will.
    # For this MVP, we will rely on EfficientNetV2's internal preprocessing.
    # If your model's input layer is tf.keras.applications.efficientnet_v2.preprocess_input, then no external normalization needed here.
    # If using a custom model or just the features, you'd normalize.
    # For simplicity, we just convert to float.
    image = tf.cast(image, tf.float32)
    return image, label


def prepare_dataset(
    dataset, img_height, img_width, augment=False, prefetch_buffer=tf.data.AUTOTUNE
):
    """Applies preprocessing and optional augmentation, then prefetches."""
    if augment:
        dataset = dataset.map(
            lambda x, y: apply_data_augmentation(x, y, img_height, img_width),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.cache().prefetch(buffer_size=prefetch_buffer)
