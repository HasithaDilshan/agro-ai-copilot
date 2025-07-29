import tensorflow as tf


class WeightedFocalLoss(tf.keras.losses.Loss):
    """
    Custom Weighted Focal Loss for multi-class classification.
    Addresses class imbalance by down-weighting easy examples and focusing on hard ones,
    with additional weighting for rare classes.
    """

    def __init__(self, gamma=2.0, alpha=None, name="weighted_focal_loss", reduction="sum_over_batch_size", **kwargs):
        """
        Initializes the WeightedFocalLoss.

        Args:
            gamma (float): Focusing parameter. When gamma > 0, easy examples are down-weighted.
            alpha (tf.Tensor or None): A 1D tensor of shape (num_classes,) containing
                                        per-class weighting factors. If None, no alpha weighting is applied.
            name (str): Name of the loss function.
            reduction (str): Type of reduction to apply to the loss. Defaults to "sum_over_batch_size".
            **kwargs: Keyword arguments for the base tf.keras.losses.Loss class.
        """
        super().__init__(name=name, reduction=reduction, **kwargs)  # Pass reduction explicitly
        self.gamma = float(gamma)  # Ensure gamma is a float
        
        # Handle alpha parameter - it might come as a list from deserialization
        if alpha is not None:
            try:
                if isinstance(alpha, (list, tuple)):
                    self.alpha = tf.constant(alpha, dtype=tf.float32)
                elif isinstance(alpha, tf.Tensor):
                    self.alpha = tf.cast(alpha, tf.float32)
                else:
                    # Handle numpy arrays or other array-like objects
                    self.alpha = tf.constant(alpha, dtype=tf.float32)
            except Exception as e:
                print(f"Warning: Could not convert alpha to tensor: {e}. Setting alpha to None.")
                self.alpha = None
        else:
            self.alpha = None

    def call(self, y_true, y_pred):
        """
        Calculates the Weighted Focal Loss.

        Args:
            y_true (tf.Tensor): True labels (one-hot encoded or integer labels if sparse_categorical_crossentropy is used).
                                Expected shape: (batch_size, num_classes) for one-hot, or (batch_size,) for integer.
            y_pred (tf.Tensor): Predicted probabilities. Expected shape: (batch_size, num_classes).

        Returns:
            tf.Tensor: Scalar loss value.
        """
        # Ensure y_pred is within valid range for log
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )

        # Convert y_true to one-hot if it's not already (for sparse_categorical_crossentropy compatibility)
        # Assuming y_true is integer-encoded (SparseCategoricalCrossentropy style)
        # We need to make it one-hot to multiply with y_pred and alpha weights
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)

        # Calculate Cross-Entropy Loss (element-wise)
        ce_loss = -y_true_one_hot * tf.math.log(y_pred)

        # Calculate pt (probability of true class)
        pt = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)

        # Calculate Focal Term
        focal_term = tf.pow(1.0 - pt, self.gamma)

        # Apply alpha weighting
        if self.alpha is not None:
            # Ensure alpha has correct shape for broadcasting or element-wise multiplication
            # It should broadcast across the batch dimension
            alpha_weight = tf.reduce_sum(y_true_one_hot * self.alpha, axis=-1)
            loss = alpha_weight * focal_term * tf.reduce_sum(ce_loss, axis=-1)
        else:
            loss = focal_term * tf.reduce_sum(ce_loss, axis=-1)

        return tf.reduce_mean(loss)  # Return scalar mean loss per batch

    def get_config(self):
        """
        Returns the config of the loss function.
        This is necessary for Keras to properly save and load custom objects.
        """
        config = super().get_config()  # Get base class config
        config.update(
            {
                "gamma": self.gamma,
                "alpha": (
                    self.alpha.numpy().tolist() if self.alpha is not None else None
                ),  # Convert tensor to list for serialization
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a loss instance from its config.
        This is also necessary for Keras to properly load custom objects.
        """
        # Create a copy of config to avoid modifying the original
        config_copy = config.copy()
        
        # Ensure we have default values for all required parameters
        if 'gamma' not in config_copy:
            config_copy['gamma'] = 2.0
        if 'alpha' not in config_copy:
            config_copy['alpha'] = None
        if 'name' not in config_copy:
            config_copy['name'] = 'weighted_focal_loss'
        if 'reduction' not in config_copy:
            config_copy['reduction'] = 'sum_over_batch_size'
            
        return cls(**config_copy)
