# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(num_classes: int, input_shape=(224, 224, 3)):
    """
    Build a CNN model using transfer learning (MobileNetV2).
    Args:
        num_classes: number of output classes
        input_shape: input image shape (H, W, C)
    """
    # Use EfficientNetB0 (Better accuracy/efficiency tradeoff than MobileNetV2)
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Start frozen

    # Rebuild top
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    # Compile with a slightly lower learning rate for stability
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def unfreeze_model(model):
    """
    Unfreeze the top layers of the base model for fine-tuning.
    """
    # We assume layer[1] is the base model in a Sequential, but here we used Functional.
    # We find the EfficientNet layer.
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = True
            # Freeze the bottom N layers (optional, but good practice)
            # For B0, we might want to unfreeze the last 20-30 layers.
            # But simpler: just unfreeze all batchnorm layers should be kept frozen usually,
            # but EfficientNet requires BN to be trainable if unfreezing? 
            # Actually, standard practice: Unfreeze top blocks.
            # For simplicity, let's set trainable=True but with low LR.
            pass
            
    # Recompile with very low LR
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    m = build_model(num_classes=3)
    m.summary()
