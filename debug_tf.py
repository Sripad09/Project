
try:
    import tensorflow as tf
    print(f"TF Version: {tf.__version__}")
    print(f"Has keras? {hasattr(tf, 'keras')}")
    if hasattr(tf, 'keras'):
        print(f"TF Keras: {tf.keras}")
except Exception as e:
    print(f"Error importing TF: {e}")

try:
    import keras
    print(f"Keras Version: {keras.__version__}")
except Exception as e:
    print(f"Error importing Keras: {e}")
