from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np

# Load a pre-trained feature extraction model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)


def preprocess_image(image):
    """
    Preprocess the input image for feature extraction.
    - Resizes the image to (224, 224).
    - Normalizes pixel values.
    - Extracts features using a MobileNetV2 model.
    - Flattens features to match model input requirements.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        np.ndarray: Flattened feature array.
    """
    # Resize image to the model's expected input size
    image = image.resize((224, 224))

    # Convert image to array and normalize pixel values
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = preprocess_input(image_array)  # Preprocess for MobileNetV2

    # Extract features using the feature extractor model
    features = feature_extractor.predict(image_array)

    # Flatten the features to a 1D array (e.g., for input shape of (None, 99))
    return features.flatten()
