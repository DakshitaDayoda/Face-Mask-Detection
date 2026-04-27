import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse
import os

# Load model
MODEL_PATH = "mask_detector_model.h5"
model = load_model(MODEL_PATH)

# Class labels (check your class_indices if needed)
CLASS_LABELS = ['mask', 'no_mask']


def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"Error: File not found -> {img_path}")
        return

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    label = CLASS_LABELS[1] if prediction > 0.5 else CLASS_LABELS[0]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print("\n--- Prediction Result ---")
    print(f"Image: {img_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask Detection on Image")
    parser.add_argument("--image", required=True, help="Path to input image")

    args = parser.parse_args()
    predict_image(args.image)