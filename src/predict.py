import os
import cv2
import numpy as np
from keras.models import load_model

model = load_model("models/model.h5", compile=False)

def predict_image(image_path):

    # MUST be string only
    if not isinstance(image_path, str):
        raise TypeError(f"Expected string path, got {type(image_path)}")

    image_path = image_path.strip()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image could not be read")

    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return model.predict(img)