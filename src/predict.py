import cv2
import numpy as np
from tensorflow.keras.models import load_model

import os
import urllib.request


MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?export=download&id=1lOpMb2k2Bz-rY0j6G9ZcKbFzvtalw62O"
    urllib.request.urlretrieve(url, MODEL_PATH)

model = load_model(MODEL_PATH)

IMG_SIZE = 224
CLASSES = ["NORMAL", "PNEUMONIA", "COVID", "TUMOR"]

def predict_image(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img)
    pred_class = CLASSES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return pred_class, confidence, img