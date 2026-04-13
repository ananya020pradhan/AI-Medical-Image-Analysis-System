import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/model.h5")

IMG_SIZE = 224
CLASSES = ["NORMAL", "PNEUMONIA", "COVID", "TUMOR"]

def predict_image(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img)
    pred_class = CLASSES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return pred_class, confidence, img