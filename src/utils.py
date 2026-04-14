import numpy as np
import cv2
import os

# ----------------------------
# IMAGE LOADER (SAFE)
# ----------------------------
def load_single_image(path, img_size=224):
    img = cv2.imread(path)

    if img is None:
        raise ValueError("Image not found or invalid format")

    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# ----------------------------
# CREATE DIRECTORY SAFELY
# ----------------------------
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ----------------------------
# LABEL MAPPING
# ----------------------------
def get_class_names():
    return ["NORMAL", "PNEUMONIA", "COVID"]