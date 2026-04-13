import cv2
import os
import numpy as np

IMG_SIZE = 224
CLASSES = ["NORMAL", "PNEUMONIA", "COVID", "TUMOR"]

def load_data(data_dir):
    data = []
    labels = []

    for category in CLASSES:
        path = os.path.join(data_dir, category)
        label = CLASSES.index(category)

        if not os.path.exists(path):
            continue

        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(label)
            except:
                pass

    data = np.array(data) / 255.0
    labels = np.array(labels)

    return data, labels