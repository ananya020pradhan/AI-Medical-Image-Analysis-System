import cv2
import os
import numpy as np

IMG_SIZE = 224
CATEGORIES = ["NORMAL", "PNEUMONIA", "COVID"]

def load_data(data_dir):
    data = []
    labels = []

    for category in CATEGORIES:
        path = os.path.join(data_dir, category)
        class_index = CATEGORIES.index(category)

        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                image = image / 255.0

                data.append(image)
                labels.append(class_index)

            except:
                continue

    return np.array(data), np.array(labels)