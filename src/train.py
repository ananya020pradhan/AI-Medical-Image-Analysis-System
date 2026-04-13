from preprocessing import load_data
from model import build_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/graphs", exist_ok=True)

data, labels = load_data("data/train")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

model = build_model()

history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

model.save("models/model.h5")

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.savefig("outputs/graphs/accuracy.png")