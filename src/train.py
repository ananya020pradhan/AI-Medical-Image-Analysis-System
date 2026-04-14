import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from src.preprocessing import load_data
from src.model import build_model


def train_model():
    print("🚀 Loading dataset...")

    # Load data
    X, y = load_data("data/train")

    print(f"Dataset Loaded: {X.shape}, {y.shape}")

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("🧠 Building model...")
    model = build_model()

    print("🔥 Training started...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32
    )

    # -------------------------------
    # CREATE FOLDERS SAFELY
    # -------------------------------
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # -------------------------------
    # SAVE MODEL
    # -------------------------------
    model.save("models/model.h5")
    print("💾 Model saved successfully!")

    # -------------------------------
    # ACCURACY GRAPH
    # -------------------------------
    plt.figure()
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("outputs/accuracy.png")
    print("📊 Accuracy graph saved!")

    # -------------------------------
    # CONFUSION MATRIX
    # -------------------------------
    print("📉 Generating Confusion Matrix...")

    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_val, y_pred_classes)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.savefig("outputs/confusion_matrix.png")
    print("📉 Confusion matrix saved!")

    # -------------------------------
    # CLASSIFICATION REPORT
    # -------------------------------
    print("\n📌 Classification Report:\n")
    print(classification_report(y_val, y_pred_classes))


    print("✅ Training Completed Successfully!")


# Run directly
if __name__ == "__main__":
    train_model()