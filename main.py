import cv2
import os
from src.predict import predict_image

def main():
    print("🧠 AI Medical Image Analysis (CLI Mode)")
    print("--------------------------------------")

    image_path = input("Enter image path: ").strip()

    # Check if file exists
    if not os.path.exists(image_path):
        print("❌ Error: File not found")
        return

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Error: Invalid image file")
        return

    try:
        # Prediction
        pred_class, confidence, _ = predict_image(image)

        print("\n✅ RESULT")
        print(f"Prediction : {pred_class}")
        print(f"Confidence : {confidence:.2f}")

        # Interpretation
        if pred_class == "NORMAL":
            print("✔ No disease detected")
        else:
            print("⚠ Possible disease detected - consult a doctor")

    except Exception as e:
        print("❌ Error during prediction:", str(e))


if __name__ == "__main__":
    main()