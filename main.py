import numpy as np
from src.predict import predict_image

print("🧠 AI Medical Image Analysis (CLI Mode)")
print("--------------------------------------")

image_path = input("Enter image path: ").strip()

try:
    # ONLY ONE CALL HERE
    prediction = predict_image(image_path)

    print("\nRaw Prediction:", prediction)

    classes = ["NORMAL", "PNEUMONIA", "COVID"]

    result = classes[np.argmax(prediction)]

    print("✅ Final Result:", result)

except Exception as e:
    print("❌ Error:", e)