import streamlit as st
import numpy as np
try:
    import cv2
except Exception:
    cv2 = None
from PIL import Image
from keras.models import load_model

from src.utils import load_single_image, get_class_names

# ----------------------------
# LOAD MODEL
# ----------------------------
model = load_model("models/model.h5")
CATEGORIES = get_class_names()

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI Medical Image Analysis",
    layout="centered"
)

# ----------------------------
# TITLE
# ----------------------------
st.title("🧠 AI-Powered Medical Image Analysis System")
st.write("Upload a Chest X-ray image to detect: Normal / Pneumonia / COVID")

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

# ============================
# IF IMAGE IS UPLOADED
# ============================
if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert PIL image to OpenCV format
    img_array = np.array(image)
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    # Preprocess using utils
    input_img = load_single_image(temp_path)

    # Predict button
    if st.button("🔍 Analyze Image"):

        with st.spinner("Analyzing... Please wait..."):
            prediction = model.predict(input_img)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        # Result display
        st.success(f"🧾 Prediction: {CATEGORIES[class_index]}")
        st.info(f"📊 Confidence: {confidence:.2f}")

        # Confidence bar
        st.progress(float(confidence))

# ============================
# ELSE PART (NO IMAGE UPLOADED)
# ============================
else:
    st.warning("⚠️ Please upload a medical image to start analysis.")

    st.markdown("""
    ### 📌 Instructions:
    1. Upload a Chest X-ray image (JPG / PNG)
    2. Click **Analyze Image**
    3. Get prediction: Normal / Pneumonia / COVID

    ### 🧠 Supported Classes:
    - Normal  
    - Pneumonia  
    - COVID-19  
    """)