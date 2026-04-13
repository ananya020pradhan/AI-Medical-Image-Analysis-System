import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from src.gradcam import get_gradcam
from src.report import generate_report
import tempfile
import urllib.request
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Medical Image Analysis",
    page_icon="🧠",
    layout="wide"
)

# ================= HEADER =================
st.title("🏥 AI-Powered Medical Image Analysis System")
st.markdown("### Upload scan + enter patient details for diagnosis")
st.markdown("---")

# ================= PATIENT INPUT (USER CONTROLLED) =================
st.sidebar.header("👤 Patient Information (Required)")

patient_name = st.sidebar.text_input("Patient Name")
patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=0)
patient_gender = st.sidebar.selectbox("Gender", ["Select", "Male", "Female", "Other"])

# ================= VALIDATION =================
if patient_name.strip() == "":
    st.warning("⚠ Please enter Patient Name")
    st.stop()

if patient_gender == "Select":
    st.warning("⚠ Please select Gender")
    st.stop()

if patient_age == 0:
    st.warning("⚠ Please enter valid Age")
    st.stop()

# ================= SETTINGS =================
show_gradcam = st.sidebar.checkbox("Show Grad-CAM", True)
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# ================= LOAD MODEL =================
import os
import urllib.request
import streamlit as st

MODEL_PATH = "model.h5"

@st.cache_resource

def load_my_model():
    try:
        from keras.models import load_model

        # Download model if not present
        if not os.path.exists(MODEL_PATH):
            url = "https://drive.google.com/uc?export=download&id=1lOpMb2k2Bz-rY0j6G9ZcKbFzvtalw62O"
            urllib.request.urlretrieve(url, MODEL_PATH)
          

        model = load_model(MODEL_PATH)
        return model

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


model = load_my_model()


IMG_SIZE = 224
CLASSES = ["NORMAL", "PNEUMONIA", "COVID", "TUMOR"]

# ================= FILE UPLOAD =================
uploaded_files = st.file_uploader(
    "📤 Upload Medical Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# ================= PREDICTION FUNCTION =================
def predict(image):
    import numpy as np
    import cv2

    # preprocess image
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # if model exists → real prediction
    if model is not None:
        pred = model.predict(img)
        class_idx = np.argmax(pred)
        confidence = float(np.max(pred))

        classes = ["NORMAL", "PNEUMONIA", "COVID", "TUMOR"]
        return classes[class_idx], confidence, img

    # if model not loaded → demo mode
    else:
        return "Demo Mode", 0.0, img
# ================= MAIN APP =================
if uploaded_files:

    cols = st.columns(2)

    for i, file in enumerate(uploaded_files):

        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        col = cols[i % 2]

        with col:
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Prediction
            pred_class, confidence, processed = predict(image)

            st.success(f"🧠 Prediction: {pred_class}")
            st.info(f"📊 Confidence: {confidence:.2f}")

            if confidence < threshold:
                st.warning("⚠ Low confidence result")

            # ================= GRAD-CAM =================
            if show_gradcam:
                if model is not None:
                    try:
                        heatmap = get_gradcam(model, processed)

                        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
                        heatmap = np.uint8(255 * heatmap)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

                        overlay = cv2.addWeighted(image_resized, 0.6, heatmap, 0.4, 0)

                        st.image(overlay, caption="🔥 Grad-CAM Visualization", use_container_width=True)

                    except Exception as e:
                        st.error(f"Grad-CAM Error: {e}")
                else:
                    st.info("ℹ️ Grad-CAM not available in Demo Mode")

            # ================= REPORT GENERATION =================
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp.name, image)

            generate_report(
                temp.name,
                pred_class,
                confidence,
                patient_name,
                patient_age,
                patient_gender
            )

            with open("report.pdf", "rb") as f:
                st.download_button(
                    label="📄 Download Medical Report",
                    data=f,
                    file_name=f"{patient_name}_report.pdf",
                    mime="application/pdf",
                    key=f"download_{i}"
                )

else:
    st.info("👆 Please upload medical images to start analysis")

# ================= FOOTER =================
st.markdown("---")
st.caption("⚕️ AI system for educational use only. Not a replacement for doctors.")
st.info("ℹ️ Model runs locally. Cloud version is in demo mode due to size limitations.")