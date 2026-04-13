import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.gradcam import get_gradcam
from src.report import generate_report
import tempfile

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
model = load_model("models/model.h5")

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
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    pred = model.predict(img)
    return CLASSES[np.argmax(pred)], np.max(pred), img

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