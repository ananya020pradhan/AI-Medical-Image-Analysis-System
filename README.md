# 🏥 AI-Powered Medical Image Analysis System

## 🔥 Overview
This project is an AI-based medical image analysis system that detects diseases such as Pneumonia, COVID-19, Brain Tumor, and Normal cases from X-ray/MRI images using Deep Learning.

It includes Explainable AI (Grad-CAM), a Streamlit web app, and automated PDF medical report generation with patient details.

⚠️ Educational project only — not for real medical diagnosis.

---

## 🎯 Problem Statement
Medical image analysis is:
- Time-consuming
- Requires expert radiologists
- Prone to human error

This system solves it by:
- Automating disease detection using AI
- Providing explainable results using Grad-CAM
- Generating structured medical reports

---

## ✨ Features
- Multi-class disease classification (Normal, Pneumonia, COVID-19, Tumor)
- CNN / MobileNetV2 deep learning model
- Grad-CAM heatmap visualization
- Streamlit interactive dashboard
- Patient information input system
- PDF medical report generation
- Confidence score + severity detection
- Multi-image upload support

---

## 🛠 Tech Stack
Python, TensorFlow, Keras, OpenCV, NumPy, Streamlit, Matplotlib, ReportLab

---

## 🧠 System Architecture
Medical Image Input  
→ Preprocessing (Resize, Normalize)  
→ CNN / MobileNetV2 Model  
→ Disease Prediction  
→ Grad-CAM Visualization  
→ Streamlit UI Output  
→ PDF Report Generation  

---

## 🔄 Workflow
1. Upload medical image  
2. Image preprocessing  
3. AI model prediction  
4. Grad-CAM explanation  
5. Streamlit UI display  
6. PDF report generation  

---

## 📂 Project Structure
AI-Medical-Image-Analysis-System/  
├── app.py  
├── main.py  
├── requirements.txt  
├── README.md  
├── models/  
│   └── model.h5  
├── src/  
│   ├── gradcam.py  
│   ├── report.py  
│   ├── predict.py  
│   ├── train.py  
├── data/  
├── outputs/  
└── images/  

---

## ⚙️ Installation
git clone https://github.com/your-username/AI-Medical-Image-Analysis-System.git  
cd AI-Medical-Image-Analysis-System  
python -m venv venv  
source venv/bin/activate (Mac/Linux)  
venv\Scripts\activate (Windows)  
pip install -r requirements.txt  

---

## 🚀 Run Project
streamlit run app.py  

---

## 📊 Output
- Disease prediction result  
- Confidence score  
- Grad-CAM heatmap  
- PDF medical report  
- Patient-based analysis  

---

## 🌐 Deployment
- Streamlit Cloud  
- AWS EC2  
- HuggingFace Spaces  

---

## 📚 Learning Outcomes
- Deep learning model building
- Medical image preprocessing
- Explainable AI (Grad-CAM)
- Streamlit deployment
- End-to-end ML system design

---

## ⚠️ Disclaimer
This project is for educational purposes only and not for real medical diagnosis.

---

## 👨‍💻 Author
Ananya Pradhan  
BTech IT Student  
AI | ML | Computer Vision Enthusiast  

## 🔗 Connect with Me

If you want to connect or collaborate, feel free to reach out:

👉 💼 LinkedIn: www.linkedin.com/in/ananya-pradhan-10bb462ba

---

## ⭐ Support
If you like this project, please star the repository.


