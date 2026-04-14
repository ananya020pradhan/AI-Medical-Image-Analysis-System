# 🧠 AI-Powered Medical Image Analysis System

An end-to-end Deep Learning project that classifies Chest X-ray images into **Normal, Pneumonia, and COVID-19** using a Convolutional Neural Network (CNN). The system also provides a simple CLI and Streamlit-based web interface for real-time prediction.

---

## 🚀 Project Overview

This project demonstrates how Artificial Intelligence can assist in medical diagnosis by analyzing chest X-ray images and predicting diseases quickly and efficiently.

It is built using **Deep Learning (CNN)** and designed as a **portfolio-ready, industry-style ML project** using publicly available datasets.

---

## 🎯 Problem Statement

Manual diagnosis of chest X-rays:
- Requires expert radiologists  
- Is time-consuming  
- Can be affected by human fatigue  

### 💡 Solution:
An AI system that:
- Automatically analyzes X-ray images  
- Classifies conditions (Normal / Pneumonia / COVID-19)  
- Assists in faster diagnosis  
- Reduces workload for healthcare professionals  

---

## 🏥 Medical Use Case

- Chest X-ray screening  
- Disease detection support system  
- Radiology assistance tool  
- Educational AI healthcare project  

---

## 🧠 Model Details

- Model Type: Convolutional Neural Network (CNN)  
- Framework: TensorFlow / Keras  
- Input Size: 224 x 224 images  
- Output Classes:
  - NORMAL  
  - PNEUMONIA  
  - COVID-19  

---

## 🏗️ System Architecture

Input Image → Preprocessing → CNN Model → Feature Extraction → Softmax Classification → Prediction Output → User Interface (CLI / Streamlit)

---

## 📁 Project Structure

AI-Medical-Image-Analysis-System/
├── data/                  Dataset (Chest X-ray images)
├── models/
│   └── model.h5           Trained CNN model
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── app/
│   └── app.py             Streamlit web app
├── main.py                CLI interface
├── requirements.txt
├── .gitignore
└── README.md

---

## ⚙️ Installation & Setup

### Clone Repository
git clone https://github.com/ananya020pradhan/AI-Medical-Image-Analysis-System.git  
cd AI-Medical-Image-Analysis-System  

### Create Virtual Environment
python -m venv venv  

source venv/bin/activate   (Mac/Linux)  
venv\Scripts\activate      (Windows)  

### Install Dependencies
pip install -r requirements.txt  

---

## 🚀 How to Run

### Train Model
python -m src.train  

### Run CLI Prediction
python main.py  

### Run Streamlit App
streamlit run app/app.py  

---

## 🖥️ Features

✔ AI-based medical image classification  
✔ Detects Normal, Pneumonia, COVID-19  
✔ Real-time image upload system  
✔ Confidence score output  
✔ Confusion matrix visualization  
✔ Modular ML pipeline  

---

## 📊 Model Performance

- Accuracy: ~85–95% (depends on dataset)
- Metrics:
  - Confusion Matrix
  - Classification Report
  - Accuracy Graph

---

## 🧪 Technologies Used

Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Scikit-learn, Streamlit

---

## 📌 Key Learnings

- Deep Learning for medical imaging  
- CNN architecture design  
- Image preprocessing techniques  
- Model evaluation  
- Streamlit deployment  
- End-to-end ML pipeline  

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and is NOT a medical diagnostic tool.

---

## 👨‍💻 Author

Ananya Pradhan  
B.Tech Information Technology  
AI / ML Enthusiast  

## 🔗 Connect with Me

- 💼 LinkedIn: www.linkedin.com/in/ananya-pradhan-10bb462ba

---

## 🌟 Future Improvements

- Grad-CAM explainability  
- More disease categories  
- Cloud deployment  
- Mobile-friendly UI  
