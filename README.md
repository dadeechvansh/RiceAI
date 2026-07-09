<div align="center">

# 🌾 RiceAI

### AI Powered Rice Variety Classification

*Predicting rice grain varieties with Machine Learning — in real time.*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Visit_App-2ea44f?style=for-the-badge)](https://riceai.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](#)

<br>

[![Stars](https://img.shields.io/github/stars/dadeechvansh/RiceAI?style=social)](#)
[![Forks](https://img.shields.io/github/forks/dadeechvansh/RiceAI?style=social)](#)
[![Watchers](https://img.shields.io/github/watchers/dadeechvansh/RiceAI?style=social)](#)

</div>

---

## 🚀 Live Demo

<div align="center">

### 🌐 The application is **fully deployed and live** — no setup required!

## 👉 **[https://riceai.onrender.com/](https://riceai.onrender.com/)** 👈

Simply visit the link above, enter grain measurements, and get an instant AI-powered prediction with confidence scores.

> ⚡ **Note:** The app is hosted on Render's free tier, so the first load may take a few seconds to spin up. Thank you for your patience!

</div>

---

## 📖 Overview

**RiceAI** is an end-to-end Machine Learning web application that predicts whether a rice grain belongs to the **Cammeo** or **Osmancik** variety, using **seven physical characteristics** extracted from grain imagery.

The project goes beyond a simple model — it **compares multiple machine learning algorithms**, evaluates them on real performance metrics, and **automatically selects the best-performing model** for deployment.

The final model is served through a **FastAPI** backend, wrapped in a clean, responsive web interface that delivers **real-time predictions with confidence scores**.

🔗 **Try it now → [riceai.onrender.com](https://riceai.onrender.com/)**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Multi-Model Comparison** | Trains and evaluates 5 different ML algorithms |
| 🏆 **Automatic Best Model Selection** | Picks the highest-performing model automatically |
| ⚡ **Real-Time Predictions** | Instant classification via a live web app |
| 📊 **Confidence Scores** | Every prediction includes a probability score |
| 🎨 **Clean Web Interface** | Built with responsive HTML5, CSS3 & JavaScript |
| 📈 **Rich Visual Analytics** | Correlation heatmaps, feature importance & more |
| ☁️ **Fully Deployed** | Live on Render — accessible from anywhere |

---

## 📊 Dataset

<div align="center">

| Attribute | Value |
|---|---|
| 📦 Total Samples | **3,810** |
| 🔢 Features | **7** |
| 🏷️ Classes | **Cammeo**, **Osmancik** |
| 📁 Format | CSV |

</div>

The dataset consists of morphological measurements extracted from images of rice grains, capturing geometric properties such as area, perimeter, axis lengths, eccentricity, and more — enough signal for a model to reliably distinguish between the two varieties.

---

## 🔬 Machine Learning Pipeline

```
🌾 Rice Dataset
      ↓
🧹 Data Cleaning
      ↓
📏 Feature Scaling
      ↓
✂️ Train-Test Split
      ↓
🤖 Model Training
      ↓
⚖️ Model Comparison
      ↓
🏆 Best Model Selection
      ↓
💾 Model Saving
      ↓
🚀 FastAPI Backend
      ↓
🔮 Live Prediction
```

---

## 🌐 Web Application Workflow

```
👤 User
      ↓
📝 Enter Grain Measurements
      ↓
🚀 FastAPI Backend
      ↓
📦 Load Model
      ↓
📏 Feature Scaling
      ↓
🌳 Random Forest Prediction
      ↓
📊 Confidence Score
      ↓
✅ Display Prediction
```

---

## 🤖 Models Compared

<div align="center">

| # | Model | Type |
|---|---|---|
| 1️⃣ | Logistic Regression | Linear |
| 2️⃣ | K-Nearest Neighbors | Instance-Based |
| 3️⃣ | Decision Tree | Tree-Based |
| 4️⃣ | Support Vector Machine | Kernel-Based |
| 5️⃣ | **Random Forest** 🏆 | Ensemble |

</div>

---

## 🏆 Best Model

<div align="center">

### 🌳 Random Forest Classifier

| Metric | Score |
|---|---|
| ✅ **Accuracy** | **91.99%** |

*Selected automatically after cross-model evaluation for delivering the best balance of accuracy and generalization.*

</div>

---

## 🛠️ Technology Stack

<div align="center">

### 🧠 Machine Learning
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

### ⚙️ Backend
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Jinja2](https://img.shields.io/badge/Jinja2-B41717?style=flat-square&logo=jinja&logoColor=white)

### 🎨 Frontend
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)

### ☁️ Deployment
![Render](https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white)

</div>

---

## 📁 Project Structure

```
RiceAI/
│
├── app.py                          # FastAPI application entry point
├── train.py                        # Model training pipeline
├── predict.py                      # Prediction logic
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Ignored files config
│
├── data/
│   └── riceClassification.csv      # Raw dataset
│
├── models/
│   ├── best_model.pkl              # Serialized best-performing model
│   └── scaler.pkl                  # Feature scaler
│
├── results/
│   └── results.json                # Model evaluation metrics
│
├── templates/
│   ├── base.html                   # Base HTML template
│   ├── index.html                  # Landing page
│   └── predictor.html              # Prediction interface
│
├── static/
│   ├── css/                        # Stylesheets
│   ├── js/                         # Client-side scripts
│   └── images/                     # UI assets
│
├── images/
│   ├── accuracy_comparison.png
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   └── feature_importance.png
│
└── screenshots/
    ├── home.png
    ├── predictor.png
    ├── prediction-result.png
    ├── model-performance.png
    └── developer.png
```

---

## 📸 Application Screenshots

<div align="center">

### 🏠 Home Page
![Home](screenshots/home.png)

### 🔮 Prediction Page
![Prediction](screenshots/predictor.png)

### ✅ Prediction Result
![Prediction Result](screenshots/prediction-result.png)

### 📊 Model Performance
![Model Performance](screenshots/model-performance.png)


</div>

---

## 📈 Performance Graphs

<div align="center">

### 📊 Accuracy Comparison
![Accuracy Comparison](images/accuracy_comparison.png)

### 🥧 Class Distribution
![Class Distribution](images/class_distribution.png)

### 🔷 Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### 🌡️ Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)

### 📉 Feature Distributions
![Feature Distributions](images/feature_distributions.png)

### 🌟 Feature Importance
![Feature Importance](images/feature_importance.png)

</div>

---

## 🔮 Future Improvements

- 🧬 Add support for additional rice varieties beyond Cammeo & Osmancik
- 📷 Enable image-based grain classification (upload a photo instead of manual measurements)
- 🧠 Experiment with deep learning models (CNNs) for improved accuracy
- 🌍 Add multi-language support for the web interface
- 📱 Build a dedicated mobile-friendly PWA experience
- 🔐 Add user authentication and prediction history tracking
- 📡 Expose a public REST API for third-party integrations

---

---

# 👨‍💻 About Developer

<div align="center">

## Vansh Dadeech

**B.Tech Computer Science & Engineering (Artificial Intelligence & Machine Learning)**  
Model Institute of Engineering and Technology (MIET), Jammu

Passionate about building end-to-end AI applications that combine **Machine Learning**, **Backend Development**, and **Modern Web Technologies**. I enjoy transforming data-driven ideas into real-world solutions with clean design, scalable architecture, and practical deployment.

Currently exploring **Machine Learning**, **FastAPI**, **Data Science**, and **Artificial Intelligence**, while continuously working on projects that strengthen both technical skills and problem-solving abilities.

<br>

[![🌐 Live Demo](https://img.shields.io/badge/🌐_Live_Demo-RiceAI-2ea44f?style=for-the-badge)](https://riceai.onrender.com/)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/vanshdadeech)

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/dadeechvansh)

</div>

---

<div align="center">

### ⭐ Thanks for visiting RiceAI!

If you found this project useful or interesting, consider giving it a ⭐ on GitHub.

**Feedback, suggestions, and contributions are always welcome.**

</div>