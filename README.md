<div align="center">

# 🌾 RiceAI

### AI-Powered Rice Variety Classification using Machine Learning

Predict rice grain varieties using a **Random Forest Classifier** trained on **3,810 morphological samples**, achieving **91.99% classification accuracy** through an interactive FastAPI web application.

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi">
<img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn">
<img src="https://img.shields.io/badge/Accuracy-91.99%25-success?style=for-the-badge">

</p>

🚀 **Live Demo:** Coming Soon

</div>

---

# Preview

![Home](screenshots/home.png)

---

# Overview

RiceAI is an end-to-end Machine Learning web application that predicts the variety of a rice grain using seven morphological measurements.

The project combines a trained **Random Forest Classifier**, a **FastAPI backend**, and a modern responsive frontend to provide real-time predictions with confidence scores.

---

# Features

- 🌾 Random Forest Classifier
- ⚡ FastAPI Backend
- 🎯 91.99% Model Accuracy
- 📊 Interactive Model Performance Dashboard
- 📈 Confidence Score Visualization
- 🎨 Modern Responsive User Interface
- 📱 Mobile Friendly Design
- 🧠 Real-Time Machine Learning Inference

---

# Screenshots

## Home Page

![Home](screenshots/home.png)

---

## Prediction Dashboard

![Prediction](screenshots/predictor.png)

---

## Prediction Result

![Result](screenshots/prediction-result.png)

---

## Model Performance

![Dashboard](screenshots/model-performance.png)

---

## Developer

![Developer](screenshots/developer.png)

---

# Technology Stack

| Category | Technologies |
|-----------|--------------|
| Frontend | HTML5, CSS3, JavaScript |
| Backend | FastAPI, Jinja2 Templates |
| Machine Learning | Scikit-learn, Pandas, NumPy |
| Model | Random Forest Classifier |
| Serialization | Joblib |
| Deployment | Render (Planned) |

---

# Dataset

- **Samples:** 3810
- **Features:** 7
- **Classes:** Cammeo, Osmancik

### Input Features

- Area
- Perimeter
- Major Axis Length
- Minor Axis Length
- Eccentricity
- Convex Area
- Extent

---

# Machine Learning Pipeline

```
Rice Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Scaling
      │
      ▼
Random Forest Training
      │
      ▼
Model Serialization
      │
      ▼
FastAPI Backend
      │
      ▼
Interactive Web Interface
```

---

# Model Performance

| Algorithm | Accuracy |
|------------|---------:|
| Logistic Regression | 91.50% |
| Decision Tree | 91.20% |
| KNN | 91.60% |
| SVM | 91.90% |
| **Random Forest** | **91.99%** |

---

# Project Structure

```text
RiceAI/
│
├── app.py
├── predict.py
├── train.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── results/
│   └── results.json
│
├── templates/
│   ├── base.html
│   ├── index.html
│   └── predictor.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
└── screenshots/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/dadeechvansh/RiceAI.git
```

Navigate to the project

```bash
cd RiceAI
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
uvicorn app:app --reload
```

Open

```
http://127.0.0.1:8000
```

---

# API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Home Page |
| GET | `/predict` | Prediction Interface |
| POST | `/predict` | Predict Rice Variety |

---

# Future Improvements

- CNN-based image classification
- Support for additional rice varieties
- Docker deployment
- Prediction history
- User authentication
- Cloud deployment

---
---

<h2 align="center">👨‍💻 Developer</h2>

<div align="center">



<h3>Vansh Dadeech</h3>

<b>B.Tech Computer Science (Artificial Intelligence & Machine Learning)</b>

Model Institute of Engineering & Technology (MIET), Jammu

<br><br>

Passionate about Machine Learning, Artificial Intelligence, FastAPI,
and building intelligent web applications using Python and Scikit-learn.

<br><br>

<a href="https://github.com/dadeechvansh">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
</a>

<a href="https://linkedin.com/in/vanshdadeech">
<img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>

<a href="https://instagram.com/vanxh_77">
<img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"/>
</a>

</div>

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star!

Made with ❤️ using **Python**, **FastAPI**, and **Scikit-learn**

</div>