# 🌾 Crop Recommendation System

A robust Machine Learning-based solution that predicts the most suitable crop to cultivate based on soil nutrients and environmental conditions. This project is aimed at helping farmers make data-driven decisions to improve agricultural productivity.

[🔗 Live Demo](https://arbaaz-01-crop-recommendation-model-main-esqjjc.streamlit.app/)

---

## 📌 Overview

This Crop Recommendation System analyzes soil and climate parameters to suggest the best crop for cultivation. It leverages **feature engineering**, **data augmentation**, and **ensemble learning** (LightGBM, CatBoost, Stacking) to achieve high predictive accuracy.

> **Developed as Major Reasearch Project (2025–26)**  
> Department of MCA, **Sardar Patel Institute of Technology, Mumbai**  
> Under the guidance of **Prof. Sakina Salmani**

---

## ✨ Features

- 🔍 Predicts crops based on soil nutrients (N, P, K, pH) and environmental conditions (temperature, humidity, rainfall)
- 🧠 Custom features: `rainfall_humidity`, `climate_index`, `pH_K`, and more
- 🔁 Data augmentation using Gaussian noise
- 🧬 Ensemble models: **LightGBM**, **CatBoost**, and **Stacking**

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Loaded the Crop Recommendation Dataset (Kaggle)
- Created derived features for deeper representation:
  - `rainfall_humidity`, `pH_K`, `N_P`, `temp_sq`, `rainfall_log`, `climate_index`
- Augmented dataset by generating synthetic samples with noise
- Split data: **80% training**, **20% testing**

### 2️⃣ Model Training
- **LightGBM** (tuned via Optuna): Accuracy - **99.84%**
- **CatBoost**: Accuracy - **99.69%**
- **Stacking Ensemble**: Combines LightGBM & CatBoost with Logistic Regression: Accuracy - **99.84%**

### 3️⃣ Evaluation
- Evaluation Metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **Top Features (via SHAP)**:
  - Humidity (0.34)
  - Phosphorus (0.27)
  - Rainfall (0.26)

### 4️⃣ Deployment
- Built frontend using **Streamlit**
- Deployed on **Streamlit Cloud** for public access

---

## 🚀 Installation

### 🔁 Clone the Repository
```bash
git clone https://github.com/arbaaz-01/Crop-Recommendation-Model.git
cd Crop-Recommendation-Model
```




### 📦 Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
````

### 📋 Requirements include:

* pandas 
* numpy
* scikit-learn
* lightgbm
* catboost
* optuna
* streamlit
* shap
* matplotlib
* pickle

> 💡 You can also install them individually if needed.

---

## 📂 Dataset

Place the `Crop_recommendation.csv` file in the **root folder** of the project.

📥 Download from [Kaggle Dataset – Crop Recommendation](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

---

## 💻 Usage

### ▶️ Run Locally

```bash
streamlit run main.py
```

Then visit: [http://localhost:port_add](http://localhost:port_add)

---

### 📥 Input Parameters

* **Soil Nutrients**:
  Nitrogen (N), Phosphorus (P), Potassium (K), pH
* **Environmental Conditions**:
  Temperature, Humidity, Rainfall

---

### 📤 Output

Predicted crop best suited for the given conditions.

---


### 🔍 Top Features (SHAP):

* Humidity (0.34)
* Phosphorus (0.27)
* Rainfall (0.26)

---

## 👨‍💻 Contributors

* **Arbaz Ali Shaikh**
* **Tiwari Prashant**
* **Pranav Shinde**

**Guide**: Prof. Sakina Salmani

---

## 🏛️ Developed At

**Sardar Patel Institute of Technology (SPIT)**
Department of CSE
Academic Year: **2025–26**
