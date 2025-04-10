# Stress Detection Using Machine Learning  
📊🚨 Detecting Stress from Physiological Signals (ECG, EDA, Respiration)

## 📌 Project Overview
This project implements a machine learning model to detect **stress** from physiological signals using the **WESAD dataset**, a benchmark multimodal dataset. The focus is on extracting handcrafted statistical features from the signals (ECG, EDA, Respiration) and using classical machine learning algorithms to classify stress vs. non-stress states.

---

## 🧠 Dataset Description

**WESAD (Wearable Stress and Affect Detection)** is a public dataset that includes physiological signals from 15 subjects. Each subject has data collected from:
- **Chest-worn device** (ECG, EDA, Respiration, EMG, Temperature, Accelerometer)
- **Wrist-worn device** (not used in this project)

We focus on the **ECG, EDA, and Respiration** signals from the **chest sensor**.

- Sampling Rate: `700 Hz`
- Labels:
  - `1, 3, 4` → **Non-Stress**
  - `2, 5, 6, 7` → **Stress**

---

## 🛠️ Feature Extraction

Signals were processed in **60-second windows**. For each window, the following features were extracted:

- **ECG → Heart Rate (HR)** was calculated using R-R intervals from cleaned ECG signal.
- **Statistical features per window**:
  - `mean`, `std`, `min`, `max` for: **HR, EDA, Respiration**
  - Most frequent label in the window

---

## 🧪 Model Training

### ✅ Model: `RandomForestClassifier`  
- Used **class_weight='balanced'** to handle class imbalance
- Trained using extracted features from all subjects
- No use of SMOTE or synthetic sampling (tested and found unnecessary)

### ⚙️ Evaluation Metrics:
```text
Accuracy       : 90%
Precision (1)  : 84%
Recall (1)     : 36%
F1-Score (1)   : 51%
```

### 🔍 Observation:
- Excellent at detecting **non-stress (0)** class.
- Acceptable performance for **stress (1)** class without oversampling.

---

## 📚 Dependencies
- scipy
- neurokit2
- numpy
- pandas
- scikit-learn
- matplotlib
