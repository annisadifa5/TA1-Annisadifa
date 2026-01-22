# TA 1 – Orchid Image Classification Using ResNet50

## 📌 Project Overview
This project is part of **Tugas Akhir 1 (TA 1)** and focuses on the development of a **machine learning image classification system** for identifying orchid species using the **ResNet50** deep learning architecture. The system aims to classify orchid images accurately by leveraging convolutional neural networks (CNN) and transfer learning.

The dataset used in this project consists of two sources:  
1. **Primary data** collected directly from field observations by capturing orchid images manually.  
2. **Secondary data** obtained from a public dataset available on **Kaggle**.

---

## 🎯 Objectives
- To design and implement an image classification model for orchids.
- To apply **ResNet50** as a transfer learning model for image classification.
- To compare and utilize both **field-collected data** and **public datasets**.
- To evaluate model performance using appropriate metrics.

---

## 🧠 Methodology
The system development follows these main stages:
1. **Data Collection**
   - Field data: Orchid images captured manually using a camera.
   - Public data: Orchid image dataset sourced from Kaggle.
2. **Data Preprocessing**
   - Image resizing and normalization.
   - Dataset labeling and class balancing.
   - Data splitting (training, validation, testing).
3. **Model Development**
   - Implementation of **ResNet50** with transfer learning.
   - Fine-tuning selected layers to improve accuracy.
4. **Training and Evaluation**
   - Model training using training data.
   - Evaluation using accuracy, loss, and confusion matrix.
5. **Result Analysis**
   - Performance comparison and analysis.
   - Identification of strengths and limitations of the model.

---

## 📂 Dataset
### 1. Field Dataset
- Images collected directly by the researcher.
- Captured under real environmental conditions.
- Represents real-world variations such as lighting and background.

### 2. Public Dataset (Kaggle)
- Orchid image dataset obtained from Kaggle.
- Used to enrich data diversity and improve model generalization.

---

## 🏗️ Model Architecture
- **Base Model**: ResNet50
- **Approach**: Transfer Learning
- **Framework**: TensorFlow / Keras
- **Task**: Multi-class image classification

---

## 📊 Evaluation Metrics
- Accuracy
- Loss
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

---

## 🛠️ Tools & Technologies
- Python
- Google Colab
- TensorFlow / Keras
- NumPy & Pandas
- Matplotlib / Seaborn
- GitHub

---

## 📁 Project Structure
```text
├── dataset/
│   ├── field_data/
│   └── kaggle_data/
├── notebooks/
│   └── training_resnet50.ipynb
├── models/
├── results/
└── README.md
