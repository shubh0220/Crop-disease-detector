---
title: Crop Disease Detector
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.58.0"
app_file: app.py
pinned: false
---
# 🌱 Crop Disease Detector using Deep Learning

## Overview

Crop Disease Detector is a deep learning-based computer vision project that classifies crop leaf images into **15 crop disease categories** using deep learning and transfer learning techniques.

The project evolved from custom CNN architectures to state-of-the-art transfer learning models and was finally deployed as an interactive Streamlit web application.

### Project Pipeline

```text
Dataset
   ↓
EDA
   ↓
Custom CNN
   ↓
CNN Optimization
   ↓
ResNet18 Transfer Learning
   ↓
EfficientNet-B0 Transfer Learning
   ↓
Inference Pipeline
   ↓
Streamlit Web Application
```

---

## 🌐 Streamlit Web Application

The final model was integrated into a Streamlit application that allows users to:

- Upload crop leaf images
- View image preview
- Predict disease category
- Display Top-3 predictions
- View prediction confidence scores

### Features

- Image Upload
- EfficientNet-B0 Inference
- Top-3 Predictions
- Confidence Scores
- Model Information Sidebar
- Real-time Disease Classification

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- Streamlit
- CUDA
- NumPy
- Matplotlib
- Scikit-Learn
- Pillow
- Jupyter Notebook

---

## Running the Application

Clone the repository:

```bash
git clone https://github.com/shubh0220/Crop-disease-detector.git
cd Crop-disease-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Project Structure

```text
crop-disease-detector/
│
├── app.py
├── utils.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── cnn_v5_gap_fc128_93_41.pth
│   ├── resnet18_95_32.pth
│   └── efficientnet_b0_99_39.pth
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_pytorch.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_efficientnet.ipynb
│   └── 05_inference.ipynb
│
├── results/
│   └── experiments.md
│
└── data/
```

---

## Key Findings

- Transfer learning significantly outperformed custom CNN architectures.
- EfficientNet-B0 achieved the highest accuracy (**99.39%**).
- CNN V5 achieved strong performance with only **112K parameters**.
- Global Average Pooling reduced model size by over **93%**.
- The final model achieved a train-test gap of only **0.29%**.
- Streamlit enabled real-time disease prediction through a web interface.
- External image testing demonstrated good generalization on unseen disease images.

---

## Future Improvements

- Hugging Face Deployment
- FastAPI Backend
- Mobile Application
- EfficientNet-B1/B2 Comparison
- Hyperparameter Optimization
- Larger Multi-Crop Dataset

# Author

**Shubh Shukla**