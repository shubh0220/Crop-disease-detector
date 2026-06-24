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

## 🚀 Live Demo

Try the deployed application:

https://sbs02-crop-disease-detector.hf.space

---

# Overview

Crop Disease Detector is a deep learning-based computer vision project that classifies crop leaf images into **15 crop disease categories** using deep learning and transfer learning techniques.

The project progressed from custom CNN architectures to state-of-the-art transfer learning models and was ultimately deployed as a public Streamlit web application on Hugging Face Spaces.

## Project Pipeline

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
   ↓
Hugging Face Deployment
```

---

# 🏆 Final Results

| Model | Parameters | Accuracy |
|---------|---------:|---------:|
| 🥇 EfficientNet-B0 Transfer Learning | 4,026,763 | **99.39%** |
| 🥈 ResNet18 Transfer Learning | 11,184,207 | 95.32% |
| 🥉 CNN V5 (Custom CNN) | 112,143 | 93.41% |

---

# Dataset

- **Total Images:** 20,638
- **Training Images:** 16,510
- **Testing Images:** 4,128
- **Classes:** 15
- **Train/Test Split:** 80/20

## Disease Categories

- Bell Pepper Bacterial Spot
- Bell Pepper Healthy
- Potato Early Blight
- Potato Healthy
- Potato Late Blight
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Healthy
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Mosaic Virus
- Tomato Septoria Leaf Spot
- Tomato Spider Mites
- Tomato Target Spot
- Tomato Yellow Leaf Curl Virus

---

# 🌐 Streamlit Web Application

The final model was integrated into a Streamlit application that allows users to:

- Upload crop leaf images
- View image preview
- Predict disease category
- Display Top-3 predictions
- View confidence scores
- Access model information

## Features

- Image Upload
- EfficientNet-B0 Inference
- Top-3 Predictions
- Confidence Scores
- Model Information Sidebar
- Real-time Disease Classification

---

# Best Model: EfficientNet-B0 🏆

## Architecture

```text
EfficientNet-B0
      ↓
Dropout(0.2)
      ↓
Linear(1280 → 15)
      ↓
Disease Prediction
```

## Statistics

- **Parameters:** 4,026,763
- **Test Accuracy:** 99.39%
- **Train Accuracy:** 99.68%
- **Generalization Gap:** 0.29%
- **Pretrained on:** ImageNet

---

# Best Custom CNN: CNN V5 ⚡

## Statistics

- **Parameters:** 112,143
- **Accuracy:** 93.41%
- **Parameter Reduction:** 93.84% compared to the baseline CNN

---

# Technologies Used

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
- Git & GitHub
- Hugging Face Spaces

---

# Training Experiments

| Experiment | Parameters | Accuracy |
|---|---:|---:|
| EfficientNet-B0 Transfer Learning | 4,026,763 | **99.39%** |
| ResNet18 Transfer Learning | 11,184,207 | 95.32% |
| CNN V5 | 112,143 | 93.41% |
| BatchNorm + Dropout(0.5) | 1,821,711 | 92.68% |
| BatchNorm + Dropout(0.3) | 1,821,711 | 92.66% |
| Augmentation + Normalize | 1,821,263 | 92.42% |
| Augmentation (10 Epochs) | 1,821,263 | 91.86% |
| CNN V3 | 95,631 | 91.81% |
| CNN V4 | 102,927 | 90.99% |
| BatchNorm Only | 1,821,711 | 90.87% |
| Baseline CNN | 1,821,263 | 89.78% |

---

# Installation

```bash
git clone https://github.com/shubh0220/Crop-disease-detector.git

cd Crop-disease-detector

pip install -r requirements.txt
```

---

# Running the Application

```bash
streamlit run app.py
```

---

# Project Structure

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

# Key Findings

- Transfer learning significantly outperformed custom CNN architectures.
- EfficientNet-B0 achieved the highest accuracy (**99.39%**).
- CNN V5 achieved strong performance with only **112K parameters**.
- Global Average Pooling reduced parameter count by over **93%**.
- EfficientNet-B0 outperformed ResNet18 while using fewer parameters.
- The final model achieved a train-test gap of only **0.29%**.
- External image testing demonstrated strong generalization on unseen disease images.
- Streamlit enabled real-time disease prediction through a web interface.
- Successfully deployed on Hugging Face Spaces.

---

# Future Improvements

- FastAPI Backend
- Mobile Application
- EfficientNet-B1/B2 Comparison
- Hyperparameter Optimization
- Larger Multi-Crop Dataset
- Multi-Disease Detection
- Disease Treatment Recommendations

---

# Author

**Shubh Shukla**

- GitHub: https://github.com/shubh0220
- Hugging Face: https://huggingface.co/SBS02
- Project: Crop Disease Detector