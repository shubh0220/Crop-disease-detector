# 🌱 Crop Disease Detector using Deep Learning

## Overview

Crop Disease Detector is a deep learning-based computer vision project that classifies crop leaf images into **15 different disease categories**.

The project explores three major approaches:

- **Custom CNN Architecture Design**
- **CNN Optimization using GAP, BatchNorm, and Dropout**
- **Transfer Learning using ResNet18 and EfficientNet-B0**

A systematic experimentation process was followed involving:

- Data Augmentation
- Input Normalization
- Batch Normalization
- Dropout Tuning
- Global Average Pooling (GAP)
- Transfer Learning
- Architecture Comparison

---

## 🏆 Final Results

| Model | Parameters | Accuracy |
|---------|---------:|---------:|
| 🥇 EfficientNet-B0 Transfer Learning | 4,026,763 | **99.39%** |
| 🥈 ResNet18 Transfer Learning | 11,184,207 | 95.32% |
| 🥉 CNN V5 (Custom CNN) | 112,143 | 93.41% |

The project demonstrates how modern transfer learning architectures can significantly outperform handcrafted CNNs while maintaining strong parameter efficiency.

---

# Dataset

- **Total Images:** 20,638
- **Number of Classes:** 15
- **Train/Test Split:** 80% / 20%
- **Training Images:** 16,510
- **Testing Images:** 4,128

## Classes

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

- Parameters: **4,026,763**
- Test Accuracy: **99.39%**
- Train Accuracy: **99.68%**
- Generalization Gap: **0.29%**
- ImageNet Pretrained Weights

---

# Best Custom CNN: CNN V5 ⚡

## Architecture

```text
Input Image (3×256×256)
          ↓
Conv2D (3 → 32)
          ↓
BatchNorm2D
          ↓
ReLU
          ↓
MaxPool
          ↓
Conv2D (32 → 64)
          ↓
BatchNorm2D
          ↓
ReLU
          ↓
MaxPool
          ↓
Conv2D (64 → 128)
          ↓
BatchNorm2D
          ↓
ReLU
          ↓
MaxPool
          ↓
Global Average Pooling
          ↓
Linear (128 → 128)
          ↓
ReLU
          ↓
Linear (128 → 15)
```

### Statistics

- Parameters: **112,143**
- Accuracy: **93.41%**
- Parameter Reduction: **93.84%** compared to the original CNN

---

# Technologies Used

- Python
- PyTorch
- TorchVision
- CUDA
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

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

# Data Preprocessing

## Augmentation

- Random Horizontal Flip
- Random Rotation (±20°)
- Color Jitter

## CNN Normalization

```python
transforms.Normalize(
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5]
)
```

## Transfer Learning Normalization

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

---

# Key Findings

- Data augmentation improved model robustness when trained for sufficient epochs.
- Input normalization stabilized training and improved performance.
- Global Average Pooling reduced parameter count by over 93%.
- CNN V5 achieved strong performance with only 112K parameters.
- ResNet18 demonstrated the effectiveness of transfer learning.
- EfficientNet-B0 achieved the highest overall accuracy.
- EfficientNet-B0 outperformed ResNet18 by **4.07%** while using fewer parameters.
- Train accuracy (**99.68%**) and test accuracy (**99.39%**) remained extremely close, indicating strong generalization.

---

# Project Structure

```text
crop-disease-detector/
│
├── models/
│   ├── cnn_v5_gap_fc128_93_41.pth
│   ├── resnet18_95_32.pth
│   ├── efficientnet_b0_99_39.pth
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_pytorch.ipynb
│   ├── 03_transfer_learning.ipynb
│   └── 04_efficientnet.ipynb
│
├── results/
│   └── experiments.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

```bash
git clone https://github.com/shubh0220/Crop-disease-detector.git
cd Crop-disease-detector
pip install -r requirements.txt
```

For GPU acceleration, install the CUDA-compatible PyTorch version separately.

---

# Results Summary

| Model | Parameters | Accuracy |
|---------|---------:|---------:|
| 🥇 EfficientNet-B0 Transfer Learning | 4,026,763 | **99.39%** |
| 🥈 ResNet18 Transfer Learning | 11,184,207 | 95.32% |
| 🥉 CNN V5 | 112,143 | 93.41% |

## Best Accuracy

**EfficientNet-B0 Transfer Learning**

- Accuracy: **99.39%**
- Parameters: **4,026,763**

## Most Efficient Model

**CNN V5**

- Accuracy: **93.41%**
- Parameters: **112,143**
- Approximately **36× fewer parameters** than EfficientNet-B0

---

# Future Improvements

- EfficientNet-B1 / B2 Comparison
- Learning Rate Schedulers
- Hyperparameter Optimization
- Streamlit/FastAPI Web Application
- Hugging Face Deployment

---

# Author

**Shubh Shukla**