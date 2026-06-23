# 🌱 Crop Disease Detector using Deep Learning

## Overview

Crop Disease Detector is a deep learning-based computer vision project that classifies crop leaf images into **15 different disease categories**.

The project explores both:

- **Custom CNN Architecture Design**
- **Transfer Learning using ResNet18**

A systematic experimentation process was followed involving:

- Data Augmentation
- Input Normalization
- Batch Normalization
- Dropout Tuning
- Global Average Pooling (GAP)
- Transfer Learning

### Final Results

| Model | Parameters | Accuracy |
|---------|---------:|---------:|
| 🏆 ResNet18 Transfer Learning | 11,184,207 | **95.32%** |
| ⚡ CNN V5 (Custom CNN) | 112,143 | **93.41%** |

The project demonstrates the trade-off between **maximum accuracy** and **parameter efficiency**.

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

# Best Custom CNN Architecture (CNN V5)

```text
Input Image (3×256×256)
          |
          ↓
Conv2D (3 → 32)
          |
BatchNorm2D
          |
ReLU
          |
MaxPool
          |
          ↓
Conv2D (32 → 64)
          |
BatchNorm2D
          |
ReLU
          |
MaxPool
          |
          ↓
Conv2D (64 → 128)
          |
BatchNorm2D
          |
ReLU
          |
MaxPool
          |
          ↓
Feature Maps (128×30×30)
          |
Global Average Pooling
          |
Flatten
          |
Linear (128 → 128)
          |
ReLU
          |
Linear (128 → 15)
          |
          ↓
Disease Prediction
```

### CNN V5 Statistics

- Parameters: **112,143**
- Accuracy: **93.41%**
- Parameter Reduction: **93.84%** compared to the original CNN

---

# Transfer Learning (ResNet18)

A pretrained ResNet18 model was fine-tuned on the crop disease dataset using ImageNet weights.

### Architecture

```text
ResNet18
    ↓
Linear(512 → 15)
    ↓
Disease Prediction
```

### ResNet18 Statistics

- Parameters: **11,184,207**
- Accuracy: **95.32%**
- Input Size: **224 × 224**
- ImageNet Pretrained Weights

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

| Experiment | Configuration | Parameters | Accuracy |
|---|---|---:|---:|
| Baseline CNN | Original CNN | 1,821,263 | 89.78% |
| Data Augmentation | Flip + Rotation + ColorJitter | 1,821,263 | 91.86% |
| Normalization | Augmentation + Normalize | 1,821,263 | 92.42% |
| BatchNorm Only | BatchNorm + Original FC | 1,821,711 | 90.87% |
| CNN V2 | BatchNorm + Dropout(0.3) | 1,821,711 | 92.66% |
| CNN V2 | BatchNorm + Dropout(0.5) | 1,821,711 | 92.68% |
| CNN V3 | GAP → FC(128→15) | 95,631 | 91.81% |
| CNN V4 | GAP → FC(128→64) → ReLU → FC(64→15) | 102,927 | 90.99% |
| CNN V5 | GAP → FC(128→128) → ReLU → FC(128→15) | 112,143 | 93.41% |
| 🏆 ResNet18 | Transfer Learning | 11,184,207 | **95.32%** |

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

## ResNet18 Normalization

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

---

# Key Findings

- Data augmentation required longer training to become beneficial.
- Input normalization improved CNN performance.
- BatchNorm alone was not effective for the original architecture.
- Dropout improved large fully connected classifiers.
- Global Average Pooling reduced parameters dramatically.
- CNN V5 achieved strong accuracy with only 112K parameters.
- ResNet18 achieved the highest overall accuracy using transfer learning.
- Transfer learning provided a +1.91% improvement over CNN V5.

---

# Project Structure

```text
crop-disease-detector/
│
├── models/
│   ├── cnn_v5_gap_fc128_93_41.pth
│   ├── resnet18_95_32.pth
│   └── ...
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_pytorch.ipynb
│   └── 03_transfer_learning.ipynb
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
| 🏆 ResNet18 Transfer Learning | 11,184,207 | **95.32%** |
| ⚡ CNN V5 | 112,143 | **93.41%** |

## Best Accuracy

**ResNet18 Transfer Learning**
- Accuracy: 95.32%

## Most Efficient Model

**CNN V5**
- Accuracy: 93.41%
- Only 112,143 parameters

---

# Future Improvements

- EfficientNet Transfer Learning
- Learning Rate Schedulers
- Hyperparameter Optimization
- Streamlit/FastAPI Web Application
- Hugging Face Deployment

---

# Author

**Shubh Shukla**