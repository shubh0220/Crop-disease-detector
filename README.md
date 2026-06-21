# 🌱 Crop Disease Detector using Deep Learning

## Overview

Crop Disease Detector is a deep learning-based computer vision project that classifies crop leaf images into 15 different disease categories.

A custom Convolutional Neural Network (CNN) was built using PyTorch and trained on a dataset containing 20,638 images.

The best model achieved a test accuracy of **92.42%** after applying data augmentation, input normalization, and training for 10 epochs.

---

## Dataset

- Total Images: 20,638
- Number of Classes: 15
- Train/Test Split: 80% / 20%
- Training Images: 16,510
- Testing Images: 4,128

---

## Model Architecture

```
Input Image (3×256×256)
          |
          ↓
Conv2D (3 → 32)
          |
    ReLU + MaxPool
          |
          ↓
Conv2D (32 → 64)
          |
    ReLU + MaxPool
          |
          ↓
Conv2D (64 → 128)
          |
    ReLU + MaxPool
          |
          ↓
Flatten
          |
          ↓
Linear (115200 → 15)
          |
          ↓
Disease Prediction
```

Total Parameters: **1,821,263**

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- CUDA
- NumPy
- Matplotlib
- Scikit-learn

---

## Training Experiments

| Experiment | Configuration | Accuracy |
|---|---|---|
| Baseline CNN | No augmentation, 5 epochs | 89.78% |
| Augmented CNN | Flip + Rotation + ColorJitter, 5 epochs | 86.12% |
| Augmented CNN | Flip + Rotation + ColorJitter, 10 epochs | 91.86% |
| Normalized CNN ⭐ | Augmentation + Normalize + 10 epochs | **92.42%** |

---

## Data Augmentation

The following augmentations were applied to improve model generalization:

- Random Horizontal Flip
- Random Rotation (±20°)
- Color Jitter (Brightness and Contrast)

---

## Project Structure

```
crop-disease-detector/
│
├── data/
├── models/
│   ├── baseline_89_78.pth
│   └── best_91_86.pth
├── notebooks/
├── results/
│   └── experiments.md
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd crop-disease-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU support, install the CUDA-compatible PyTorch version separately.

---

## Results

Final Model Accuracy:

**92.42%**

Key improvements:

- Overall accuracy improved from 91.86% → 92.42%
- Class 6 improved from 63.72% → 74.03%
- Better performance on difficult disease categories after normalization

The project demonstrates how proper experimentation with data augmentation and training strategies can significantly improve CNN performance.

---

## Saved Models

- baseline_89_78.pth
- best_91_86.pth
- best_normalized_92_42.pth ⭐

## Future Improvements

- Add Dropout
- Replace the custom CNN with ResNet/EfficientNet
- Build a Flask/FastAPI web application
- Deploy the model to the cloud

---

## Author

Shubh Shukla