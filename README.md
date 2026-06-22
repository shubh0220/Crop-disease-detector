# 🌱 Crop Disease Detector using Deep Learning

## Overview

Crop Disease Detector is a deep learning-based computer vision project that classifies crop leaf images into **15 different disease categories**.

A custom Convolutional Neural Network (CNN) was built using PyTorch and trained on a dataset containing **20,638 images**.

The project followed a systematic experimentation approach involving:
- Data augmentation
- Input normalization
- Batch Normalization
- Dropout tuning
- Global Average Pooling (GAP) based architecture optimization

The final **CNN V5 architecture** achieved a test accuracy of **93.41%** while reducing the model size from **1.82 million parameters to only 112,143 parameters**.

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

# Final CNN Architecture (CNN V5)

```
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

### Model Statistics

- **Total Parameters:** 112,143
- **Test Accuracy:** 93.41%
- **Parameter Reduction:** 93.84% compared to the previous 1.82M parameter CNN

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

| Experiment | Architecture / Configuration | Parameters | Accuracy |
|---|---|---:|---:|
| Baseline CNN | Conv → ReLU → Pool → Flatten → FC | 1,821,263 | 89.78% |
| Data Augmentation | Flip + Rotation + ColorJitter (10 Epochs) | 1,821,263 | 91.86% |
| Normalization | Augmentation + Normalize | 1,821,263 | 92.42% |
| BatchNorm Only | BatchNorm + Original FC | 1,821,711 | 90.87% |
| CNN V2 | BatchNorm + Dropout(0.3) | 1,821,711 | 92.66% |
| CNN V2 | BatchNorm + Dropout(0.5) | 1,821,711 | 92.68% |
| CNN V3 | GAP → FC(128→15) | 95,631 | 91.81% |
| CNN V4 | GAP → FC(128→64) → ReLU → FC(64→15) | 102,927 | 90.99% |
| ⭐ CNN V5 (Best) | GAP → FC(128→128) → ReLU → FC(128→15) | **112,143** | **93.41%** |

---

# Data Preprocessing & Augmentation

## Augmentation

The following augmentation techniques were used to improve generalization:

- Random Horizontal Flip
- Random Rotation (±20°)
- Color Jitter (Brightness and Contrast)

## Normalization

Input images were normalized using:

```python
transforms.Normalize(
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5]
)
```

---

# Key Experimental Findings

- Data augmentation required longer training (10 epochs) to provide performance improvements.
- Input normalization stabilized training and increased accuracy from **91.86% to 92.42%**.
- Batch Normalization alone did not improve the original architecture.
- Dropout improved the large fully connected classifier, with **Dropout(0.5)** performing slightly better than **Dropout(0.3)**.
- Global Average Pooling reduced the model size by more than **93%**.
- A simple GAP classifier achieved competitive accuracy with only **95,631 parameters**.
- Excessive feature compression (**128 → 64**) reduced performance.
- The final CNN V5 preserved the 128-dimensional GAP representation and added a non-linear transformation, achieving the highest accuracy of **93.41%**.

---

# Project Structure

```
crop-disease-detector/
│
├── models/
│   ├── baseline_89_78.pth
│   ├── best_91_86.pth
│   ├── best_normalized_92_42.pth
│   ├── batchnorm_only_90_87.pth
│   ├── batchnorm_dropout_0_3_92_66.pth
│   ├── batchnorm_dropout_0_5_92_68.pth
│   ├── cnn_v3_gap_91_81.pth
│   ├── cnn_v4_gap_fc64_90_99.pth
│   └── cnn_v5_gap_fc128_93_41.pth ⭐
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_pytorch.ipynb
│
├── results/
│   └── experiments.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/shubh0220/Crop-disease-detector.git
cd Crop-disease-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the CUDA-compatible PyTorch version separately.

---

# Final Results

## ⭐ Best Model: CNN V5

### Performance

- **Test Accuracy:** 93.41%
- **Parameters:** 112,143
- **Previous Best CNN:** 92.68% with 1.82M parameters
- **Parameter Reduction:** 93.84%

### Final Architecture

```
GAP → FC(128 → 128) → ReLU → FC(128 → 15)
```

The final model achieved a higher accuracy while being approximately **16× smaller** than the previous best CNN.

Detailed training losses, class-wise accuracies, and all experiments are available in:

```
results/experiments.md
```

---

# Future Improvements

- Apply transfer learning using ResNet and EfficientNet.
- Experiment with learning rate schedulers.
- Perform hyperparameter tuning.
- Build a Streamlit or FastAPI web application.
- Deploy the model using cloud platforms or Hugging Face Spaces.

---

# Author

**Shubh Shukla**