# 🌱 Crop Disease Detector using Deep Learning

## Overview

Crop Disease Detector is a deep learning-based computer vision project that classifies crop leaf images into 15 different disease categories.

A custom Convolutional Neural Network (CNN) was built using PyTorch and trained on a dataset containing **20,638 images**.

The project focused on systematic model improvements using data augmentation, input normalization, Batch Normalization, and Dropout tuning.

The best model achieved a test accuracy of **92.68%** using BatchNorm and Dropout(0.5).

---

## Dataset

- Total Images: 20,638
- Number of Classes: 15
- Train/Test Split: 80% / 20%
- Training Images: 16,510
- Testing Images: 4,128

### Classes

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

## Final CNN Architecture

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
Flatten
          |
Dropout(0.5)
          |
Linear (115200 → 15)
          |
          ↓
Disease Prediction
```

Total Parameters: **1,821,711**

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- CUDA
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Training Experiments

| Experiment | Configuration | Accuracy |
|---|---|---|
| Baseline CNN | No augmentation, 5 epochs | 89.78% |
| Augmented CNN | Flip + Rotation + ColorJitter, 5 epochs | 86.12% |
| Augmented CNN | Flip + Rotation + ColorJitter, 10 epochs | 91.86% |
| Normalized CNN | Augmentation + Normalize | 92.42% |
| BatchNorm CNN | BatchNorm only | 90.87% |
| CNN V2 ⭐ | BatchNorm + Dropout(0.5) + Normalize | **92.68%** |
| CNN V2 | BatchNorm + Dropout(0.3) + Normalize | 92.66% |

---

## Data Preprocessing & Augmentation

The following preprocessing techniques were applied:

### Augmentation

- Random Horizontal Flip
- Random Rotation (±20°)
- Color Jitter (Brightness and Contrast)

### Normalization

```python
transforms.Normalize(
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5]
)
```

---

## Project Structure

```
crop-disease-detector/
│
├── models/
│   ├── baseline_89_78.pth
│   ├── best_91_86.pth
│   ├── best_normalized_92_42.pth
│   ├── batchnorm_only_90_87.pth
│   ├── batchnorm_dropout_0_3_92_66.pth
│   └── batchnorm_dropout_0_5_92_68.pth ⭐
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

Final Best Model Accuracy:

# ⭐ 92.68%

Key observations:

- Data augmentation required longer training to improve performance.
- Input normalization improved accuracy from 91.86% → 92.42%.
- BatchNorm alone reduced accuracy to 90.87%.
- BatchNorm + Dropout provided the best generalization.
- Dropout(0.5) slightly outperformed Dropout(0.3).

Detailed training losses, class-wise accuracy, and all experiments are available in:

```
results/experiments.md
```

---

## Saved Models

- baseline_89_78.pth
- best_91_86.pth
- best_normalized_92_42.pth
- batchnorm_only_90_87.pth
- batchnorm_dropout_0_3_92_66.pth
- batchnorm_dropout_0_5_92_68.pth ⭐

---

## Future Improvements

- Replace the large fully connected layer with Global Average Pooling (GAP)
- Experiment with deeper CNN architectures
- Apply Transfer Learning using ResNet/EfficientNet
- Build a Streamlit or FastAPI web application
- Deploy the model on Hugging Face

---

## Author

Shubh Shukla