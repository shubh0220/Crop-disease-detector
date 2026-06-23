# Crop Disease Detection - Experiment Log

## Dataset Information

- Total Images: 20,638
- Number of Classes: 15
- Train/Test Split: 80/20
- Training Images: 16,510
- Testing Images: 4,128


# Experiment 1: Baseline CNN (No Augmentation)

## Model Architecture

- Conv2D(3 → 32, kernel_size=3)
- ReLU
- MaxPool(2x2)

- Conv2D(32 → 64, kernel_size=3)
- ReLU
- MaxPool(2x2)

- Conv2D(64 → 128, kernel_size=3)
- ReLU
- MaxPool(2x2)

- Flatten
- Linear(115200 → 15)

Total Parameters: 1,821,263


## Training Configuration

- Epochs: 5
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.001
- Data Augmentation: No


## Training Loss

| Epoch | Loss |
|---|---|
| 1 | 0.9299 |
| 2 | 0.3787 |
| 3 | 0.2232 |
| 4 | 0.1335 |
| 5 | 0.0981 |


## Result

- Test Accuracy: **89.78%**

---

# Experiment 2: CNN with Data Augmentation (5 Epochs)

## Data Augmentation

- RandomHorizontalFlip()
- RandomRotation(20°)
- ColorJitter(brightness=0.2, contrast=0.2)


## Training Configuration

- Epochs: 5
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.001


## Training Loss

| Epoch | Loss |
|---|---|
| 1 | 1.4001 |
| 2 | 0.7532 |
| 3 | 0.6221 |
| 4 | 0.5207 |
| 5 | 0.4649 |


## Result

- Test Accuracy: **86.12%**

### Observation

The model performed worse than the baseline because the augmentation made the training task harder and 5 epochs were not sufficient for convergence.

---

# Experiment 3: CNN with Data Augmentation (10 Epochs)


## Data Augmentation

- RandomHorizontalFlip()
- RandomRotation(20°)
- ColorJitter(brightness=0.2, contrast=0.2)


## Training Configuration

- Epochs: 10
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.001


## Training Loss

| Epoch | Loss |
|---|---|
| 1 | 1.2350 |
| 2 | 0.6930 |
| 3 | 0.5394 |
| 4 | 0.4669 |
| 5 | 0.3907 |
| 6 | 0.3484 |
| 7 | 0.3079 |
| 8 | 0.2962 |
| 9 | 0.2615 |
| 10 | 0.2399 |


## Result

- Test Accuracy: **91.86%** ⭐

### Major Improvements

| Class | Before | After |
|---|---|---|
| Class 0 | 79.31% | 96.10% |
| Class 4 | 82.76% | 96.43% |
| Class 6 | 59.09% | 63.72% |
| Class 10 | 80.75% | 92.24% |

---

# Experiment 4: CNN with Data Augmentation + Normalization (10 Epochs) 


## Data Augmentation & Preprocessing

### Data Augmentation

- RandomHorizontalFlip()
- RandomRotation(20°)
- ColorJitter(brightness=0.2, contrast=0.2)

### Normalization

```python
transforms.Normalize(
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5]
)
```

This converts input pixel values from approximately:

```
0 → 1
```

to:

```
-1 → +1
```

---

## Training Configuration

- Epochs: 10
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.001
- CNN Architecture: Same as previous experiments


---

## Training Loss

| Epoch | Loss |
|---|---|
| 1 | 1.0775 |
| 2 | 0.5928 |
| 3 | 0.4897 |
| 4 | 0.4147 |
| 5 | 0.3516 |
| 6 | 0.3080 |
| 7 | 0.2876 |
| 8 | 0.2489 |
| 9 | 0.2395 |
| 10 | 0.2173 |

---

## Result

Test Accuracy:

**92.42% ⭐

---

## Class-wise Comparison with Previous Best

| Class | Previous | Normalized Model |
|---|---|---|
| Class 0 | 96.10% | 92.07% |
| Class 3 | 86.10% | 90.15% |
| Class 6 | 63.72% | 74.03% |
| Class 8 | 86.32% | 92.23% |
| Class 9 | 91.92% | 94.01% |
| Class 10 | 92.24% | 88.24% |

---

## Observation

Adding normalization improved the overall performance of the CNN.

Overall accuracy:

```
91.86%
   ↓
92.42%
```

Improvement:

```
+0.56%
```

The largest improvement was observed in **Class 6**, which increased from **63.72% to 74.03%**, showing better generalization for difficult classes.

---

## Conclusion

Normalizing the input images helped the network train with a more stable input distribution centered around zero.

This resulted in:
- Better optimization
- Lower final training loss
- Improved test accuracy
- Better performance on difficult classes

The best normalized model was saved as:

```
models/best_normalized_92_42.pth
```
---

# Experiment 5: CNN with BatchNorm + Dropout(0.5) + Normalization (10 Epochs)

## Architecture Changes

### Batch Normalization

Added Batch Normalization after each convolution layer:

```
Conv → BatchNorm → ReLU → MaxPool
```

### Dropout

Added Dropout before the final classification layer:

```python
nn.Dropout(0.5)
```

---

## Training Configuration

- Data Augmentation:
  - RandomHorizontalFlip()
  - RandomRotation(20°)
  - ColorJitter(brightness=0.2, contrast=0.2)

- Input Normalization:

```python
transforms.Normalize(
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5]
)
```

- Epochs: 10
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.001

---

## Training Loss

| Epoch | Loss |
|---|---|
| 1 | 7.428 |
| 2 | 1.994 |
| 3 | 0.787 |
| 4 | 0.545 |
| 5 | 0.467 |
| 6 | 0.418 |
| 7 | 0.393 |
| 8 | 0.359 |
| 9 | 0.325 |
| 10 | 0.296 |

---

## Result

Test Accuracy:

**92.68% ⭐**

---

## Important Improvements

- Class 6 improved from **74.03% → 79.79%**
- Class 8 improved from **92.23% → 96.39%**
- Class 14 achieved **100% accuracy**

---

## Saved Model

```
models/batchnorm_dropout_0_5_92_68.pth
```

---

# Experiment 6: CNN with BatchNorm Only (10 Epochs)

## Architecture

```
Conv → BatchNorm → ReLU → MaxPool

Flatten → Linear
```

---

## Training Loss

| Epoch | Loss |
|---|---|
| 1 | 6.706 |
| 2 | 1.685 |
| 3 | 0.714 |
| 4 | 0.508 |
| 5 | 0.483 |
| 6 | 0.428 |
| 7 | 0.386 |
| 8 | 0.343 |
| 9 | 0.311 |
| 10 | 0.296 |

---

## Result

Test Accuracy:

**90.87%**

---

## Observation

- BatchNorm alone did not improve the model.
- Removing Dropout reduced the overall performance.

---

## Saved Model

```
models/batchnorm_only_90_87.pth
```

---

# Experiment 7: CNN with BatchNorm + Dropout(0.3) + Normalization (10 Epochs)

## Architecture Change

Changed Dropout value:

```python
nn.Dropout(0.5)
```

to:

```python
nn.Dropout(0.3)
```

---

## Training Loss

| Epoch | Loss |
|---|---|
| 1 | 6.936 |
| 2 | 1.763 |
| 3 | 0.823 |
| 4 | 0.512 |
| 5 | 0.422 |
| 6 | 0.381 |
| 7 | 0.340 |
| 8 | 0.323 |
| 9 | 0.297 |
| 10 | 0.277 |

---

## Result

Test Accuracy:

**92.66%**

---

## Observation

- Accuracy was very close to Dropout(0.5).
- Dropout(0.3) improved Class 4 accuracy significantly.
- Dropout(0.5) performed better on Class 6 and Class 8.

---

## Saved Model

```
models/batchnorm_dropout_0_3_92_66.pth
```

---
---

# Experiment 8: CNN V3 with Global Average Pooling (GAP)

## Architecture Changes

Replaced the large fully connected layer with Global Average Pooling.

Old classifier:

```
Flatten
↓
Linear(115200 → 15)
```

New classifier:

```
Global Average Pooling
↓
Flatten
↓
Linear(128 → 15)
```

---

## Motivation

The goal was to reduce the large number of parameters in the final classifier while maintaining good classification performance.

---

## Parameter Comparison

- Previous CNN: 1,821,711 parameters
- CNN V3: 95,631 parameters

Parameter reduction: **94.75%**

---

## Training Loss

| Epoch | Loss |
|---|---|
| 1 | 1.378 |
| 2 | 0.891 |
| 3 | 0.720 |
| 4 | 0.607 |
| 5 | 0.531 |
| 6 | 0.490 |
| 7 | 0.446 |
| 8 | 0.413 |
| 9 | 0.386 |
| 10 | 0.371 |

---

## Results

Test Accuracy:

**91.81%**

### Observation

- The model became approximately 19× smaller.
- Accuracy dropped by only 0.87% compared to the previous best model.
- Global Average Pooling successfully reduced overfitting and model size but removed some spatial information.

---

## Saved Model

```
models/cnn_v3_gap_91_81.pth
```

---

# Experiment 9: CNN V4 (GAP + FC(128→64) + ReLU + Dropout)

## Architecture

```
GAP
 ↓
FC(128 → 64)
 ↓
ReLU
 ↓
Dropout(0.3)
 ↓
FC(64 → 15)
```

---

## Motivation

Added a small hidden classifier to learn complex combinations of GAP features.

---

## Training Loss

| Epoch | Loss |
|---|---|
| 1 | 1.566 |
| 2 | 1.071 |
| 3 | 0.911 |
| 4 | 0.809 |
| 5 | 0.722 |
| 6 | 0.642 |
| 7 | 0.598 |
| 8 | 0.562 |
| 9 | 0.534 |
| 10 | 0.491 |

---

## Results

Test Accuracy:

**88.47%**

### Observation

- Dropout significantly reduced the performance of the small GAP model.
- The model suffered from underfitting due to reduced capacity and strong regularization.

---

## Result

This experiment showed that adding Dropout to a small model is not always beneficial.

---

# Experiment 10: CNN V4 without Dropout

## Architecture

```
GAP
 ↓
FC(128 → 64)
 ↓
ReLU
 ↓
FC(64 → 15)
```

---

## Motivation

To check whether Dropout was responsible for the poor performance of Experiment 9.

---

## Training Loss

| Epoch | Loss |
|---|---|
| 1 | 1.379 |
| 2 | 0.859 |
| 3 | 0.672 |
| 4 | 0.558 |
| 5 | 0.491 |
| 6 | 0.437 |
| 7 | 0.400 |
| 8 | 0.370 |
| 9 | 0.343 |
| 10 | 0.312 |

---

## Results

Test Accuracy:

**90.99%**

### Observation

- Removing Dropout improved accuracy by 2.52%.
- The model still performed worse than CNN V3, suggesting that reducing the feature dimension from 128 to 64 removed useful information.

---

## Saved Model

```
models/cnn_v4_gap_fc64_90_99.pth
```

---

# Experiment 11: CNN V5 (GAP + FC(128→128) + ReLU + FC(128→15)) 

## Architecture

```
GAP
 ↓
FC(128 → 128)
 ↓
ReLU
 ↓
FC(128 → 15)
```

---

## Motivation

Previous experiments showed that:
- GAP was effective.
- Compressing 128 features into 64 reduced performance.

This experiment keeps the full 128-dimensional feature representation while adding a non-linear transformation.

---

## Parameter Count

- CNN V5: 112,143 parameters

Compared to the previous best CNN:

- CNN V2: 1,821,711 parameters

Parameter reduction: **93.84%**

---

## Training Loss

| Epoch | Loss |
|---|---|
| 1 | 1.359 |
| 2 | 0.810 |
| 3 | 0.646 |
| 4 | 0.544 |
| 5 | 0.464 |
| 6 | 0.400 |
| 7 | 0.381 |
| 8 | 0.344 |
| 9 | 0.325 |
| 10 | 0.302 |

---

## Results

Test Accuracy:

**93.41% **

### Key Improvements

- Achieved the highest accuracy of all experiments.
- Reduced the model size by approximately 16× compared to CNN V2.
- Maintained the complete 128 GAP features while allowing the network to learn better feature interactions.

---

## Saved Model

```
models/cnn_v5_gap_fc128_93_41.pth
```

---
# Experiment 12: Transfer Learning with ResNet18

## Objective

After optimizing a custom CNN architecture using Batch Normalization, Dropout, and Global Average Pooling (GAP), the next step was to compare its performance against a pretrained deep learning model.

This experiment uses **ResNet18 pretrained on ImageNet** and fine-tunes it on the crop disease dataset.

---

## Why Transfer Learning?

Training a CNN from scratch requires learning all visual features from random initialization.

A pretrained ResNet18 has already learned useful low-level and high-level visual features such as:

- Edges
- Corners
- Textures
- Shapes
- Object patterns

from over one million ImageNet images.

Instead of learning from scratch, the model can adapt these features to crop disease classification.

---

## Dataset

- Total Images: 20,638
- Classes: 15
- Train Split: 16,510
- Test Split: 4,128

---

## Data Preprocessing

### Data Augmentation

```python
transforms.RandomHorizontalFlip()
transforms.RandomRotation(20)
transforms.ColorJitter(
    brightness=0.2,
    contrast=0.2
)
```

### ImageNet Normalization

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### Input Size

```python
224 × 224
```

---

## Model Architecture

### Original ResNet18

```text
ResNet18
    ↓
Linear(512 → 1000)
```

### Modified ResNet18

```text
ResNet18 Feature Extractor
            ↓
512 Features
            ↓
Linear(512 → 15)
            ↓
Disease Prediction
```

The final classification layer was replaced to support the 15 crop disease classes.

---

## Training Configuration

### Optimizer

```python
Adam(lr=0.001)
```

### Loss Function

```python
CrossEntropyLoss()
```

### Epochs

```python
10
```

### Batch Size

```python
32
```

---

## Parameter Count

### ResNet18

```text
11,184,207 Parameters
```

### CNN V5

```text
112,143 Parameters
```

ResNet18 contains approximately **100× more parameters** than the custom CNN V5 model.

---

## Training Loss

| Epoch | Average Loss |
|---------|---------:|
| 1 | 0.4059 |
| 2 | 0.1911 |
| 3 | 0.1484 |
| 4 | 0.1290 |
| 5 | 0.1180 |
| 6 | 0.0967 |
| 7 | 0.0812 |
| 8 | 0.0943 |
| 9 | 0.0677 |
| 10 | 0.0794 |

---

## Results

### Test Accuracy

**95.32%**

---

## Class-wise Accuracy

| Class | Accuracy |
|---------|---------:|
| 0 | 99.47% |
| 1 | 99.36% |
| 2 | 96.65% |
| 3 | 100.00% |
| 4 | 92.59% |
| 5 | 97.15% |
| 6 | 74.27% |
| 7 | 94.50% |
| 8 | 98.84% |
| 9 | 96.88% |
| 10 | 84.99% |
| 11 | 98.94% |
| 12 | 98.57% |
| 13 | 95.56% |
| 14 | 96.63% |

---

## Comparison with CNN V5

| Model | Parameters | Accuracy |
|---------|---------:|---------:|
| CNN V5 | 112,143 | 93.41% |
| ResNet18 | 11,184,207 | 95.32% |

### Improvement

```text
Accuracy Gain:
95.32% - 93.41%
= +1.91%
```

---

## Observations

- ResNet18 converged significantly faster than the custom CNN.
- The pretrained ImageNet features provided a strong starting point.
- The model achieved the highest overall accuracy among all experiments.
- Several classes achieved near-perfect accuracy.
- Class 6 remained challenging, suggesting dataset complexity rather than architectural limitations.
- Although ResNet18 improved accuracy, it required approximately 100× more parameters than CNN V5.

---

## Saved Model

```text
models/resnet18_95_32.pth
```

---

## Conclusion

Transfer learning using ResNet18 achieved the best overall performance in the project.

The experiment demonstrated the effectiveness of pretrained feature extractors for image classification tasks and established a strong benchmark for future experiments using EfficientNet and other modern architectures.

### Final Result

🏆 **ResNet18 Transfer Learning**

```text
Accuracy: 95.32%
Parameters: 11,184,207
```
# Experiment 13: EfficientNet-B0 Transfer Learning 🏆Best Model

## Objective

The goal of this experiment was to evaluate a modern transfer learning architecture and compare its performance against:

- Custom CNN architectures (CNN V1–V5)
- ResNet18 Transfer Learning

EfficientNet-B0 was selected because it provides an excellent balance between accuracy and parameter efficiency.

---

## Why EfficientNet?

EfficientNet introduces compound scaling, which balances:

- Network Depth
- Network Width
- Input Resolution

Unlike traditional CNN scaling approaches, EfficientNet scales all three dimensions simultaneously, resulting in higher accuracy with fewer parameters.

The model was initialized using pretrained ImageNet weights and fine-tuned for crop disease classification.

---

## Dataset

- Total Images: 20,638
- Classes: 15
- Training Images: 16,510
- Testing Images: 4,128

---

## Data Preprocessing

### Training Transform

```python
transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
```

### Test Transform

```python
transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
```

---

## Proper Dataset Split

A clean train/test split was created before applying transforms.

### Verification

```text
Training Images: 16510
Testing Images : 4128
Overlap        : 0
```

This ensured that no image appeared in both the training and testing sets.

---

## Model Architecture

### Original EfficientNet-B0

```text
EfficientNet-B0
      ↓
Dropout(0.2)
      ↓
Linear(1280 → 1000)
```

### Modified EfficientNet-B0

```text
EfficientNet-B0 Feature Extractor
                ↓
Dropout(0.2)
                ↓
Linear(1280 → 15)
                ↓
Disease Prediction
```

---

## Parameter Count

```text
4,026,763 Parameters
```

### Comparison

| Model | Parameters |
|---------|---------:|
| CNN V5 | 112,143 |
| EfficientNet-B0 | 4,026,763 |
| ResNet18 | 11,184,207 |

---

## Training Configuration

### Optimizer

```python
Adam(lr=0.001)
```

### Loss Function

```python
CrossEntropyLoss()
```

### Batch Size

```text
32
```

### Epochs

```text
10
```

---

## Training Loss

| Epoch | Average Loss |
|---------|---------:|
| 1 | 0.2913 |
| 2 | 0.1035 |
| 3 | 0.0819 |
| 4 | 0.0637 |
| 5 | 0.0572 |
| 6 | 0.0538 |
| 7 | 0.0522 |
| 8 | 0.0462 |
| 9 | 0.0439 |
| 10 | 0.0382 |

---

## Results

### Training Accuracy

```text
99.68%
```

### Test Accuracy

```text
99.39%
```

### Generalization Gap

```text
99.68% - 99.39%
= 0.29%
```

The extremely small gap indicates strong generalization and no significant evidence of overfitting.

---

## Class-wise Accuracy

| Class | Accuracy |
|---------|---------:|
| 0 | 99.46% |
| 1 | 99.64% |
| 2 | 100.00% |
| 3 | 98.47% |
| 4 | 100.00% |
| 5 | 100.00% |
| 6 | 98.21% |
| 7 | 99.72% |
| 8 | 98.51% |
| 9 | 99.72% |
| 10 | 98.83% |
| 11 | 99.33% |
| 12 | 99.20% |
| 13 | 100.00% |
| 14 | 100.00% |

---

## Comparison with Previous Best Models

| Model | Parameters | Accuracy |
|---------|---------:|---------:|
| CNN V5 | 112,143 | 93.41% |
| ResNet18 | 11,184,207 | 95.32% |
| EfficientNet-B0 | 4,026,763 | **99.39%** |

### Improvement Over ResNet18

```text
99.39% - 95.32%
= +4.07%
```

### Improvement Over CNN V5

```text
99.39% - 93.41%
= +5.98%
```

---

## Observations

- EfficientNet-B0 achieved the highest accuracy of all experiments.
- The model significantly outperformed ResNet18 despite using fewer parameters.
- Class 6, previously one of the most difficult classes, improved from approximately 74–78% accuracy to over 98%.
- Multiple classes achieved perfect classification accuracy.
- The training and testing accuracies remained extremely close, indicating strong generalization.
- EfficientNet demonstrated the effectiveness of modern transfer learning architectures for plant disease classification.

---

## Saved Model

```text
models/efficientnet_b0_99_39.pth
```

---

## Conclusion

EfficientNet-B0 became the best-performing model in the project.

It achieved:

- Highest Accuracy: **99.39%**
- Strong Generalization
- Fewer Parameters than ResNet18
- Near-perfect class-wise performance

This experiment established EfficientNet-B0 as the final champion architecture for the Crop Disease Detector project.

# Final Experiment Leaderboard

| Rank | Model | Parameters | Accuracy |
|---|---|---:|---:|
| 🥇 | EfficientNet-B0 Transfer Learning | 4,026,763 | **99.39%** |
| 🥈 | ResNet18 Transfer Learning | 11,184,207 | 95.32% |
| 🥉 | CNN V5 (GAP + FC(128→128) + ReLU + FC(128→15)) | 112,143 | 93.41% |
| 4 | BatchNorm + Dropout(0.5) + Normalization | 1,821,711 | 92.68% |
| 5 | BatchNorm + Dropout(0.3) + Normalization | 1,821,711 | 92.66% |
| 6 | Augmentation + Normalization | 1,821,263 | 92.42% |
| 7 | Augmentation (10 Epochs) | 1,821,263 | 91.86% |
| 8 | CNN V3 (GAP → FC(128→15)) | 95,631 | 91.81% |
| 9 | CNN V4 (GAP → FC(128→64) → ReLU → FC(64→15)) | 102,927 | 90.99% |
| 10 | BatchNorm Only | 1,821,711 | 90.87% |
| 11 | Baseline CNN | 1,821,263 | 89.78% |
| 12 | CNN V4 + Dropout | 102,927 | 88.47% |
---

## Best Accuracy

🏆 **EfficientNet-B0 Transfer Learning**

- Accuracy: **99.39%**
- Parameters: **4,026,763**
- Train Accuracy: **99.68%**
- Test Accuracy: **99.39%**
- Generalization Gap: **0.29%**

## Most Efficient Model

⚡ **CNN V5**

- Accuracy: **93.41%**
- Parameters: **112,143**
- ~36× fewer parameters than EfficientNet-B0
- ~100× fewer parameters than ResNet18
## Key Takeaway

Three strong models emerged from the project:

### EfficientNet-B0
- Highest overall accuracy
- Strong transfer learning performance
- Achieved 99.39% accuracy
- Smaller than ResNet18 while performing better

### ResNet18
- Strong transfer learning baseline
- Achieved 95.32% accuracy
- Demonstrated the effectiveness of pretrained ImageNet features

### CNN V5
- Lightweight custom architecture
- Excellent parameter efficiency
- Achieved 93.41% accuracy with only 112K parameters

The project successfully explored custom CNN optimization and modern transfer learning architectures for crop disease classification.

### Final Conclusion

- Data augmentation initially reduced accuracy when trained for only 5 epochs.
- Increasing the training duration to 10 epochs allowed the model to better learn from augmented images and improved the test accuracy to **91.86%**.
- Adding input normalization stabilized training by centering pixel values around zero, increasing the accuracy to **92.42%**.
- Adding Batch Normalization alone did not improve performance and reduced the accuracy to **90.87%**.
- Combining Batch Normalization with Dropout improved generalization, with **Dropout(0.5)** achieving **92.68%** accuracy and outperforming **Dropout(0.3)**.
- Introducing Global Average Pooling (GAP) drastically reduced the model size by removing the large fully connected classifier. The pure GAP model achieved **91.81%** accuracy with only **95,631 parameters**.
- Adding a small hidden classifier with excessive compression (**128 → 64**) reduced performance, showing that too much feature reduction can hurt classification accuracy.
- Removing Dropout from the small GAP classifier improved accuracy from **88.47% to 90.99%**, indicating that strong regularization was unnecessary for lightweight models.
- The final GAP architecture (**GAP → FC(128→128) → ReLU → FC(128→15)**) achieved the best balance between model capacity and efficiency, reaching **93.41%** accuracy with only **112,143 parameters**.
- Compared to the previous best CNN (**1.82 million parameters, 92.68% accuracy**), CNN V5 improved accuracy by **0.73%** while reducing the number of parameters by approximately **93.8%**.
- Transfer learning using **ResNet18 pretrained on ImageNet** significantly improved performance and achieved **95.32%** test accuracy.
- ResNet18 outperformed CNN V5 by **1.91%**, demonstrating the power of pretrained feature extraction.
- EfficientNet-B0 Transfer Learning achieved the highest overall performance in the project.
- EfficientNet-B0 reached **99.39%** test accuracy while using only **4.03 million parameters**.
- The model outperformed ResNet18 by **4.07%** while using approximately **64% fewer parameters**.
- Training accuracy (**99.68%**) and test accuracy (**99.39%**) remained extremely close, resulting in a generalization gap of only **0.29%**.
- Several disease classes achieved **100% classification accuracy**, and previously difficult classes improved dramatically.
- The project demonstrated three successful approaches:
  - **EfficientNet-B0** for maximum accuracy (**99.39%**)
  - **ResNet18** as a strong transfer learning baseline (**95.32%**)
  - **CNN V5** for maximum efficiency (**93.41% with only 112,143 parameters**)
- The final best-performing model was saved as:

```text
models/efficientnet_b0_99_39.pth
```

The project successfully explored custom CNN optimization, architecture design, and transfer learning techniques for crop disease classification.

Among all experiments, EfficientNet-B0 achieved the best overall performance, demonstrating how modern pretrained architectures can significantly outperform handcrafted CNNs while maintaining strong parameter efficiency.
```

### Final Results

| Model | Parameters | Accuracy |
|---------|---------:|---------:|
| EfficientNet-B0 Transfer Learning | 4,026,763 | **99.39%** |
| ResNet18 Transfer Learning | 11,184,207 | 95.32% |
| CNN V5 | 112,143 | 93.41% |

