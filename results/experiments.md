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

# Experiment 11: CNN V5 (GAP + FC(128→128) + ReLU + FC(128→15)) ⭐ Best Model

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

**93.41% ⭐**

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


# Final Experiment Leaderboard

| Rank | Model | Parameters | Accuracy |
|---|---|---:|---:|
| 🥇 | CNN V5 (GAP + FC(128→128) + ReLU + FC(128→15)) | 112,143 | **93.41%** |
| 🥈 | BatchNorm + Dropout(0.5) + Normalization | 1,821,711 | 92.68% |
| 🥉 | BatchNorm + Dropout(0.3) + Normalization | 1,821,711 | 92.66% |
| 4 | Augmentation + Normalization | 1,821,263 | 92.42% |
| 5 | Augmentation (10 Epochs) | 1,821,263 | 91.86% |
| 6 | CNN V3 (GAP → FC(128→15)) | 95,631 | 91.81% |
| 7 | CNN V4 (GAP → FC(128→64) → ReLU → FC(64→15)) | 102,927 | 90.99% |
| 8 | BatchNorm Only | 1,821,711 | 90.87% |
| 9 | Baseline CNN | 1,821,263 | 89.78% |
| 10 | CNN V4 + Dropout (GAP → FC(128→64) → ReLU → Dropout → FC(64→15)) | 102,927 | 88.47% |


---

### Final Conclusion

- Data augmentation initially reduced accuracy when trained for only 5 epochs.
- Increasing the training duration to 10 epochs allowed the model to better learn from augmented images and improved the test accuracy to **91.86%**.
- Adding input normalization stabilized training by centering pixel values around zero, increasing the accuracy to **92.42%**.
- Adding Batch Normalization alone did not improve performance and reduced the accuracy to **90.87%**.
- Combining Batch Normalization with Dropout improved generalization, with **Dropout(0.5)** achieving **92.68%** accuracy and outperforming **Dropout(0.3)**.
- Introducing Global Average Pooling (GAP) drastically reduced the model size by removing the large fully connected classifier. The pure GAP model achieved **91.81%** accuracy with only **95,631 parameters**.
- Adding a small hidden classifier with excessive compression (**128 → 64**) reduced performance, showing that too much feature reduction can hurt classification accuracy.
- Removing Dropout from the small GAP classifier improved accuracy from **88.47% to 90.99%**, indicating that strong regularization was unnecessary for lightweight models.
- The final GAP architecture (**GAP → FC(128→128) → ReLU → FC(128→15)**) achieved the best balance between model capacity and efficiency, reaching the highest test accuracy of **93.41%** with only **112,143 parameters**.
- Compared to the previous best CNN (**1.82 million parameters, 92.68% accuracy**), the final model improved accuracy by **0.73%** while reducing the number of parameters by approximately **93.8%**.
- The final best model was saved as `models/cnn_v5_gap_fc128_93_41.pth`.

