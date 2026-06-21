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

# Experiment 5: CNN with BatchNorm + Dropout(0.5) + Normalization (10 Epochs) ⭐ Best Model

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

# Final Experiment Leaderboard

| Rank | Model | Accuracy |
|---|---|---|
| 🥇 | BatchNorm + Dropout(0.5) + Normalization | **92.68%** |
| 🥈 | BatchNorm + Dropout(0.3) + Normalization | **92.66%** |
| 🥉 | Augmentation + Normalization | **92.42%** |
| 4 | Augmentation (10 Epochs) | 91.86% |
| 5 | BatchNorm Only | 90.87% |
| 6 | Baseline CNN | 89.78% |

---

### Final Conclusion

- Data augmentation initially reduced accuracy when trained for only 5 epochs.
- Increasing the training duration to 10 epochs allowed the model to better learn from augmented images and improved the test accuracy to **91.86%**.
- Adding input normalization further stabilized training by centering pixel values around zero, increasing the accuracy to **92.42%**.
- Adding Batch Normalization alone did not improve the model and reduced the accuracy to **90.87%**.
- Combining Batch Normalization with Dropout significantly improved the model's generalization ability.
- Dropout tuning showed that **Dropout(0.5)** performed slightly better than **Dropout(0.3)**, achieving the highest test accuracy of **92.68%**.
- The best model achieved strong performance across most disease classes, with major improvements in difficult classes such as **Class 6**, which reached **79.79%** accuracy.
- The final best model was saved as `models/batchnorm_dropout_0_5_92_68.pth`.
