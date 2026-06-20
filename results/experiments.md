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

# Experiment 3: CNN with Data Augmentation (10 Epochs) ⭐ Best Model


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


### Final Conclusion

- Data augmentation initially reduced accuracy when trained for only 5 epochs.
- Increasing training time to 10 epochs allowed the model to learn from the augmented images.
- The final augmented model achieved better generalization and improved the test accuracy from **89.78% to 91.86%**.
- The best model was saved as `models/best_91_86.pth`.
