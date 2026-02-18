# Deep Learning for Forest Cover Type Classification

A deep neural network trained to classify seven types of forest cover from cartographic features, achieving 94.07% test accuracy on a dataset of 581,012 samples. Built as part of advanced studies in Mathematical Statistics and Machine Learning at Stockholm University.

## Problem

The UCI Forest Covertype dataset presents two compounding challenges: scale (581,012 samples, 54 features) and severe class imbalance — the two dominant cover types account for over 85% of observations, while the rarest class represents less than 4%. A naive model simply predicts the majority class and looks deceptively accurate.

The goal was to build a model that performs well *across all seven classes* — including the rare ones.

## Approach

**Preprocessing**
- 80/10/10 train/validation/test split applied *before* standardisation to prevent data leakage
- Z-score standardisation fitted exclusively on training data, then applied to validation and test sets
- Random oversampling of minority classes to full parity with the majority class, following Buda et al. (2018) — who demonstrate that oversampling is the dominant strategy for addressing class imbalance in deep learning

**Architecture**
A 12-layer barrel-shaped MLP implemented in Keras:
- Starts at 54 input features, expands via a doubling protocol (54 → 80 → 160 → 320 → 640 → 1280) to capture complex non-linear representations
- Compresses symmetrically back down (1280 → 640 → 320 → 160 → 80 → 40 → 20) before the output layer
- ReLU activations throughout hidden layers; Softmax output for 7-class probability distribution
- Adam optimizer (lr=1e-4, momentum=0.9); Sparse Categorical Crossentropy loss

**Training**
- 500 epochs, batch size 512
- Batch size was a critical finding: sizes of 16 and 32 caused severe gradient variance and loss spikes on this large dataset. Batch 512 acted as a natural noise filter, producing stable, smooth convergence
- No significant overfitting observed — validation loss tracks training loss throughout

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **94.07%** |
| Class 7 Accuracy (rarest class, <4% of data) | **87%** |

The confusion matrix confirms strong performance across all seven classes — including the minority classes that represent the real challenge in this dataset.

## Key Findings

- Oversampling combined with deep architecture effectively neutralises class imbalance without causing overfitting on duplicated samples
- Batch size matters more than learning rate tuning at this data scale — instability from small batches cannot be resolved by momentum or rate adjustments alone
- The barrel architecture (expand then compress) outperformed a shallower 4-layer baseline with a noticeable reduction in generalisation gap

## Stack

Python · Keras (TensorFlow) · NumPy · Pandas · Scikit-learn · Matplotlib · Seaborn · ucimlrepo

## Data

UCI Forest Covertype dataset — 581,012 samples, 54 cartographic features, 7 cover type classes.
Source: `ucimlrepo` (id=31) / [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/31/covertype)

## References

- Buda, M., Maki, A., & Mazurowski, M. A. (2018). *A systematic study of the class imbalance problem in convolutional neural networks.* Neural Networks, 106, 249–259.
- Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.

---

*Part of advanced studies in Mathematical Statistics and Machine Learning, Stockholm University.*
