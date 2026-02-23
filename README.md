# Deep Learning for Classification

Two end-to-end deep learning projects tackling different data modalities, architectures, and core challenges — built as part of advanced studies in Mathematical Statistics and Machine Learning at Stockholm University.

| Project | Data | Architecture | Challenge | Result |
|---------|------|-------------|-----------|--------|
| Forest Cover Classification | Tabular (581K samples) | 12-layer MLP | Class imbalance | 94.07% accuracy |
| CIFAR-10 Image Classification | Images (60K samples) | CNN | Overfitting | 81% accuracy |

---

## Project 1 — Forest Cover Type Classification (MLP)

### Problem
Classify seven types of forest cover from 54 cartographic features across 581,012 samples. The core challenge: two dominant classes account for over 85% of observations, making naive models deceptively accurate while failing on rare classes.

### Approach
- 80/10/10 train/validation/test split applied before standardisation to prevent data leakage
- Z-score standardisation fitted on training data only
- Random oversampling of minority classes to full parity with the majority class (Buda et al., 2018)
- 12-layer barrel-shaped MLP: expands from 54 inputs via doubling protocol (54→80→160→320→640→1280) then compresses symmetrically to 7 outputs
- ReLU activations, Softmax output, Adam optimizer (lr=1e-4, momentum=0.9)
- Critical finding: batch size 512 was essential for stable convergence — sizes of 16 and 32 caused severe gradient variance on this scale of data

### Results
- **Test accuracy: 94.07%**
- **Class 7 accuracy: 87%** (representing <4% of original data)
- Stable learning curves with no significant overfitting

---

## Project 2 — CIFAR-10 Image Classification (CNN)

### Problem
Classify 60,000 colour images (32×32 pixels) across 10 classes. The core challenge here is the opposite of Project 1: the dataset is balanced, but small images and a relatively shallow network make overfitting the primary enemy.

### Approach
A systematic progression of architectures, each addressing the limitations of the previous:

**Baseline CNN (ReLU vs tanh comparison)**
- 3-layer Conv2D architecture with MaxPooling
- ReLU achieved ~68% accuracy; tanh achieved ~63%
- tanh underperformed due to vanishing gradients and computational overhead — gradients approach zero for large inputs, halting learning in deeper layers

**Adding Batch Normalisation**
- BN sandwiched between Conv2D (with `use_bias=False`) and activation, following Chollet (2021)
- Stabilised training and improved accuracy to ~69%
- However, validation loss failed to converge — clear overfitting signal

**Adding Data Augmentation**
- Random horizontal flips and rotations applied to training images
- Expanded effective dataset size without collecting new data
- Substantially closed the train/validation gap

**Full Pipeline (BN + Augmentation + Dropout)**
- Combining all three regularisation strategies pushed accuracy to **~81%**
- Clean, converging learning curves with no overfitting

### Key Finding
Batch normalisation alone stabilises optimisation but does not prevent overfitting. Data augmentation addresses the root cause. Dropout adds a final layer of regularisation. Each technique solves a distinct problem — the combination is what matters.

---

## Stack

Python · TensorFlow · Keras · NumPy · Pandas · Matplotlib · Seaborn · Scikit-learn · ucimlrepo

## References

- Buda, M., Maki, A., & Mazurowski, M. A. (2018). *A systematic study of the class imbalance problem in convolutional neural networks.* Neural Networks, 106, 249–259.
- Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
- Zhang, A. et al. (2023). *Dive into Deep Learning.* Available at d2l.ai
- Ioffe, S. & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training.*
- Santurkar, S. et al. (2018). *How Does Batch Normalization Help Optimization?*

---

*Part of advanced studies in Mathematical Statistics and Machine Learning, Stockholm University.*
