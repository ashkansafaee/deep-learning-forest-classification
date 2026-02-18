# %% [markdown]
# ### Executive summary 
# This project presents a deep learning approach to classifying forest types using the UCI Forest Covertype dataset. The primary challenge of significant class imbalance was addressed through systematic preprocessing and architectural framework. 
# 
# * **Handling class imbalance**: Following the "oversampling" recommendation from Buda et al. (2018), minority classes were resampled until they matched the majority class. 
# * **Data leakage**: To prevent 'data leakage', the dataset was partitioned into 80/10/10 before applying standardization derived exclusively from the training set. 
# * **Neural network architecture**: A 12-layer deep neural network was implemented using a "doubling step size"-protocol, starting from the 7 output dimensions, and doubling neuron counts per layers to capture complex non-linear properties, while ensuring that compression phase was not too steep, and decreasing with the same protocol until the final output layer. 
# * **Training configuration**: The model utilized ReLU-activiations for hidden layers, and Softmax output layer for multinomial prediction, and the Adam optimizer for efficient gradient descent with momentum at 0.9 and learning rate at $1 \times 10^{-4}$. 
# * **Key results**: The model achieved a test accuracy of 94.07%. Notably, the model demonstrated a high precision for rare-classes, such as 87% for Class 7, which represent less than 4% of the original data. 
# * **Conclusion**: The stability of the learning curves and high performance on minority classes validate the deep architecture combined with oversampling mitigated the detrimental effects of class imbalance without causing significant overfitting. 

# %%
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# fetch dataset 
covertype = fetch_ucirepo(id=31)

# data (as pandas dataframes) 
X = covertype.data.features 
y = covertype.data.targets 

# metadata 
# print(covertype.metadata) 
# variable information 
# print(covertype.variables)

df = pd.concat([X ,y], axis=1)

print("--- Class Counts ---")
print(df['Cover_Type'].value_counts())
print("\n--- Class Proportions ---")
print(df['Cover_Type'].value_counts(normalize=True))

plt.figure(figsize=(10, 6)) 
sns.countplot(x='Cover_Type', data=df)
plt.title('Distribution of Forest Cover Types')
plt.show() 


# %% [markdown]
# The Forest Covertype dataset presents a significant challenge for neural network classification due to its inherent class imbalance. To ensure robust model generalization across all seven cover types, this study adopts the preprocessing framework proposed by [A systematic study of the class imbalance problem in convolutional neural networks](#ref-buda).
# 
# Based on a systematic analysis of deep learning architectures, several key conclusions from recent literature guide our data preparation:
# 
# * **Impact of Imbalance**: The detrimental effect of class imbalance on classification performance is significant, necessitating intervention prior to model training.
# * **Method selection**: Among various techniques, oversampling emerges as the dominant method for addressing class imbalance in almost all analyzed deep learning scenarios. 
# * **Target distribution** : Following the recommendations to completely eliminate imbalance, the minority classes were oversampled until they reached parity with the majority class distribution. 
# * **Overfitting resistance** : Unlike classical machine learning models, deep neural networks (such as MLP) do not exhubut significant overfitting when trained on duplicated samples genereated through oversampling. 

# %%
# Split data into training and test data

# Convert the DataFrame to a Numpy array
X_np = X.to_numpy()
y_np = y.to_numpy()

# Randomly permute the array
np.random.seed(222)
indices = np.random.permutation(len(X_np))

# Define 80/10/10 split
train_split_limit = int(0.8 * len(X_np))
test_split_limit = int(0.9 * len(X_np))

train_idx = indices[ :train_split_limit]
val_idx = indices[train_split_limit:test_split_limit]
test_idx = indices[test_split_limit: ]

X_train_raw, y_train = X_np[train_idx], y_np[train_idx]
X_val_raw, y_val = X_np[val_idx], y_np[val_idx]
X_test_raw, y_test = X_np[test_idx], y_np[test_idx]

# %% [markdown]
# According to the litterature and best practices in the machine learning community, we are recommended to split into training and test before standarizing. This is to prevent **Data Leakage**, a phenomenon where information from the evaluation sets 'leaks' into the training process, often leading to overly optimistic performance. 
# 
# "Preprocessing steps such as scaling, imputation or feature selection should be fitted only on the training data and then applied to the validation set, rather than fitting them on the entire dataset before spitting". ([What is data leakage in machine learning?](#ref-ibm))

# %%
# Calculating the mean and standard deviation for each column
train_mean = np.mean(X_train_raw, axis=0)
train_std = np.std(X_train_raw, axis=0)

# Standardizing X and adding a small epsilon to avoid division by zero if std is zero. 
X_train = (X_train_raw - train_mean) / (train_std + 1e-12)
X_val = (X_val_raw - train_mean) / (train_std + 1e-12)
X_test = (X_test_raw - train_mean) / (train_std + 1e-12)

print("\nStandardization Complete.")

# %% [markdown]
# Standardization was applied to normalize input variances. Each feature was transformed using z-score standardization formula 
# 
# $$z = \frac{x-\mu}{\sigma + \varepsilon}.$$ 

# %%
# Implement oversamling method for rare classes to match majority
unique, counts = np.unique(y_train, return_counts=True)
max_samples = np.max(counts)
X_bal, y_bal = [], []

for cls in unique:
    train_indices = np.where(y_train == cls)[0]
    
    # Oversampling
    resample_idx = np.random.choice(train_indices, size = max_samples, replace= True)
    
    X_bal.append(X_train[resample_idx])
    y_bal.append(y_train[resample_idx])

X_final = np.vstack(X_bal) # Stack arrays in sequence row wise
y_final = np.concatenate(y_bal) # Join a sequence of arrays along an existing axis

print("Pre-processing complete. Ready for Neural Network.")

# %%
import keras
from keras import layers

# Adjust labels for 0-indexing
y_train_shifted = y_final - 1
y_val_shifted = y_val - 1

model = keras.Sequential(
    [
        # Following a (maximum) double step size as suggested in class by professor 
        layers.Dense(12, activation="relu", input_shape=(X_final.shape[1],)),

        layers.Dense(20, activation="relu"),

        layers.Dense(38, activation="relu"),

        layers.Dense(70, activation="relu"),

        layers.Dense(120, activation="relu"),

        layers.Dense(200, activation="relu"),

        layers.Dense(350, activation="relu"),

        layers.Dense(200, activation="relu"),

        layers.Dense(120, activation="relu"),

        layers.Dense(70, activation="relu"),
    
        layers.Dense(38, activation="relu"),
    
        layers.Dense(20, activation="relu"),

        # Output: 7 tree types
        layers.Dense(7, activation="softmax"), 
    ]
)

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 1e-4, 
                                      ema_momentum= 0.9),

    # Use sparse categorical crossentropy for integer labels
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"],
)

history = model.fit(
    X_final, 
    y_train_shifted,
    epochs = 500,
    batch_size = 512,
    validation_data = (X_val, y_val_shifted)
)

# %% [markdown]
# The neural network was initialized with a 12-layer architecture utilizing a "double step-size" scaling protocol. Starting from the 7 output dimensions (corresponding to the number of cover types), the number of neurons was incrementally scaled upwards by approximately doubling the layer size at each step, as recommended by Chun.
# 
# Utilizing Rectified Linear Units (ReLU) for the hidden layers to facilitate efficient gradient propagation. For the output layer, a Softmax activation function was implemented. While Sigmoid is suitable for binary tasks, Softmax is the established standard for multinomial class prediction (7 cover types) as it produces a probability distribution across mutually exclusive classes.
# 
# The model utilizes the Adam optimizer, an adaptive learning rate method that builds upon stochastic gradient descent. Adam is particularly effective for datasets with a large number of observations, such as the 581,012 samples in the Covertype dataset, and is regarded in literature as a robust default choice for complex classification problems ([Deep Learning with Pyton](#ref-chollet)). 
# 
# For the objective function, Sparse Categorical Crossentropy was selected. According to [Keras documentation](#ref-keras), this is the recommended loss function when dealing with two or more labels provided as integers, as it eliminates the need for manual one-hot encoding while effectively penalizing misclassifications in a multi-class context.

# %%
# model.fit returns a History object
# This object has a member history, which is a dictionary 
# containing data about everything that happened during training
history_dict = history.history
history_dict.keys()

# %%
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

df_history = pd.DataFrame(history.history) # Create a DataFrame from history
df_history['epoch'] = range(1, len(df_history) + 1)

# Melt plot to fit seaborns long-format
df_plot = df_history.melt(id_vars = 'epoch', 
                          value_vars = ['loss', 'val_loss'], 
                          var_name = 'Dataset', 
                          value_name = 'Loss')

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.lineplot(data=df_plot, x='epoch', y='Loss', hue='Dataset', style='Dataset', markers=True)
plt.title("[Cover Type] Training and Validation Loss", fontsize=15)
plt.xlabel("Epochs")
plt.ylabel("Loss")

# %% [markdown]
# Initially, a shallower network architecture with four layers was implemented. While the loss curves indicated successful learning, a clear generalization gap was observed. To adress this, the network's depth and capacity were increased in accordance with [Buda et al. (2018)](#ref-buda) whose findings suggest that oversampling is the dominant strategy for addressing class imbalance and that deeper architectures help mitigate its detrimental effects. 
# 
# Following these principles and recommendations regarding a "double step-size" protocol, the network was extended to a 12-layer barrel-shaped architecture. This structure allowed the 54 features to be expanded into higher-dimensional representations before converging on the 7 cover type classes .
# 
# **From Batch 16 to 512: Solving numerical stability**
# The most critical turning point in the training process was the transition in batch size.
# 
# **The Failure of Small Batches**: Attempting to use a batch size of 16, or even 32, while theoretically beneficial for generalization, led to extreme gradient variance. Because the dataset is so large ($N \approx 581,000$), a batch of 16 was not representative enough, causing the "loss spikes". Even with added momentum and varied learning rates, the training remained unstable.
# 
# **The Stability of Batch 512**: Increasing the batch size to 512, while keeping momentum at 0.9 and maintaining learning rate at 1e-4 acted as a natural filter for this noise. This allowed the Adam optimizer to calculate more reliable updates, resulting in the smooth, stable convergence seen in the final 500-epoch learning curve.
# 
# The resulting learning curves behave as expected: validation loss remains slightly above training loss, and both demonstrate steady convergence. No significant overfitting is observed, as the validation curve does not exhibit an upward trend. While early stopping around 300 epochs would have been sufficient, the extended training allowed the model to improve without compromising generalization, ultimately reaching a test accuracy of 94.07%.

# %%
# Adjust labels for 0-indexing
y_test_shifted = y_test - 1 
eval_results = model.evaluate(X_test, y_test_shifted, verbose=0)

print(f"Test Loss: {eval_results[0]:.4f}")
print(f"Test Accuracy: {eval_results[1]:.4f}")

# %% [markdown]
# When evaluating the model on unseen test data, the classification accuracy reached 94.07 %. These results confirm that the increased depth and capacity of the network were justified by the high performance and that the oversampling strategy effectively addressed the original dataset's skew.
# 
# The high accuracy on the test set demonstrates that the model successfully learned to generalize well rather than merely memorizing training samples. This further supports the conclusion that the "detrimental" effects of class imbalance can be mitigated through systematic oversampling and the use of deep architectures. 

# %%
from sklearn.metrics import confusion_matrix

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis = 1) # Choose class with highest probability

cm = confusion_matrix(y_test_shifted, y_pred, normalize='pred')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix (Normaliserad)')
plt.show()

# %% [markdown]
# The confusion matrix confirms that the model performs well on the two majority classes (Class 1 and Class 2), as expected. More importantly, it demonstrates strong performance across the minority classes (3 through 7), which is a direct result of our oversampling strategy. Despite Class 3 representing less than 6.1% of the total observations, the model achieved reliable classification results. Most notably, the model reached an 87% accuracy on Class 7, even though this class constitutes less than 4% of the original dataset.
# 
# These results highlight the effectivness of the chosen methodology proving that the model can successfully distinguish rare classes without being overshadowed by dominant ones.  

# %% [markdown]
# ### References
# * <a name="ref-buda"></a> **Buda, M., Maki, A., & Mazurowski, M. A. (2018).** *A systematic study of the class imbalance problem in convolutional neural networks.* Neural Networks, 106, 249-259. Available via: [arXiv:1710.05381](https://arxiv.org/pdf/1710.05381)
# * <a name="ref-keras"></a> **Keras API documentation**. Available via: [keras.io](https://keras.io/api/losses/probabilistic_losses/#sparsecategoricalcrossentropy-class)
# * <a name="ref-ibm"></a> **IBM Research (2023)** : What is data leakage in machine learning? IBM Think Topics. Available via: [ibm.com](https://www.ibm.com/think/topics/data-leakage-machine-learning)
# * <a name="ref-chollet"></a> **Chollet, F. (2021)**. Deep Learning with Python (2nd ed.). Manning Publications. Online-version available via: [deeplearningwithpython.io](https://deeplearningwithpython.io)
# 
# ### Acknowledgements
# The development of this project was supported by a combination of academic research, community expertise, and iterative technical discussions.
# 
# * **Conceptual Strategy**: The implementation of the random oversampling strategy was developed through a combination of theoretical research and practical experimentation. The conceptual framework was refined through iterative discussions with an AI-collaborator, which helped align the technical execution with the findings of Buda et al. (2018).
# * **Foundational Coding**: The architechtural design and implementation of the Keras model followed best practices and principles outlined in [Deep Learning with Python by François Chollet](https://deeplearningwithpython.io), ensuring a robust and standardized approach of deep learning.
# * **Oversampling Strategy**: The approach to balancing classes was informed by the methodologies described in "[How to handle class imbalance](https://towardsdatascience.com/how-to-handle-imbalanced-datasets-in-machine-learning-projects-a95fa2cd491a/)". The NumPy-based resampling logic utilizing np.random.choice and np.vstack was guided by community-verified implementations found on [Stack Overflow](https://stackoverflow.com/questions/23391608/balance-numpy-array-with-over-sampling).
# 
# ### Technical Remarks: 
# Potential enhancements, such as tracking the learning rate and graphing training accuracy, were excluded due to the high computational cost and strict project timeline. 


