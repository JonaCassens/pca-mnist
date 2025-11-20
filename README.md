# Coursework C: Principal Component Analysis (PCA) and MNIST Classification

This repository contains **CourseworkC.ipynb**, which explores the application of Principal Component Analysis (PCA) for dimensionality reduction within a machine learning context.

## Overview

The primary objective of this notebook is to demonstrate and evaluate the effectiveness of PCA when applied to the standard **MNIST handwritten digit dataset**. The project involves:

1.  **Neural Network Baseline:** Training a simple Multi-Layer Perceptron (MLP) on the full 28x28 pixel dataset to establish a baseline accuracy and training time.
2.  **PCA Implementation:** Applying the full mathematical process of PCA (covariance matrix calculation, eigendecomposition, and data projection) to the dataset.
3.  **Dimensionality Reduction:** Transforming the 784-dimensional image vectors into a reduced feature set (e.g., 100 or 20 principal components).
4.  **Performance Analysis:** Training the same MLP model on the reduced datasets to compare training time, reconstruction quality, and final classification accuracy against the baseline.

## Key Findings

The notebook demonstrates a clear trade-off: significant dimensionality reduction leads to faster training times, but retaining too few components (e.g., 20) results in a noticeable drop in classification accuracy due to information loss.

## Requirements

The notebook requires the following standard Python libraries:
* `numpy`
* `matplotlib`
* `scipy` (for linear algebra)
* `tensorflow` / `keras`
* `time`

---

## 2. `CourseworkD_21019291.ipynb` (Double Pendulum Prediction)

### Description (for file summary/commit)
Coursework D for "Practical Machine Learning for Physicists." This notebook models the chaotic dynamics of a double pendulum using Lagrangian mechanics and applies a Recurrent Neural Network (RNN/LSTM) to predict its future state. A core challenge is predicting the full system state based on incomplete input data (position of only the lower mass).

```markdown
# Coursework D: Predicting Chaotic Systems with Machine Learning (Double Pendulum)

This repository contains **CourseworkD_21019291.ipynb**, which explores the use of machine learning to model and predict the behaviour of a chaotic dynamical system.

## Overview

The project focuses on the classic physics problem of the **double pendulum**. The notebook is structured to:

1.  **Physics Simulation:** Use Lagrangian mechanics to derive the equations of motion (EOMs) for the double pendulum and simulate its chaotic, non-linear dynamics numerically (e.g., using `solve_ivp`).
2.  **Data Generation:** Generate high-fidelity time-series data for the pendulum's state variables (angles and angular velocities).
3.  **Machine Learning Task:** Train a **Recurrent Neural Network (RNN)**, specifically an **LSTM**, to predict the future state of the system.
4.  **Incomplete Information Challenge:** Critically, the model is trained to predict the full system state (four variables: $\theta_1$, $\dot{\theta}_1$, $\theta_2$, $\dot{\theta}_2$) using only *partial* input data, such as the position of the lower mass.

## Goal

The aim is to assess the efficacy and robustness of sequence prediction models (LSTMs) in forecasting the complex, chaotic evolution of a physics system, especially when facing the challenge of incomplete or noisy observational data.

## Requirements

The notebook requires the following standard Python libraries:
* `numpy`
* `matplotlib`
* `scipy.integrate` (for `solve_ivp`)
* `tensorflow` / `keras` (for building and training the LSTM model)
