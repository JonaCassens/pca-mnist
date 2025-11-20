# Principal Component Analysis (PCA) and MNIST Classification

This repository contains **PCA-MNIST.ipynb**, which explores the application of Principal Component Analysis (PCA) for dimensionality reduction within a machine learning context.

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



