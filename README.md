# 🧠 Neural Networks & Deep Learning

A collection of three assignments implementing various machine learning algorithms, deep learning models, and autoencoders, developed for the course *Neural Networks - Deep Learning* (2025-2026) at the Aristotle University of Thessaloniki, Computer Science Department.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Exercise 1 — Image Classification (CNN, MLP)](#exercise-1--image-classification-cnn-mlp)
- [Exercise 2 — Support Vector Machines (SVM)](#exercise-2--support-vector-machines-svm)
- [Exercise 3 — Autoencoders and Generative Models](#exercise-3--autoencoders-and-generative-models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview

This repository contains end-to-end implementations of classical machine learning classifiers (kNN, NCC, SVM) and deep learning models (CNN, MLP, Autoencoders). The project is divided into three main assignments, progressively advancing from basic statistical classifiers to complex generative neural networks.

The models are trained and evaluated on two standard benchmark datasets:
- **CIFAR-10:** Used for image classification tasks, consisting of 32x32 RGB images across 10 distinct classes (e.g., airplanes, automobiles, birds, cats).
- **MNIST:** Used for unsupervised learning, generative, and reconstruction tasks, consisting of grayscale handwritten digits.

**Key Highlights of the Repository:**
- **Traditional Machine Learning:** Explores instance-based learning (k-Nearest Neighbors, Nearest Class Centroid) and maximum-margin classifiers (Support Vector Machines). It evaluates how manual feature extraction techniques like Principal Component Analysis (PCA), Histogram of Oriented Gradients (HOG), and Color Histograms impact classical model performance.
- **Deep Learning for Classification:** Implements and compares Multi-Layer Perceptrons (MLPs) and custom Convolutional Neural Networks (CNNs). The CNN implementations heavily utilize modern training and regularization techniques, including Data Augmentation, Batch Normalization, Dropout, and Global Average Pooling to maximize accuracy and prevent overfitting.
- **Generative Autoencoders:** Investigates latent space representations through various Autoencoder architectures (Dense, Convolutional, and U-Net). The tasks go beyond simple dimensionality reduction and image reconstruction, extending into complex generative problems such as predicting the next sequential digit and visually generating the mathematical sum of two input digit images.

---

## Project Structure

```
.
├── CNN_MLP/                            # kNN, NCC, CNN, MLP on CIFAR-10
│   ├── dataProccessing.py
│   ├── kNearestNeighbors.py
│   ├── nearestClassCentroid.py
│   ├── convolutionalNeuralNetwork.py
│   ├── kNN-NCC.ipynb
│   ├── CNN.ipynb
│   ├── MLP.ipynb
│   └── report.pdf
│
├── SVM/                                # Support Vector Machines on CIFAR-10
│   ├── dataProccessing.py
│   ├── SVM_A.ipynb
│   ├── SVM_B.ipynb
│   ├── SVM_C.ipynb
│   ├── SVM_D.ipynb
│   └── report2.pdf
│
├── Autoencoder/                        # Autoencoders on MNIST
│   ├── dataProccessing.py
│   ├── nextDigit.py
│   ├── digitAdder.py
│   ├── Autoencoder_A.ipynb
│   ├── Autoencoder_B.ipynb
│   ├── Autoencoder_C.ipynb
│   └── report3.pdf
│
└── Neural_Networks_Deep_Learning.pdf   # Final project presentation

```

---

## Exercise 1 — Image Classification (CNN, MLP)

Focuses on classifying the CIFAR-10 dataset using traditional and deep learning architectures.
- **Preprocessing:** Dimensionality reduction using PCA to 50 dimensions.
- **Traditional Classifiers:** K-Nearest Neighbors utilizing Euclidean, Cosine, and Manhattan distances. Nearest Class Centroid approach was also evaluated.
- **Deep Learning:** An extensive Convolutional Neural Network (CNN) was designed using Conv2D layers with 32, 64, 128, and 256 filters, enhanced by Data Augmentation, Batch Normalization, Dropout, GlobalAveragePooling2D, and label smoothing. To demonstrate the CNN's superior spatial processing capabilities, a basic Multi-Layer Perceptron (MLP) was implemented purely for comparison.

---

## Exercise 2 — Support Vector Machines (SVM)

Investigates the performance of Support Vector Machines on CIFAR-10.
- **Binary Classification:** Groups the dataset into two super-classes: Animals and Vehicles.
- **Kernels Tested:** Linear, Polynomial, RBF, and Sigmoid.
- **Optimization:** Hyperparameter tuning was performed alongside feature extraction using Histogram of Oriented Gradients (HOG). Color Histogram Features were also evaluated.
- **Multi-class:** Evaluates SVM limits when transitioning from 2 to 10 classes.

---

## Exercise 3 — Autoencoders and Generative Models

Implements various Autoencoder architectures on the MNIST dataset.
- **Simple Digit Reconstruction:** Compares Dense Autoencoders and Convolutional Autoencoders against PCA. Quality was assessed using MSE, PSNR, and SSIM metrics.
- **Next Digit Reconstruction:** A generative regression task aimed at predicting the next digit. Implementations include Dense, Convolutional, and U-Net architectures.
- **Digit Addition Reconstruction:** Trains the network to visually output the sum of two input digits. Evaluated Dense, Convolutional, and U-Net models.

<div align="center">
  <img width="800" alt="Digit Addition Reconstruction Example" src="https://github.com/user-attachments/assets/de61d967-72c3-4657-b7bc-16c91d0bb457" />
</div>

---

## Results

| Exercise | Task / Problem | Best Performing Model | Dataset | Metric | Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Image Classification (10 classes) | Custom Convolutional Neural Network (CNN) | CIFAR-10 | Test Accuracy | **89.72%** |
| **1** | Image Classification (10 classes) | Multi-Layer Perceptron (MLP) | CIFAR-10 | Test Accuracy | 49.49% |
| **2** | Binary Classification (2 classes) | SVM (RBF Kernel) | CIFAR-10 | Test Accuracy | **92.59%** |
| **3** | Simple Digit Reconstruction | Convolutional Autoencoder | MNIST | Test MSE | **0.0024** |
| **3** | Next Digit Reconstruction | Convolutional Autoencoder | MNIST | Test MSE | 0.0605 |
| **3** | Digit Addition Reconstruction | Dense Autoencoder | MNIST | Test MSE | 0.0594 |

### Key Observations

* **CNN vs. MLP:** The CNN vastly outperformed the MLP on image data. Spatial layers (Conv2D) proved far superior to the MLP's flattened approach for processing the CIFAR-10 images.
* **SVM Limitations:** While the optimal SVM model achieved excellent accuracy (92.59%) for the 2-class problem, performance dropped significantly when shifting to all 10 classes, showing that SVMs excel when clear margins can be established between fewer categories.
* **Autoencoder Architecture:** While the Convolutional Autoencoder provided very accurate simple reconstruction, PCA reconstruction was nearly as effective and vastly more computationally efficient.
* **U-Net and Skip Connections:** In the generative visual addition task, the classic Dense architecture outperformed U-Net. The U-Net's Skip Connections inadvertently passed "raw" input artifacts directly to the output, confusing the decoder rather than helping it synthesize the mathematical sum.
---

## Installation

### Requirements

The projects utilize the following primary machine learning libraries:
- Keras of TensorFlow
- Scikit-Learn

Install general dependencies:
```bash
pip install tensorflow scikit-learn numpy matplotlib
```

## Usage

Navigate to the respective exercise directories and run the Jupyter Notebooks or Python scripts to train and evaluate the models.

```bash
# Assignment 1
cd CNN_MLP/
jupyter notebook kNN-NCC.ipynb
jupyter notebook CNN.ipynb
jupyter notebook MLP.ipynb

#Assignment 2
cd SVM/
jupyter notebook SVM_A.ipynb
jupyter notebook SVM_B.ipynb
jupyter notebook SVM_C.ipynb
jupyter notebook SVM_D.ipynb

#Assignment 3
cd Autoencoder/
jupyter notebook Autoencoder_A.ipynb
jupyter notebook Autoencoder_B.ipynb
jupyter notebook Autoencoder_C.ipynb
```
