# Convolutional Neural Networks (CNN) for Image Classification  

This repository contains implementations of **Convolutional Neural Networks (CNNs)** applied to the **CIFAR-10 dataset** for image classification.  
It includes both a baseline CNN implementation and improvements such as **Batch Normalization**, **Dropout**.
---

## Project Overview  
Convolutional Neural Networks (CNNs) are powerful deep learning models designed to learn spatial hierarchies of features from images.  
They are widely used in computer vision tasks such as object detection, facial recognition, and medical image analysis.  

This project explores:  
- A CNN implemented with TensorFlow/Keras
- Performance improvements using Batch Normalization, Dropout, and Fully Connected Layer tuning

---

## Dataset  
- **CIFAR-10** dataset  
- **60,000 color images** (32x32 pixels, 10 classes)  
  - **50,000 training images**  
  - **10,000 test images**  
- Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck  

---

## CNN Architecture  

### Baseline CNN (TensorFlow/Keras)  
- **Conv Layer 1:** 32 filters (3×3), ReLU  
- **Conv Layer 2:** 32 filters (3×3), ReLU  
- **Max Pooling:** 2×2  
- **Conv Layer 3:** 64 filters (3×3), ReLU  
- **Conv Layer 4:** 64 filters (3×3), ReLU  
- **Max Pooling:** 2×2  
- **Conv Layer 5:** 128 filters (3×3), ReLU  
- **Max Pooling:** 2×2  
- **Flatten**  
- **Dense Layer:** 256 neurons, ReLU  
- **Output Layer:** Softmax (10 classes)  
- **Optimizer:** Adam (learning rate = 0.001)  
- **Loss:** Sparse categorical cross-entropy  
- **Training:** 20 epochs, batch size = 128  

---

## Results  

| Model                | Accuracy | Loss   |
|-----------------------|----------|--------|
| Baseline CNN          | 70.64%   | 1.3005 |
| Improved CNN          | **79.33%** | 0.6270 |

---

## Improvements  

1. **Batch Normalization**  
   - Normalizes activations after convolution layers.  
   - Stabilizes training and speeds convergence.  

2. **Dropout**  
   - Prevents overfitting by randomly dropping neurons during training.  
   - Applied: 25% after convolutional layers, 50% after dense layer.  

3. **Increased Neurons in Dense Layer**  
   - Increased from **256 → 512** for improved learning capacity.  

4. **Learning Rate Tuning**  
   - Adam optimizer with **learning rate = 0.001** improved convergence.  

---

## Transfer Learning (Bonus)  
- Implemented **ResNet18** using **PyTorch**  
- Modified final fully connected layer (512 → 10 for CIFAR-10 classes)  
- Optimizer: **SGD, learning rate = 0.001**  

---

## Conclusion  
- CNN on CIFAR-10 achieved **70.64% accuracy** baseline.  
- With improvements (BatchNorm, Dropout, FC tuning, LR tuning), accuracy increased to **79.33%**.  

---

## Requirements  
Install dependencies:  
```bash
pip install numpy tensorflow keras torch torchvision matplotlib
```

---

## How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/harsh7z/cnn-image-classification.git
   cd cnn-image-classification
   ```
2. Run the notebooks:  
   ```bash
   jupyter notebook hw4.ipynb
   ```
3. Train models and evaluate performance.  
