# Deep Learning ESA – Section A Theory Consolidation
_(Covers all papers: July 2022, Oct 2022, Mar 2023, Mar/Oct 2024, Model Paper)_

## 🔹 1. Activation Functions in Neural Networks

**Definition**: Activation functions introduce **non-linearity** in neural networks, allowing them to learn complex patterns.

**Common Types**:
- **Sigmoid**: `σ(x) = 1 / (1 + e^(-x))` → Range (0, 1)
- **Tanh**: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))` → Range (-1, 1)
- **ReLU**: `f(x) = max(0, x)`
- **Softmax**: `exp(x_i) / Σexp(x_j)` → Used in multi-class output layer

---

## 🔹 2. CNN Architecture & Layer Usage

**Components**:
- **Convolution Layer**: Extracts features
- **Pooling Layer**: Reduces feature size
- **Flatten Layer**: Converts matrix to vector
- **Dense Layer**: Final classification

---

## 🔹 3. Overfitting in Neural Networks

**Definition**: Good training accuracy, poor test accuracy.

**Causes**: Too complex, small data, long training

**Remedies**: Dropout, L2 regularization, early stopping, data augmentation

---

## 🔹 4. Vanishing Gradient Problem

**Problem**: Gradients become very small, slowing learning

**Solution**: ReLU, Batch Norm, ResNet, LSTM

---

## 🔹 5. Learning Rate & Momentum

- **Learning Rate**: Size of weight update
- **Momentum**: Adds past update to current for faster convergence

---

## 🔹 6. Batch Normalization

**Definition**: Normalizes inputs per mini-batch

**Benefits**: Faster training, better generalization

---

## 🔹 7. Gradient Descent & Variants

- **Batch GD**: Full dataset
- **SGD**: One sample
- **Mini-Batch GD**: Small batch (best of both)

---

## 🔹 8. Object Detection Metrics

- **IoU**, **Precision**, **Recall**, **mAP**, **F1 Score**

---

## 🔹 9. Non-Maximum Suppression (NMS)

**Definition**: Suppresses overlapping boxes, keeps the most confident one.

---

## 🔹 10. Transfer Learning

**Definition**: Reuse of pretrained models for new tasks

**Benefits**: Fast, requires less data, powerful features reused

---

## 🔹 11. Siamese Network

**Definition**: Twin networks comparing input pairs

**Applications**: Face verification, few-shot learning

---

## 🔹 12. Backpropagation

**Definition**: Chain-rule-based gradient calculation

**Types**: Static (Feedforward), Recurrent (RNN)

---

## 🔹 13. YOLO Algorithm

**Definition**: Real-time object detection, one-shot

**Steps**: Grid-based prediction, bounding boxes, NMS

---

## 🔹 14. Single vs Multi-Stage Object Detection

| Type           | Description                          | Example     |
|----------------|--------------------------------------|-------------|
| Single-stage   | Fast, direct prediction               | YOLO, SSD   |
| Multi-stage    | Two steps: region proposal + refine  | Faster R-CNN|

---

## 🔹 15. GANs (Generative Adversarial Networks)

**Definition**: Generator + Discriminator trained together

**Uses**: Image synthesis, augmentation, super-resolution

---

## 🔹 16. Need for Activation Functions

**Purpose**: To add non-linearity and learn complex functions

---