# Deep Learning Exam – July 2022
## Section A – Theory Answers (20 Point)

---

### **Q1 a) List and brief about the activation functions in neural networks.** *(4 Point)*

#### 1. **Sigmoid**
- Formula: `σ(x) = 1 / (1 + e^(-x))`
- Output Range: (0, 1)
- Commonly used in binary classification
- **Limitation**: Saturates for large inputs, leading to vanishing gradients

#### 2. **Tanh**
- Formula: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
- Output Range: (-1, 1)
- Zero-centered activation, better than sigmoid

#### 3. **ReLU (Rectified Linear Unit)**
- Formula: `f(x) = max(0, x)`
- Fast and efficient
- **Limitation**: Dying ReLU problem when neurons get stuck

#### 4. **Softmax**
- Converts raw scores into probability distribution
- Used in the output layer for multi-class classification

---

### **Q1 b) What is the role of convolution, max-pool, and flatten layers in CNN?** *(4 Point)*

| Layer          | Role                                                                 |
|----------------|----------------------------------------------------------------------|
| Convolution    | Extracts local features using filters (e.g., edges, textures)       |
| Max Pooling    | Reduces dimensionality while retaining important features           |
| Flatten        | Converts 2D feature maps to 1D vector before feeding to dense layers |

---

### **Q1 c) What are the various methods to handle overfitting in Neural Networks?** *(4 Point)*

1. **Dropout** – Randomly drops neurons during training to prevent co-adaptation
2. **L2 Regularization** – Penalizes large weights by adding them to the loss function
3. **Early Stopping** – Stops training when validation error starts increasing
4. **Data Augmentation** – Expands dataset with transformations (flip, rotate, etc.)
5. **Cross-Validation** – Ensures model generalization on unseen data

---

### **Q1 d) How is transfer learning helpful for prediction?** *(4 Point)*

- Uses a **pre-trained model** (e.g., on ImageNet) for a new task
- Helps reduce training time and data requirements
- Retains useful low-level features (edges, corners)
- Common in computer vision and NLP
- Only top layers are retrained for new task

---

### **Q1 e) Briefly explain Siamese Network.** *(4 Point)*

- A Siamese Network has **two or more identical subnetworks** with shared weights
- Measures **similarity** between two inputs
- Outputs feature embeddings which are compared using a distance metric
- **Applications**: Face verification, signature matching, one-shot learning
- Learns to tell whether two inputs are similar or different

---


# Deep Learning Exam – October 2022  
## Section A – Theory Answers (20 Point)

---

### **Q1 a) What are the different types of pooling layers (at least 3)? Explain each briefly.** *(4 Point)*

#### 1. **Max Pooling**
- Selects the maximum value from a patch of the feature map.
- Helps retain the most dominant features.

#### 2. **Average Pooling**
- Takes the average of values in the patch.
- Used when feature intensity is less important.

#### 3. **Global Average Pooling**
- Averages entire feature maps to produce a single number per map.
- Reduces overfitting and used before dense layers.

---

### **Q1 b) How many parameters need to be trained for the network:**  
_Input layer: 6 → Hidden Layer: 4 → Hidden Layer: 2 → Output layer (binary)_ *(4 Point)*

#### Parameters to be trained:

1. **Input to Hidden1 (6 → 4)**:  
   \( (6 \times 4) + 4 = 28 \)  

2. **Hidden1 to Hidden2 (4 → 2)**:  
   \( (4 \times 2) + 2 = 10 \)

3. **Hidden2 to Output (2 → 1)**:  
   \( (2 \times 1) + 1 = 3 \)

✅ **Total parameters = 28 + 10 + 3 = 41**

---

### **Q1 c) What is Vanishing Gradient Problem? How to get rid of this?** *(4 Point)*

- In deep networks, during backpropagation, gradients become very small in early layers.
- Leads to **very slow or no learning** in initial layers.

#### Solutions:
- Use **ReLU** instead of sigmoid/tanh
- Use **Batch Normalization**
- Use **Residual connections** (as in ResNet)
- Use architectures like **LSTM/GRU** for sequential tasks

---

### **Q1 d) List various metrics for object detection. Why is IoU used?** *(4 Point)*

#### Metrics:
- **IoU (Intersection over Union)**
- **mAP (mean Average Precision)**
- **Precision & Recall**
- **F1 Score**

#### Why use IoU?
- Measures overlap between predicted and ground truth boxes.
- IoU > threshold (usually 0.5) → correct detection.
- Helps evaluate localization accuracy.

---

### **Q1 e) Explain Backpropagation. What are the two types of Backpropagation Networks?** *(4 Point)*

#### Backpropagation:
- Algorithm to compute gradients for updating weights using **chain rule**

# Deep Learning Exam – March 2023  
## Section A – Theory Answers (20 Point)

---

### **Q1 a) Summarize the Neural Network process. Write a short note on Learning Rate and Momentum.** *(4 Point)*

#### ✅ Neural Network Process:
1. **Input layer** receives raw data.
2. Data flows through **hidden layers**, where weights and biases are applied.
3. **Activation functions** add non-linearity.
4. **Output layer** makes prediction.
5. Loss is computed using a **loss function**.
6. **Backpropagation** adjusts weights to minimize the loss.
7. Process repeats for multiple **epochs**.

#### ✅ Learning Rate:
- Controls the size of weight updates during training.
- Too small → slow learning; too large → overshooting minima.

#### ✅ Momentum:
- Adds a fraction of previous update to the current one.
- Helps accelerate convergence and escape local minima.

---

### **Q1 b) What is Batch Normalization? What are the benefits of Batch Normalization?** *(4 Point)*

#### ✅ Batch Normalization:
- Technique to normalize the inputs to a layer for each mini-batch.
- Helps maintain a mean of 0 and standard deviation of 1.

#### ✅ Benefits:
- Speeds up training.
- Reduces internal covariate shift.
- Allows higher learning rates.
- Acts as a regularizer, sometimes reducing the need for dropout.

---

### **Q1 c) What is Gradient Descent? List 3 variations and explain each briefly.** *(4 Point)*

#### ✅ Gradient Descent:
- Optimization algorithm to minimize loss by updating weights in the direction of negative gradient.

#### ✅ Variations:
1. **Batch Gradient Descent**  
   - Uses entire dataset for each update.  
   - Stable but slow and memory intensive.

2. **Stochastic Gradient Descent (SGD)**  
   - Uses one data point per update.  
   - Faster, but has high variance.

3. **Mini-Batch Gradient Descent**  
   - Uses small batches of data.  
   - Combines speed and stability.

---

### **Q1 d) Why do we need Activation Functions? Write a short note on ReLU. What are the drawbacks of ReLU?** *(4 Point)*

#### ✅ Purpose of Activation Functions:
- Introduce non-linearity to the model.
- Enable the network to learn complex patterns and relationships.

#### ✅ ReLU (Rectified Linear Unit):
- Formula: `f(x) = max(0, x)`
- Simple and computationally efficient.
- Helps avoid vanishing gradient problem.

#### ❌ Drawbacks:
- **Dying ReLU**: neurons may output zero for all inputs.
- Not zero-centered.

---

### **Q1 e) Explain the algorithm behind YOLO.** *(4 Point)*

#### ✅ YOLO (You Only Look Once):
- Single-shot object detection algorithm.
- Divides image into grid cells; each predicts bounding boxes and class probabilities.

#### Steps:
1. Input image is divided into `S x S` grid.
2. Each grid predicts:
   - **Bounding box** coordinates (x, y, w, h)
   - **Confidence score**
   - **Class probabilities**
3. Non-Maximum Suppression (NMS) removes redundant detections.

#### ✅ Advantages:
- Real-time speed
- End-to-end prediction
- Fewer false positives

---
# Deep Learning Exam – March/October 2024  
## Section A – Theory Answers (20 Point)

---

### **Q1 a) What are causes of overfitting in a deep learning model? Name 2 methods to resolve.** *(4 Point)*

#### ✅ Causes of Overfitting:
- Too complex model with many parameters
- Insufficient training data
- Too many training epochs
- Noisy or irrelevant features
- Lack of regularization

#### ✅ Methods to Reduce Overfitting:
1. **Dropout** – Randomly disables neurons during training to reduce over-reliance.
2. **Data Augmentation** – Increases dataset size and variability using transformations.
(Also acceptable: L2 regularization, early stopping, cross-validation)

---

### **Q1 b) What is the usage of Convolution, Pooling, and Dense layers in CNN?** *(4 Point)*

| Layer       | Purpose                                                              |
|-------------|----------------------------------------------------------------------|
| **Convolution** | Detects spatial patterns/features using learnable filters         |
| **Pooling**     | Reduces feature map size, controls overfitting, speeds up training |
| **Dense**       | Fully connected layer used for final classification               |

---

### **Q1 c) What is the Vanishing Gradient Problem? How to get rid of this?** *(4 Point)*

#### ✅ Vanishing Gradient Problem:
- In deep networks, gradients get smaller as they are backpropagated.
- Early layers receive almost zero updates → learning halts.

#### ✅ Solutions:
- Use **ReLU** activation instead of sigmoid/tanh
- Apply **Batch Normalization**
- Use **Residual connections** (ResNet)
- Use **LSTM** or **GRU** for sequential models

---

### **Q1 d) List various performance metrics for object detection. What is the use of Non-Maximal Suppression?** *(4 Point)*

#### ✅ Metrics:
- **IoU (Intersection over Union)**
- **Precision / Recall**
- **mAP (mean Average Precision)**
- **F1 Score**

#### ✅ Non-Maximal Suppression (NMS):
- Eliminates multiple overlapping boxes for the same object.
- Keeps only the box with the highest confidence.
- Improves detection accuracy and reduces duplicates.

---

### **Q1 e) Why are activation functions needed in deep neural networks? Define Tanh, ReLU, and Softmax.** *(4 Point)*

#### ✅ Purpose:
- Introduce **non-linearity**
- Allow networks to model complex patterns

#### ✅ Activation Functions:

| Function  | Formula                        | Usage                                   |
|-----------|--------------------------------|-----------------------------------------|
| **Tanh**  | `(e^x - e^(-x)) / (e^x + e^(-x))` | Range: (-1, 1), zero-centered           |
| **ReLU**  | `f(x) = max(0, x)`              | Fast training, avoids vanishing gradient |
| **Softmax** | `exp(x_i) / Σexp(x_j)`        | Converts scores to probabilities; used in output layer for multi-class classification |

---
# Deep Learning Exam – Model Question Paper  
## Section A – Theory Answers (20 Point)

---

### **Q1 a) Explain the Convolutional Neural Network (CNN) architecture in detail.** *(4 Point)*

#### ✅ CNN Architecture Components:

1. **Input Layer**  
   - Takes image data (e.g., 128x128x3)

2. **Convolution Layers**  
   - Apply filters/kernels to extract features like edges, textures  
   - Output: feature maps

3. **Activation Function (ReLU)**  
   - Adds non-linearity

4. **Pooling Layers (Max/Average Pooling)**  
   - Downsamples feature maps to reduce spatial size  
   - Helps with translation invariance

5. **Flatten Layer**  
   - Converts 2D data into 1D vector for Dense layers

6. **Dense (Fully Connected) Layers**  
   - Final decision-making layers  
   - Often followed by softmax for classification

7. **Output Layer**  
   - Produces final class predictions

---

### **Q1 b) Explain overfitting in neural networks. How to overcome it?** *(4 Point)*

#### ✅ Overfitting:
- Model performs well on training data but poorly on unseen/test data
- Learns noise or irrelevant details

#### ✅ Solutions:
- **Dropout** – Randomly deactivate neurons
- **Data Augmentation** – Increases effective dataset size
- **Early Stopping** – Stops training when validation loss increases
- **Regularization (L2)** – Penalizes large weights

---

### **Q1 c) What are activation functions? Why are they used in neural networks?** *(4 Point)*

#### ✅ Activation Functions:
- Mathematical functions applied after each layer
- Introduce **non-linearity** into the model
- Without them, deep models behave like linear functions

#### ✅ Examples:
- **ReLU** – Used in hidden layers
- **Tanh** – Output centered at 0
- **Sigmoid** – Output between 0 and 1
- **Softmax** – Used in output layer for multi-class classification

---

### **Q1 d) Difference between single-stage and multi-stage object detection models?** *(4 Point)*

| Type           | Description                                                        | Example           |
|----------------|--------------------------------------------------------------------|-------------------|
| **Single-stage** | Detects objects in one forward pass over image                   | YOLO, SSD         |
| **Multi-stage**  | Generates region proposals, then classifies and refines them     | Faster R-CNN      |

- Single-stage: Fast, less accurate  
- Multi-stage: Slower, more accurate

---

### **Q1 e) Briefly explain GANs. What are their advantages?** *(4 Point)*

#### ✅ GAN (Generative Adversarial Network):
- Comprises two networks:
  1. **Generator** – Tries to produce fake data
  2. **Discriminator** – Tries to distinguish real vs fake data
- They are trained adversarially until generator produces realistic data

#### ✅ Advantages:
- Can generate highly realistic data (images, text, audio)
- Used in image synthesis, super-resolution, data augmentation
- Helps in creating synthetic training data for low-resource domains

---
