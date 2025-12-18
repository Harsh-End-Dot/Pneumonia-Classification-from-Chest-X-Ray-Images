# Pneumonia-Classification-from-Chest-X-Ray-Images

## Project Overview
This repository presents a **deep learning–based pneumonia classification system** using chest X-ray images.  
The project evaluates multiple **state-of-the-art convolutional neural network (CNN) architectures** under a unified preprocessing and augmentation pipeline to identify the most effective model for binary classification of **NORMAL** vs **PNEUMONIA** cases.

---

## Motivation
-Pneumonia remains a major global health concern, causing approximately **2.5 million deaths annually**, with a significant impact on children and elderly populations.
- Manual interpretation of chest X-ray images is time-consuming and subjective, with diagnostic agreement among radiologists often reported around 70–80% in complex cases.
- Many healthcare settings face a shortage of experienced radiologists, leading to delayed or inconsistent diagnosis.
- Advances in deep learning and transfer learning enable automated analysis of chest X-rays, offering a consistent and scalable decision-support approach for pneumonia classification.

This project explores **AI-assisted diagnosis** using deep learning to provide **accurate, consistent, and scalable decision support** for pneumonia detection.

---

## Dataset
- **Source:** Kaggle Chest X-Ray Pneumonia Dataset
- **Original Size:** 5,856 images  
  - Pneumonia: 4,273  
  - Normal: 1,583
- **After Augmentation:** 7,000 images

### Dataset Split
| Set | Images | Shape |
|---|---|---|
| Training | 4,900 | (256, 256, 3) |
| Validation | 1,050 | (256, 256, 3) |
| Test | 1,050 | (256, 256, 3) |

Class imbalance was addressed by augmenting the **NORMAL** class to ensure balanced learning. 

---

## Preprocessing & Augmentation
### Preprocessing
- Image resizing to **256 × 256**
- Grayscale to **3-channel RGB conversion**
- Pixel normalization

### Augmentation Techniques
- Rotation (±10°)
- Horizontal & vertical shifts
- Zoom-in / zoom-out
- Brightness adjustment
- Horizontal flipping

These techniques improved generalization and reduced overfitting.

---

## Model Architectures
The following **transfer learning models** were implemented and evaluated:

1. **Xception**
2. **DenseNet121**
3. **ResNet50**
4. **InceptionV3**

Each model:
- Used ImageNet pretrained weights
- Employed a **custom classification head**
- Was trained using identical data splits and evaluation criteria

---

## Training Strategy
- Frozen pretrained backbone layers
- Custom dense layers with:
  - Batch Normalization
  - Dropout regularization
- Binary classification using **sigmoid activation**
- Optimized using standard deep learning training procedures

---

## Evaluation Metrics
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- AUC (Area Under ROC Curve)

---

## Results

### Performance Comparison (Test Set)

| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| Xception | ~95–96% | High | High | Strong |
| ResNet50 | ~94.86% | High | High | Strong |
| InceptionV3 | ~94.38% | High | High | Strong |
| **DenseNet121** | **96.95%** | **97.36%** | **97.66%** | **0.9940** |

**DenseNet121** achieved the best overall performance, demonstrating excellent feature reuse, stable convergence, and strong discriminative capability. 

---

## Observations
- All models achieved **>94% accuracy**
- DenseNet121 showed:
  - Minimal overfitting
  - Strong gradient flow
  - Superior generalization
- Training and validation curves showed **stable convergence** across models

---

## Technology Stack
- **Language:** Python
- **Frameworks:** TensorFlow, Keras
- **Libraries:** OpenCV, NumPy, Pandas, Matplotlib, scikit-learn
- **Environment:** Jupyter Notebook, Kaggle (GPU-enabled)

---

## How to Run
```bash
git clone <repository-url>
cd <repository-folder>
jupyter notebook
