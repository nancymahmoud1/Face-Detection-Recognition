## FaceVector

### Overview

**FaceVector** is a modular computer vision system that performs both **face detection** and **face recognition**. Detection is handled in real time using OpenCV’s Haar cascades, providing a fast and reliable way to localize faces in images.

The recognition component is powered by **machine learning**, using **Principal Component Analysis (PCA)** to extract facial embeddings and classify identities. The system supports RGB and grayscale datasets, trains automatically on first run, and evaluates performance using standard classification metrics and ROC analysis.

> This project demonstrates how classical detection methods and unsupervised learning can be combined into a reproducible pipeline for facial analysis—bridging traditional vision techniques with lightweight machine learning.

![GUI Screenshot](https://github.com/user-attachments/assets/617433f4-ee4e-4b1b-bafb-de1ccef0462a)

---

### Features

* Real-time face detection using OpenCV Haar cascades
* PCA-based face recognition with dynamic model training
* Intuitive interface with image viewer and detection settings
* Supports both **grayscale** and **RGB** datasets
* Train/test split logic built-in for reproducibility
* Performance evaluation using ROC curves
* Auto-caching and log tracking for efficiency

---

### 🧠 Model Overview: PCA for Face Recognition

FaceVector uses **Principal Component Analysis (PCA)** to reduce high-dimensional image data into a compact set of features known as **eigenfaces**. This enables fast and interpretable face recognition with limited training data.

#### 📌 How PCA Works in FaceVector:

1. **Flattening** – Each face image is converted into a 1D vector.
2. **Covariance Matrix** – A matrix capturing variance between pixel intensities is computed.
3. **Eigen Decomposition** – The eigenvectors (eigenfaces) represent directions of maximum variance.
4. **Projection** – New face images are projected into this low-dimensional eigenspace.
5. **Classification** – Recognition is done by comparing the projected test image with training projections using **Euclidean distance**.

Only **four images per person** are used for training, ensuring the model remains efficient while demonstrating the effectiveness of unsupervised learning.

---

### How It Works

#### Recognition Pipeline:

* The datasets are split into **train/test folders**, where each person has **4 training images**.
* On the **first run**, the app:

  * Preprocesses the selected dataset (RGB or grayscale)
  * Trains a **PCA model** using the training images
  * Saves the PCA model, embeddings, and labels for future use
* On later runs:

  * If models exist, the app loads them automatically.
  * Users select an image from the test set.
  * The PCA model compares it to trained embeddings and predicts identity using **Euclidean distance**.

#### 🔍 Detection Pipeline:

* Face detection is powered by OpenCV Haar cascades:

  * `cascadeFaceClassifier.xml`
  * `haarcascade_frontalface_default.xml`
  * `haarcascade_eye.xml`
  * `haarcascade_smile.xml`

---

### Installation

```bash
git clone https://github.com/YassienTawfikk/FaceVector.git
cd FaceVector
pip install -r requirements.txt
python main.py
```

---

### 🛠 Tech Stack

* **Programming Language:** Python
* **GUI:** PyQt5, Qt Designer
* **Libraries:** OpenCV, NumPy, scikit-learn, Matplotlib
* **ML Model:** Principal Component Analysis (PCA)
* **Datasets:** AT\&T (ORL), Georgia Tech (GT)

---

### 📊 Performance Evaluation

FaceVector evaluates its PCA-based recognition model using standard classification metrics and ROC curve visualization. The system achieves high accuracy and reliability even with a small training set per identity.

#### 📈 ROC Curve

![ROC Curve](https://github.com/user-attachments/assets/76535f8d-79c4-43f7-a342-e28f459d9aa4)

> The ROC curve above illustrates the model's ability to distinguish between different identities across thresholds.

#### 🧪 Evaluation Metrics

| Metric      | Value  |
| ----------- | ------ |
| Accuracy    | 0.975  |
| Precision   | 0.9833 |
| Recall      | 0.975  |
| F1 Score    | 0.9733 |
| Specificity | 0.9987 |
| AUC Score   | 0.9231 |

These results confirm the model’s effectiveness in both correctly identifying known faces and minimizing false positives, even with only four training images per person.

---

## Contributions

<div>
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/YassienTawfikk" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126521373?v=4" width="150px;" alt="Yassien Tawfik"/>
        <br />
        <sub><b>Yassien Tawfik</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/madonna-mosaad" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/127048836?v=4" width="150px;" alt="Madonna Mosaad"/>
        <br />
        <sub><b>Madonna Mosaad</b></sub>
      </a>
    </td>    
    <td align="center">
      <a href="https://github.com/nancymahmoud1" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/125357872?v=4" width="150px;" alt="Nancy Mahmoud"/>
        <br />
        <sub><b>Nancy Mahmoud</b></sub>
      </a>
    </td>    
    <td align="center">
      <a href="https://github.com/nariman-ahmed" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126989278?v=4" width="150px;" alt="Nariman Ahmed"/>
        <br />
        <sub><b>Nariman Ahmed</b></sub>
      </a>
    </td>        
  </tr>
</table>
</div>

---
