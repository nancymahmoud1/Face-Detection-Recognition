import os
import numpy as np
import cv2
import joblib
import json
from numpy.linalg import norm

# Paths to saved model components
PCA_MODEL_PATH = "models/pca_model.pkl"
X_TRAIN_PROJ_PATH = "models/X_train_proj.npy"
Y_TRAIN_PATH = "models/y_train.npy"
LABEL_MAP_PATH = "models/label_map.json"

# Image size must match training
IMG_SIZE = (100, 100)


def load_models():
    """Loads the PCA model, training projections, labels, and label map."""
    pca = joblib.load(PCA_MODEL_PATH)
    X_train_proj = np.load(X_TRAIN_PROJ_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    return pca, X_train_proj, y_train, label_map


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_resized = cv2.resize(img, IMG_SIZE)
    img_equalized = cv2.equalizeHist(img_resized)  # ADD THIS TO MATCH prediction
    return img_equalized.flatten()


def predict(image_path):
    """
    Predicts the person identity from a test image.

    Returns:
        predicted_label: numeric label
        predicted_name: person folder name (e.g., 'person_07')
        distance: Euclidean distance to closest match
    """
    pca, X_train_proj, y_train, label_map = load_models()
    test_vector = preprocess_image(image_path)
    test_proj = pca.transform(test_vector.reshape(1, -1))

    # Compute Euclidean distances
    distances = np.linalg.norm(X_train_proj - test_proj, axis=1)
    best_match_idx = np.argmin(distances)

    predicted_label = int(y_train[best_match_idx])
    predicted_name = label_map[str(predicted_label)]
    return predicted_label, predicted_name, distances[best_match_idx]
