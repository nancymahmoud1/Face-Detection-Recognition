import os
import numpy as np
import cv2
import joblib
import json
from sklearn.decomposition import PCA

# Configuration
GRAYSCALE_DIR = "../../datasets/Processed/train/grayscale"
RGB_DIR = "../../datasets/Processed/train/RGB"
MODEL_DIR = "../../models"
IMG_SIZE = (100, 100)
N_COMPONENTS = 50


def preprocess_image(image_path):
    """
    Reads an image, converts it to grayscale, resizes and equalizes it.
    Returns the flattened vector.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, IMG_SIZE)
    img_equalized = cv2.equalizeHist(img_resized)
    return img_equalized.flatten()


def load_training_data():
    """
    Loads training data from both grayscale and RGB folders.
    Converts RGB images to grayscale before processing.
    Returns:
        X (np.array): image vectors
        y (np.array): corresponding labels
        label_map (dict): mapping from label to person folder
    """
    X = []
    y = []
    label_map = {}
    current_label = 0

    for person_folder in sorted(os.listdir(GRAYSCALE_DIR)):
        gray_path = os.path.join(GRAYSCALE_DIR, person_folder)
        rgb_path = os.path.join(RGB_DIR, person_folder)

        if not os.path.isdir(gray_path):
            continue

        label_map[str(current_label)] = person_folder
        count = 0

        # Load grayscale images
        gray_images = sorted(f for f in os.listdir(gray_path) if f.endswith(".jpg"))
        for image_name in gray_images:
            image_path = os.path.join(gray_path, image_name)
            X.append(preprocess_image(image_path))
            y.append(current_label)
            count += 1

        # Load RGB images (converted to grayscale)
        if os.path.isdir(rgb_path):
            rgb_images = sorted(f for f in os.listdir(rgb_path) if f.endswith(".jpg"))
            for image_name in rgb_images:
                image_path = os.path.join(rgb_path, image_name)
                X.append(preprocess_image(image_path))
                y.append(current_label)
                count += 1

        print(f"📁 Loaded {count} images for {person_folder}")
        current_label += 1

    return np.array(X), np.array(y), label_map


def train_and_save():
    """
    Loads data, trains PCA model, and saves everything.
    """
    print("📦 Loading training data...")
    X_train, y_train, label_map = load_training_data()

    print("📊 Fitting PCA model...")
    pca = PCA(n_components=N_COMPONENTS)
    X_train_proj = pca.fit_transform(X_train)

    print("💾 Saving model and data...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca_model.pkl"))
    np.save(os.path.join(MODEL_DIR, "X_train_proj.npy"), X_train_proj)
    np.save(os.path.join(MODEL_DIR, "y_train.npy"), y_train)
    with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print("✅ Training complete. Model saved to /models.")


if __name__ == "__main__":
    train_and_save()
