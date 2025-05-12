import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# ─────────────── CONFIG ───────────────
os.makedirs("../../static/model evaluation", exist_ok=True)
TEST_DIR = Path("../../datasets/Processed/test/grayscale")
MODEL_DIR = Path("../../models")
SAVE_DIR = Path("../../static/model evaluation")
ROC_PATH = SAVE_DIR / "roc_curve.png"
CM_PATH = SAVE_DIR / "confusion_matrix.png"
IMG_SIZE = (100, 100)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ─────────────── LOAD MODEL ───────────────
def load_pca_model():
    mean = np.load(MODEL_DIR / "pca_mean.npy")
    std = np.load(MODEL_DIR / "pca_std.npy")
    PCs = np.load(MODEL_DIR / "pca_components.npy")
    embeddings = np.load(MODEL_DIR / "pca_embeddings.npy")
    labels = np.load(MODEL_DIR / "pca_labels.npy")
    label_map = json.loads((MODEL_DIR / "label_map.json").read_text())
    return mean, std, PCs, embeddings, labels, label_map


# ─────────────── IMAGE PREPROCESSING ───────────────
def preprocess_image(image_path, mean, std, PCs):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    eq = clahe.apply(gray).astype(np.float32).flatten()
    z = (eq - mean) / std
    proj = z @ PCs
    proj = proj / (np.linalg.norm(proj) + 1e-9)
    return proj


# ─────────────── COSINE SIMILARITY ───────────────
def cosine_similarity(vec, mat):
    return mat @ vec


# ─────────────── EVALUATION + PLOTTING ───────────────
def evaluate_and_plot():
    mean, std, PCs, train_embeds, train_labels, label_map = load_pca_model()

    y_true = []
    y_pred = []
    y_scores = []

    for person_folder in sorted(TEST_DIR.iterdir()):
        if not person_folder.is_dir():
            continue
        person_id = person_folder.name
        for img_path in sorted(person_folder.glob("*.jpg")):
            test_proj = preprocess_image(img_path, mean, std, PCs)
            sims = cosine_similarity(test_proj, train_embeds)
            best_idx = np.argmax(sims)
            pred_label = train_labels[best_idx]

            # Append data
            true_label = [k for k, v in label_map.items() if v == person_id][0]
            y_true.append(int(true_label))
            y_pred.append(int(pred_label))
            y_scores.append(np.max(sims))

    y_binary = [int(t == p) for t, p in zip(y_true, y_pred)]
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_binary, y_scores)
    roc_auc = auc(fpr, tpr)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    label_list = [label_map[str(i)] for i in range(len(label_map))]

    plot_roc(fpr, tpr, roc_auc, ROC_PATH)
    save_confusion_matrix(cm, label_list, CM_PATH)

    print("✅ Evaluation Completed")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC Score: {roc_auc:.4f}")
    print("Confusion Matrix:\n", cm)
    print(f"🖼️  ROC curve saved to: {ROC_PATH}")
    print(f"🖼️  Confusion matrix saved to: {CM_PATH}")


# ─────────────── ROC PLOT ───────────────
def plot_roc(fpr, tpr, roc_auc, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – PCA Face Recognition")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ─────────────── CONFUSION MATRIX PLOT ───────────────
def save_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix – PCA Face Recognition")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ─────────────── MAIN ───────────────
if __name__ == "__main__":
    evaluate_and_plot()