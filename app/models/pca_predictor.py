import numpy as np, cv2, json
from pathlib import Path

IMG_SIZE = (100, 100)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ─────────── load model ───────────
mean_vec = np.load("models/pca_mean.npy")
std_vec = np.load("models/pca_std.npy")
PCs = np.load("models/pca_components.npy")
X_train = np.load("models/pca_embeddings.npy")  # already L2-normed
y_train = np.load("models/pca_labels.npy")
label_map = json.loads(("models/label_map.json").read_text())


# ─────────── helper ───────────────
def preprocess(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    eq = clahe.apply(gray).astype(np.float32).flatten()
    # z-score
    z = (eq - mean_vec) / std_vec
    # project & L2-norm
    proj = (z @ PCs)
    proj = proj / (np.linalg.norm(proj) + 1e-9)
    return proj


# ─────────── predict ──────────────
def predict(img_path: str):
    q = preprocess(img_path)
    # cosine distance = 1 - cosine similarity (since both vectors are L2-normed)
    cos_sim = X_train @ q
    best_idx = int(np.argmax(cos_sim))
    pred_lab = int(y_train[best_idx])
    pred_name = label_map[str(pred_lab)]
    distance = 1 - cos_sim[best_idx]  # 0 = perfect match, 2 = opposite
    return pred_lab, pred_name, float(distance)
