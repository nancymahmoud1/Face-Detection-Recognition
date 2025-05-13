import json
import numpy as np
import cv2
from pathlib import Path
from app.models.roc import evaluate_and_plot
from pathlib import Path

# Dynamically resolve base project directory from this script’s location
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # assuming you're in app/models

# Consistent cross-platform paths
GRAYSCALE_DIR = PROJECT_ROOT / "datasets/Processed/train/grayscale"
RGB_DIR = PROJECT_ROOT / "datasets/Processed/train/RGB"
MODEL_DIR = PROJECT_ROOT / "models"

# Config constants
IMG_SIZE = (100, 100)
VAR_THRESH = 0.95
MAX_COMPONENTS = 150

# ───────────────────────────────── preprocessing ────────────────────────────────
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(img_path: Path) -> np.ndarray:
    """read → to-grey → resize → CLAHE → flatten → float32"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    eq = clahe.apply(gray)
    return eq.astype(np.float32).flatten()


# ───────────────────────────────── data loading ────────────────────────────────
def load_data():
    X, y, label_map = [], [], {}
    label = 0

    for person in sorted(p.name for p in GRAYSCALE_DIR.iterdir() if p.is_dir()):
        label_map[str(label)] = person
        # grayscale imgs
        for p in sorted((GRAYSCALE_DIR / person).glob("*.jpg")):
            X.append(preprocess(p));
            y.append(label)
        # RGB imgs (converted to grey)
        for p in sorted((RGB_DIR / person).glob("*.jpg")):
            X.append(preprocess(p));
            y.append(label)
        # print(f"📁  loaded {y.count(label):2d} imgs  → {person}")
        label += 1

    return np.vstack(X), np.array(y, dtype=np.int32), label_map


# ───────────────────────────────── PCA via SVD ─────────────────────────────────
def train_and_save():
    X, y, label_map = load_data()

    # z-score normalise pixels (per-feature)
    mean_vec = X.mean(0)
    std_vec = X.std(0) + 1e-9  # avoid /0
    Xz = (X - mean_vec) / std_vec

    # economical SVD
    # print("   running SVD …")
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)

    # decide k to keep ≥ VAR_THRESH variance
    var_ratio = (S ** 2) / np.sum(S ** 2)
    cum_var = np.cumsum(var_ratio)
    k = int(np.searchsorted(cum_var, VAR_THRESH) + 1)
    k = min(k, MAX_COMPONENTS)
    # print(f"   → keeping {k} PCs (cum. var = {cum_var[k - 1]:.3f})")

    PCs = Vt[:k].T  # (features × k)
    X_proj = Xz @ PCs  # (n_samples × k)
    # L2-normalise projections (for cosine similarity later)
    X_proj = X_proj / np.linalg.norm(X_proj, axis=1, keepdims=True)

    # ─── save ───
    MODEL_DIR.mkdir(exist_ok=True)
    np.save(MODEL_DIR / "pca_mean.npy", mean_vec)
    np.save(MODEL_DIR / "pca_std.npy", std_vec)
    np.save(MODEL_DIR / "pca_components.npy", PCs.astype(np.float32))
    np.save(MODEL_DIR / "pca_embeddings.npy", X_proj.astype(np.float32))
    np.save(MODEL_DIR / "pca_labels.npy", y)
    (MODEL_DIR / "label_map.json").write_text(json.dumps(label_map, indent=2))

    # print("✅  PCA model + embeddings saved to /models")

    # ─── evaluate model performance on test set ───
    # print("🔍  Running evaluation on test set …")

    evaluate_and_plot()
