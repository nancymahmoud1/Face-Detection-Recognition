import os
import json
import numpy as np
from pathlib import Path
import warnings

# Suppress joblib/loky warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky")

# Dynamically resolve project root directory
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
REQUIRED_FILES = [
    MODEL_DIR / "pca_mean.npy",
    MODEL_DIR / "pca_std.npy",
    MODEL_DIR / "pca_components.npy",
    MODEL_DIR / "pca_embeddings.npy",
    MODEL_DIR / "pca_labels.npy",
    MODEL_DIR / "label_map.json",
]


def ensure_model_exists():
    """Ensure PCA model exists, or train and save it."""
    if not all(f.exists() for f in REQUIRED_FILES):
        # print("🚧 PCA model files not found. Training model first…")
        from app.models import pca_trainer
        pca_trainer.train_and_save()


def main():
    ensure_model_exists()
    from app.controller import MainWindowController

    controller = MainWindowController()
    controller.run()


if __name__ == "__main__":
    main()
