from pathlib import Path
import random
import shutil
import cv2
from typing import List, Sequence


class FaceDatasetPreprocessor:
    """
    ATnT  → grayscale  (no face crop)
    GeorgiaTech → RGB  (face-cropped)

    Final tree
    └── Processed/
        ├── train/
        │   ├── grayscale/   person_xx/…
        │   └── RGB/         person_xx/…
        └── test/
            ├── grayscale/   person_xx/…
            └── RGB/         person_xx/…
    """

    _DST_MAP = {"ATnT": "grayscale", "GeorgiaTech": "RGB"}

    def __init__(
            self,
            root_dir: str | Path,
            clean_root: str | Path,
            datasets: Sequence[str] = ("ATnT", "GeorgiaTech"),
            subjects_per_set: int = 20,
            images_per_subject: int = 6,
            train_images: int = 4,
            random_state: int = 42,
            georgia_crop_size: tuple[int, int] = (112, 112),
            crop_expansion: float = 0.15,
    ):
        self.root_dir = Path(root_dir)
        self.clean_root = Path(clean_root)
        self.datasets = datasets
        self.subjects_per_set = subjects_per_set
        self.images_per_subject = images_per_subject
        self.train_images = train_images
        self.random_state = random_state
        random.seed(random_state)

        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_path)
        self.gt_crop_size = georgia_crop_size
        self.crop_expansion = crop_expansion

        self._current_dataset: str | None = None

    # ─────────────────────────────── public ──────────────────────────── #
    def reduce_dataset(self) -> None:
        self._reset_clean_root()

        for ds_name in self.datasets:
            self._current_dataset = ds_name
            src_root = self.root_dir / ds_name
            folder_tag = self._DST_MAP[ds_name]

            subjects = self._pick_subject_folders(src_root)
            for person_idx, person_path in enumerate(subjects, 1):
                person_id = f"person_{person_idx:02d}"
                imgs = self._pick_images(person_path)
                train, test = imgs[: self.train_images], imgs[self.train_images:]

                self._copy_images(
                    train,
                    self.clean_root / "train" / folder_tag / person_id,
                )
                self._copy_images(
                    test,
                    self.clean_root / "test" / folder_tag / person_id,
                )

    # alias
    __call__ = reduce_dataset

    # ────────────────────────────── internals ────────────────────────── #
    def _reset_clean_root(self):
        if self.clean_root.exists():
            shutil.rmtree(self.clean_root)
        (self.clean_root / "train").mkdir(parents=True, exist_ok=True)
        (self.clean_root / "test").mkdir(parents=True, exist_ok=True)

    def _pick_subject_folders(self, dataset_root: Path) -> List[Path]:
        subs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
        if len(subs) < self.subjects_per_set:
            raise ValueError(f"Need {self.subjects_per_set}, found {len(subs)}")
        return random.sample(subs, self.subjects_per_set)

    def _pick_images(self, person_folder: Path) -> List[Path]:
        imgs = sorted(
            p for p in person_folder.iterdir() if p.suffix.lower() in {".pgm", ".jpg"}
        )
        if len(imgs) < self.images_per_subject:
            raise ValueError(f"{person_folder} has only {len(imgs)} images.")
        return imgs[: self.images_per_subject]

    # ---------------------------- copy + crop -------------------------- #
    def _copy_images(self, paths: Sequence[Path], dst_dir: Path):
        dst_dir.mkdir(parents=True, exist_ok=True)

        for src in paths:
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)

            if self._current_dataset == "GeorgiaTech":
                img = self._crop_georgia(img)

            if len(img.shape) == 2:  # ATnT safeguard (unlikely after IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(str(dst_dir / (src.stem + ".jpg")), img)

    def _crop_georgia(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) == 0:
            return cv2.resize(bgr, self.gt_crop_size)

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        exp = int(max(w, h) * self.crop_expansion)
        x0, y0 = max(x - exp, 0), max(y - exp, 0)
        x1, y1 = min(x + w + exp, bgr.shape[1]), min(y + h + exp, bgr.shape[0])
        face = bgr[y0:y1, x0:x1]
        return cv2.resize(face, self.gt_crop_size, cv2.INTER_LINEAR)


# ──────────────────────────────── CLI demo ───────────────────────────── #
if __name__ == "__main__":
    pre = FaceDatasetPreprocessor(
        root_dir="../../datasets/Original",
        clean_root="../../datasets/Processed",
        subjects_per_set=20,
        images_per_subject=6,
        train_images=4,
        random_state=2025,
    )
    pre()
    print("✓ Done.  New structure: Processed/{train,test}/{grayscale,RGB}/…")
