from pathlib import Path
import random
import shutil
import cv2
from typing import List, Sequence


class FaceDatasetPreprocessor:
    """
    Pre-process ATnT (PGM, grayscale) and GeorgiaTech (JPEG, RGB) datasets.

    Features implemented
    --------------------
    1) reduce_dataset()
       – keep `subjects_per_set` people
       – keep `images_per_subject` images per person
       – split into train / test (first `train_images` go to *train*)
       – optional face-crop **only for GeorgiaTech**   ⭐

    GeorgiaTech crop details
    ------------------------
    * Detector : OpenCV Haar cascade (`haarcascade_frontalface_default.xml`)
    * Strategy : take the largest detected face; expand box by `crop_expansion`
    * Alignment: kept simple (no rotation) – you can plug landmarks if needed.
    """

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
            crop_expansion: float = 0.15,  # enlarge bbox 15 %
    ):
        self.root_dir = Path(root_dir)
        self.clean_root = Path(clean_root)
        self.datasets = datasets
        self.subjects_per_set = subjects_per_set
        self.images_per_subject = images_per_subject
        self.train_images = train_images
        self.test_images = images_per_subject - train_images
        self.random_state = random_state
        random.seed(random_state)

        # ⭐ Haar cascade initialisation (one-time)
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_path)
        self.gt_crop_size = georgia_crop_size
        self.crop_expansion = crop_expansion

        # will be set inside the main loop
        self._current_dataset = None  # type: str | None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def reduce_dataset(self) -> None:
        """Create the light-weight train/test split in self.clean_root."""
        self._make_empty_clean_root()

        for ds_name in self.datasets:
            self._current_dataset = ds_name  # ⭐ tell helpers where we are

            src_root = self.root_dir / ds_name
            dst_root = self.clean_root / ds_name
            subjects = self._pick_subject_folders(src_root)

            for person_idx, person_path in enumerate(subjects, start=1):
                person_id = f"person_{person_idx:02d}"
                images = self._pick_images(person_path)
                train_imgs = images[: self.train_images]
                test_imgs = images[self.train_images:]

                # copy to train/
                self._copy_images(
                    train_imgs,
                    dst_root / "train" / person_id,
                    convert_ext=".jpg",
                )
                # copy to test/
                self._copy_images(
                    test_imgs,
                    dst_root / "test" / person_id,
                    convert_ext=".jpg",
                )

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def _make_empty_clean_root(self) -> None:
        if self.clean_root.exists():
            shutil.rmtree(self.clean_root)
        self.clean_root.mkdir(parents=True, exist_ok=True)

    def _pick_subject_folders(self, dataset_root: Path) -> List[Path]:
        subfolders = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
        if len(subfolders) < self.subjects_per_set:
            raise ValueError(
                f"{dataset_root} has only {len(subfolders)} subjects, "
                f"expected ≥ {self.subjects_per_set}"
            )
        return random.sample(subfolders, self.subjects_per_set)

    def _pick_images(self, person_folder: Path) -> List[Path]:
        imgs = sorted(
            [p for p in person_folder.iterdir() if p.suffix.lower() in {".pgm", ".jpg"}]
        )
        if len(imgs) < self.images_per_subject:
            raise ValueError(
                f"{person_folder} has only {len(imgs)} images, "
                f"expected ≥ {self.images_per_subject}"
            )
        return imgs[: self.images_per_subject]

    # -------------------------------------------------------------- ⭐ --- #
    #  copy / convert  (+ GeorgiaTech-only face crop)
    # -------------------------------------------------------------------- #
    def _copy_images(
            self,
            paths: Sequence[Path],
            target_dir: Path,
            convert_ext: str = ".jpg",
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)

        for src in paths:
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)  # always BGR
            if img is None:
                raise IOError(f"Failed to read {src}")

            if self._current_dataset == "GeorgiaTech":
                img = self._crop_face_georgia(img)

            # ATnT is grayscale; convert to 3-channel to keep file IO uniform
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            dst_file = target_dir / (src.stem + convert_ext)
            cv2.imwrite(str(dst_file), img)

    # ------------------------------------------------------------------ #
    #  GeorgiaTech face-crop helper          ⭐
    # ------------------------------------------------------------------ #
    def _crop_face_georgia(self, bgr_img):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60),
        )
        if len(faces) == 0:
            return cv2.resize(bgr_img, self.gt_crop_size)  # fallback: centre-crop later
        # pick largest box
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # expand bbox
        exp = int(max(w, h) * self.crop_expansion)
        x0 = max(x - exp, 0)
        y0 = max(y - exp, 0)
        x1 = min(x + w + exp, bgr_img.shape[1])
        y1 = min(y + h + exp, bgr_img.shape[0])

        face = bgr_img[y0:y1, x0:x1]
        return cv2.resize(face, self.gt_crop_size, interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------ #
    #  Convenience
    # ------------------------------------------------------------------ #
    def __call__(self):
        self.reduce_dataset()


# ---------------------------------------------------------------------- #
# Example usage
# ---------------------------------------------------------------------- #
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
    print("✓ Done. GeorgiaTech faces cropped; ATnT left intact.")
