from pathlib import Path
import random
import shutil
import cv2  # Pillow works too – choose one in your environment
from typing import List, Sequence


class FaceDatasetPreprocessor:
    """
    A light, extensible pre-processing pipeline for the ATnT (PGM, gray-scale)
    and GeorgiaTech (JPEG, RGB) face datasets.

    First feature implemented:
        ➜ reduce_dataset()
            – keeps N subjects,
            – keeps K images per subject,
            – splits into train / test,
            – writes the result to a clean root directory.

    Later you can attach more features (resize, CLAHE, histogram
    equalisation, augmentation, etc.) as additional methods.
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

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def reduce_dataset(self) -> None:
        """Create the light-weight train/test split in self.clean_root."""
        self._make_empty_clean_root()

        for ds_name in self.datasets:
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
                    train_imgs, dst_root / "train" / person_id, convert_ext=".jpg"
                )
                # copy to test/
                self._copy_images(
                    test_imgs, dst_root / "test" / person_id, convert_ext=".jpg"
                )

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def _make_empty_clean_root(self) -> None:
        if self.clean_root.exists():
            shutil.rmtree(self.clean_root)
        self.clean_root.mkdir(parents=True, exist_ok=True)

    def _pick_subject_folders(self, dataset_root: Path) -> List[Path]:
        """Return self.subjects_per_set folders, sorted for reproducibility."""
        subfolders = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
        if len(subfolders) < self.subjects_per_set:
            raise ValueError(
                f"{dataset_root} has only {len(subfolders)} subjects, "
                f"expected ≥ {self.subjects_per_set}"
            )
        return random.sample(subfolders, self.subjects_per_set)

    def _pick_images(self, person_folder: Path) -> List[Path]:
        """
        Pick self.images_per_subject images for a single person.
        Orders deterministically (lexicographic) *before* sampling.
        """
        imgs = sorted(
            [p for p in person_folder.iterdir() if p.suffix.lower() in {".pgm", ".jpg"}]
        )
        if len(imgs) < self.images_per_subject:
            raise ValueError(
                f"{person_folder} has only {len(imgs)} images, "
                f"expected ≥ {self.images_per_subject}"
            )
        return imgs[: self.images_per_subject]  # → first 6; swap for random if desired

    def _copy_images(
            self, paths: Sequence[Path], target_dir: Path, convert_ext: str = ".jpg"
    ) -> None:
        """
        Copy (and convert if necessary) a list of images to *target_dir*.
        All output files are written as JPEG for consistency.
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        for src in paths:
            # read (supports both PGM & JPG)
            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)

            if img is None:
                raise IOError(f"Failed to read {src}")

            # ensure 3-channel for GeorgiaTech; leave gray as is
            if len(img.shape) == 2:  # grayscale → keep
                pass
            elif img.shape[2] == 3:  # already RGB/BGR
                pass
            else:
                raise RuntimeError(f"Unexpected image shape {img.shape} in {src}")

            # write to target
            dst_file = target_dir / (src.stem + convert_ext)
            cv2.imwrite(str(dst_file), img)

    # ------------------------------------------------------------------ #
    #  Convenience / CLI
    # ------------------------------------------------------------------ #
    def __call__(self):
        """Allows:  pre = FaceDatasetPreprocessor(...);  pre()"""
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
    pre()  # runs the reduction pipeline
    print("✓ Done. Reduced datasets saved in 'clean_datasets/'.")
