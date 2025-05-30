import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    def __init__(self, cascade_path: str):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade classifier from {cascade_path}")

    def detect_faces(self,
                     image: np.ndarray,
                     scale_factor: float = 1.1,
                     min_neighbors: int = 5,
                     min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return list(faces)

    def draw_faces(self,
                   image: np.ndarray,
                   faces: List[Tuple[int, int, int, int]],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 20) -> np.ndarray:
        image_copy = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
        return image_copy
