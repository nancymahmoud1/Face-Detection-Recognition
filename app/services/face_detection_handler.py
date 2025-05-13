import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
from app.models.face_detector import FaceDetector


class FaceDetectionService:
    def __init__(self):
        """Initialize the face detection service with available cascade classifiers"""
        cascade_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'haarCascades')
        self.cascade_files = {
            'face': os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml'),
            'profile': os.path.join(cascade_dir, 'haarcascade_profileface.xml'),
        }
        self.detectors = {}
        self._initialize_detectors()   # Initialize all detectors

    def _initialize_detectors(self):
        """Initialize all cascade classifiers"""
        failed_detectors = []
        for name, path in self.cascade_files.items():
            try:
                if not os.path.exists(path):
                    failed_detectors.append(name)
                    continue

                detector = FaceDetector(path)
                if detector.face_cascade.empty():
                    failed_detectors.append(name)
                    continue

                self.detectors[name] = detector  # Store successful detector

            except Exception as e:
                failed_detectors.append(name)

        if failed_detectors:
            pass

        if not self.detectors:
            raise RuntimeError("No cascade classifiers could be loaded. Please check OpenCV installation.")

    def detect_faces(self,
                     image: np.ndarray,
                     detector_type: str = 'face',
                     scale_factor: float = 1.1, # How much to reduce image size at each scale
                     min_neighbors: int = 5,
                     min_size: Tuple[int, int] = (30, 30)) -> Tuple[
        Optional[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Detect faces or other features in the image

        Args:
            image: Input image (BGR format)
            scale_factor: how much the image size is reduced at each image scale(1.1 = 10% reduction)
            min_neighbors: how many neighbors each candidate rectangle should have
            min_size: Minimum object size to detect

        Returns:
            Tuple of (processed image with rectangles, list of detected rectangles)

        Raises:
            ValueError: If detector_type is not available
            RuntimeError: If image processing fails
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a BGR color image")

        if detector_type not in self.detectors:
            available = list(self.detectors.keys())
            raise ValueError(f"Detector type '{detector_type}' not available. "
                             f"Available types: {available}")

        try:
            # Get the appropriate detector
            detector = self.detectors[detector_type]

            # Detect features
            rectangles = detector.detect_faces(
                image,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
                min_size=min_size
            )

            # Draw rectangles on a copy of the image
            result_image = detector.draw_faces(image, rectangles)

            return result_image, rectangles

        except cv2.error as e:
            raise RuntimeError(f"Image processing failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Detection failed: {str(e)}")

