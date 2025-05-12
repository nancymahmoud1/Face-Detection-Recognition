import cv2
import numpy as np
from typing import List, Tuple, Optional
from .roc import PerformanceEvaluator, DetectionMetrics


class FaceDetector:
    def __init__(self, cascade_path: str):
        """
        Initialize the face detector with a Haar cascade classifier

        Args:
            cascade_path: Path to the Haar cascade XML file
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade classifier from {cascade_path}")

    def detect_faces(self,
                     image: np.ndarray,
                     scale_factor: float = 1.1,
                     min_neighbors: int = 5,
                     min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the input image using Haar cascade classifier

        Args:
            image: Input image (BGR format)
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_size: Minimum possible object size

        Returns:
            List of detected face rectangles (x, y, width, height)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return list(faces)  # Convert numpy array to list of tuples

    def draw_faces(self,
                   image: np.ndarray,
                   faces: List[Tuple[int, int, int, int]],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
        """
        Draw rectangles around detected faces

        Args:
            image: Image to draw on
            faces: List of face rectangles (x, y, width, height)
            color: BGR color tuple for rectangle
            thickness: Thickness of rectangle border

        Returns:
            Image with face rectangles drawn
        """
        image_copy = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
        return image_copy

    def evaluate_performance(self,
                           test_images: List[np.ndarray],
                           ground_truth: List[List[Tuple[int, int, int, int]]],
                           iou_threshold: float = 0.5) -> DetectionMetrics:
        """
        Evaluate face detector performance using precision, recall, and ROC curve

        Args:
            test_images: List of test images
            ground_truth: List of ground truth face rectangles for each image
            iou_threshold: IoU threshold for considering a detection as correct

        Returns:
            DetectionMetrics object containing performance metrics
        """
        all_predictions = []
        all_ground_truth = []
        
        for img, gt_faces in zip(test_images, ground_truth):
            # Get predictions
            pred_faces = self.detect_faces(img)
            all_predictions.extend(pred_faces)
            all_ground_truth.extend(gt_faces)
        
        return PerformanceEvaluator.evaluate_performance(
            all_predictions,
            all_ground_truth,
            iou_threshold
        )

    def plot_roc_curve(self, metrics: DetectionMetrics, save_path: Optional[str] = None):
        """
        Plot ROC curve using the evaluation metrics

        Args:
            metrics: DetectionMetrics object containing evaluation results
            save_path: Optional path to save the plot
        """
        PerformanceEvaluator.plot_roc_curve(metrics, save_path)