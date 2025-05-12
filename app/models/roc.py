import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1_score: float
    average_precision: float
    false_positive_rate: np.ndarray
    true_positive_rate: np.ndarray
    thresholds: np.ndarray
    roc_auc: float


class PerformanceEvaluator:
    @staticmethod
    def evaluate_performance(predictions: List[Tuple[int, int, int, int]],
                           ground_truth: List[Tuple[int, int, int, int]],
                           iou_threshold: float = 0.5) -> DetectionMetrics:
        """
        Evaluate face detector performance using precision, recall, and ROC curve

        Args:
            predictions: List of predicted face rectangles (x, y, width, height)
            ground_truth: List of ground truth face rectangles (x, y, width, height)
            iou_threshold: IoU threshold for considering a detection as correct

        Returns:
            DetectionMetrics object containing performance metrics
        """
        all_predictions = []
        all_labels = []
        
        # Convert predictions and ground truth to binary labels
        # 1 if IoU > threshold, 0 otherwise
        for pred in predictions:
            max_iou = 0
            for gt in ground_truth:
                iou = PerformanceEvaluator._calculate_iou(pred, gt)
                max_iou = max(max_iou, iou)
            all_predictions.append(max_iou)
            all_labels.append(1 if max_iou > iou_threshold else 0)

        # Calculate precision and recall
        precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
        average_precision = average_precision_score(all_labels, all_predictions)
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1_score = np.max(f1_score)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
        roc_auc = auc(fpr, tpr)
        
        return DetectionMetrics(
            precision=np.mean(precision),
            recall=np.mean(recall),
            f1_score=f1_score,
            average_precision=average_precision,
            false_positive_rate=fpr,
            true_positive_rate=tpr,
            thresholds=thresholds,
            roc_auc=roc_auc
        )

    @staticmethod
    def plot_roc_curve(metrics: DetectionMetrics, save_path: Optional[str] = None):
        """
        Plot ROC curve using the evaluation metrics

        Args:
            metrics: DetectionMetrics object containing evaluation results
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        plt.plot(metrics.false_positive_rate, metrics.true_positive_rate,
                label=f'ROC curve (AUC = {metrics.roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def _calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes

        Args:
            box1: First bounding box (x, y, width, height)
            box2: Second bounding box (x, y, width, height)

        Returns:
            IoU score between 0 and 1
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area 