import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, List
import cv2

class FaceRecognitionPCA:
    def __init__(self, n_components: int = 100):
        """
        Initialize the Face Recognition PCA class.
        
        Args:
            n_components (int): Number of principal components to keep
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.mean_face = None
        self.eigenfaces = None
        self.coefficients = None
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image for PCA analysis.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Flattened and normalized image
        """
        # Convert to grayscale if image is in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Flatten the image
        flattened = image.flatten()
        
        # Normalize the image
        normalized = flattened / 255.0
        
        return normalized
    
    def fit(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform PCA analysis on the training images.
        
        Args:
            images (List[np.ndarray]): List of training images
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean face and eigenfaces
        """
        # Preprocess all images
        processed_images = np.array([self.preprocess_image(img) for img in images])
        
        # Perform PCA
        self.pca.fit(processed_images)
        
        # Store the mean face and eigenfaces
        self.mean_face = self.pca.mean_
        self.eigenfaces = self.pca.components_
        
        # Compute coefficients for all training images
        self.coefficients = self.pca.transform(processed_images)
        
        return self.mean_face, self.eigenfaces
    
    def get_coefficients(self, image: np.ndarray) -> np.ndarray:
        """
        Get the PCA coefficients for a single image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: PCA coefficients for the image
        """
        if self.pca is None:
            raise ValueError("PCA model has not been fitted yet. Call fit() first.")
            
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Get the coefficients
        coefficients = self.pca.transform(processed_image.reshape(1, -1))
        
        return coefficients.flatten()
