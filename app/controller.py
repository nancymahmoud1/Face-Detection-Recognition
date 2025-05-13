import numpy as np
from PyQt5 import QtWidgets, QtCore

# Core utility and services
from app.utils.clean_cache import remove_directories
from app.services.image_service import ImageServices
from app.services.face_detection_handler import FaceDetectionService

# Main GUI design
from app.design.main_layout import Ui_MainWindow

# Image processing functionality
import cv2
import glob
import os
import random
from app.services.face_recognition_service import FaceRecognitionService
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel


class MainWindowController:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.MainWindow = QtWidgets.QMainWindow()

        self.path = None
        self.original_image = None
        self.processed_image = None

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        # Initialize services
        self.srv = ImageServices()
        self.face_service = FaceDetectionService()

        # Connect signals to slots
        self.setup_connections()

        # Setup face detection controls
        self.setup_face_detection_controls()

    def show_error(self, message: str):
        """Show error message dialog"""
        QtWidgets.QMessageBox.critical(self.MainWindow, "Error", message)

    def show_info(self, message: str):
        """Show information message dialog"""
        QtWidgets.QMessageBox.information(self.MainWindow, "Information", message)

    def setup_face_detection_controls(self):
        """Setup face detection controls and their connections"""

        # Connect slider value changes to update labels
        self.ui.scale_factor_slider.valueChanged.connect(self.update_scale_factor_label)
        self.ui.min_neighbors_slider.valueChanged.connect(self.update_min_neighbors_label)
        self.ui.min_size_slider.valueChanged.connect(self.update_min_size_label)

        # Connect detect faces button
        self.ui.detect_faces_btn.clicked.connect(self.detect_faces)

        # Set initial values
        self.update_scale_factor_label(self.ui.scale_factor_slider.value())
        self.update_min_neighbors_label(self.ui.min_neighbors_slider.value())
        self.update_min_size_label(self.ui.min_size_slider.value())

    def setup_connections(self):
        """Setup all button and control connections"""
        # Basic UI connections
        self.ui.upload_button.clicked.connect(self.upload_image)
        self.ui.reset_image_button.clicked.connect(self.reset_image)
        self.ui.save_image_button.clicked.connect(self.save_image)
        self.ui.clear_image_button.clicked.connect(self.clear_image)
        self.ui.quit_app_button.clicked.connect(self.quit_application)

        # Face detection and recognition connections
        self.ui.recognize_faces_btn.clicked.connect(self.recognize_and_retrieve)
        # self.ui.dataset_combo.currentIndexChanged.connect(self.on_dataset_changed)
        self.ui.color_mode_combo.currentIndexChanged.connect(self.on_color_mode_changed)

        # Setup face detection controls
        self.setup_face_detection_controls()

    def update_scale_factor_label(self, value):
        """Update the scale factor label with the current slider value"""
        scale_factor = value / 10.0  # Convert slider value (11-20) to scale factor (1.1-2.0)
        self.ui.scale_factor_label.setText(f"Scale Factor: {scale_factor:.1f}")

    def update_min_neighbors_label(self, value):
        """Update the min neighbors label with the current slider value"""
        self.ui.min_neighbors_label.setText(f"Min Neighbors: {value}")

    def update_min_size_label(self, value):
        """Update the min size label with the current slider value"""
        self.ui.min_size_label.setText(f"Min Size: {value}")

    def detect_faces(self):
        """Detect faces in the current image using the selected detector and parameters"""
        if not hasattr(self, 'original_image'):
            self.show_error("Please upload an image first")
            return

        try:
            # Get parameters from UI
            # detector_type = self.ui.detector_type_combo.currentText()
            scale_factor = self.ui.scale_factor_slider.value() / 10.0  # Convert to float
            min_neighbors = self.ui.min_neighbors_slider.value()
            min_size = (self.ui.min_size_slider.value(), self.ui.min_size_slider.value())

            # Detect faces using the face service
            self.processed_image, faces = self.face_service.detect_faces(
                self.original_image,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
                min_size=min_size
            )

            # Update the processed image
            self.srv.clear_image(self.ui.processed_groupBox)
            self.srv.set_image_in_groupbox(self.ui.processed_groupBox, self.processed_image)

            # Show detection results
            if faces:
                #self.show_info(f"Found {len(faces)} faces")
                self.update_info_text(f"Found {len(faces)} faces")
            else:
                self.show_info("No faces detected")

        except Exception as e:
            self.show_error(f"An error occurred during face detection: {str(e)}")

    def upload_image(self):
        self.path = self.srv.upload_image_file()

        if not self.path:
            return

        self.original_image = cv2.imread(self.path)
        if self.original_image is None:
            return

        self.processed_image = self.original_image.copy()

        # Clear any existing images displayed in the group boxes
        self.srv.clear_image(self.ui.original_groupBox)
        self.srv.clear_image(self.ui.processed_groupBox)

        # Display the images in their respective group boxes
        self.srv.set_image_in_groupbox(self.ui.original_groupBox, self.original_image)
        self.srv.set_image_in_groupbox(self.ui.processed_groupBox, self.processed_image)

        # Show the group boxes if they're hidden
        self.ui.original_groupBox.show()
        self.ui.processed_groupBox.show()

    def reset_image(self):
        if self.original_image is None:
            return

        self.processed_image = self.original_image.copy()
        self.srv.clear_image(self.ui.processed_groupBox)
        self.srv.set_image_in_groupbox(self.ui.processed_groupBox, self.original_image)

    def save_image(self):
        if self.processed_image is None:
            # print("Error: Processed image is None.")
            return

        self.srv.save_image(self.processed_image)

    def clear_image(self):
        if self.original_image is None:
            return

        self.srv.clear_image(self.ui.processed_groupBox)
        self.srv.clear_image(self.ui.original_groupBox)
        self.ui.info_textbox.clear()
        self.original_image = None
        self.processed_image = None

    def quit_application(self):
        """Close the application."""
        remove_directories()
        self.app.quit()

    def recognize_and_retrieve(self):
        if not self.path:
            self.show_error("Please upload an image first.")
            return

        # Recognize identity
        result = FaceRecognitionService.recognize_face(self.path)
        if "error" in result:
            self.show_error(result["error"])
            return

        # Determine color mode from path (RGB or grayscale)
        if "/RGB/" in self.path or os.path.normpath(self.path).split(os.sep)[-3].lower() == "rgb":
            color_mode = "RGB"
        else:
            color_mode = "grayscale"

        # Get predicted label (e.g., person_05)
        person_folder = result["name"]
        train_dir = f"datasets/Processed/train/{color_mode}/{person_folder}"

        # Pick a matching training image
        matching_images = glob.glob(os.path.join(train_dir, "*.jpg"))
        sample_train_image = random.choice(matching_images) if matching_images else None

        if not sample_train_image:
            self.show_error(f"No training image found for {person_folder} in {color_mode} mode.")
            return

        # Show result info
        #self.show_info(f"Identified as: {person_folder}\nDistance: {result['distance']:.2f}\nMode: {color_mode}")
        self.update_info_text(f"Identified as: {person_folder}\nDistance: {result['distance']:.2f}\nMode: {color_mode}")
        # Display matched image
        matched_img = cv2.imread(sample_train_image)
        if matched_img is not None:
            self.srv.clear_image(self.ui.processed_groupBox)
            self.srv.set_image_in_groupbox(self.ui.processed_groupBox, matched_img)

    def update_info_text(self, message):
        """Updates the info text box with the given message.

        Args:
            message (str): The message to display in the text box
        """
        # Append the message to the text box
        self.ui.info_textbox.clear()
        self.ui.info_textbox.append(message)

        # Ensure the text box scrolls to show the latest message
        self.ui.info_textbox.verticalScrollBar().setValue(
            self.ui.info_textbox.verticalScrollBar().maximum()
        )

    def on_dataset_changed(self, index):
        """Handle dataset selection change"""
        # TODO: Implement dataset change handling
        pass

    def on_color_mode_changed(self, index):
        """Handle color mode change"""
        # TODO: Implement color mode change handling
        pass

    def run(self):
        """Start the application and show the main window"""
        self.MainWindow.showFullScreen()
        return self.app.exec_()
