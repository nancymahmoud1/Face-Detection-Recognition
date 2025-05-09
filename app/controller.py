import numpy as np
from PyQt5 import QtWidgets, QtCore

# Core utility and services
from app.utils.clean_cache import remove_directories
from app.services.image_service import ImageServices

# Main GUI design
from app.design.main_layout import Ui_MainWindow

# Image processing functionality
import cv2


class MainWindowController:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.MainWindow = QtWidgets.QMainWindow()

        self.path = None
        self.path_1 = None
        self.path_2 = None

        self.original_image = None
        self.processed_image = None

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        self.srv = ImageServices()

        # Connect signals to slots
        self.setup_connections()

    def run(self):
        """Run the application."""
        self.MainWindow.showFullScreen()
        self.app.exec_()

    def setup_connections(self):
        """Setup all button and control connections"""
        # Basic UI connections
        self.ui.upload_button.clicked.connect(self.upload_image)
        self.ui.reset_image_button.clicked.connect(self.reset_image)
        self.ui.save_image_button.clicked.connect(self.save_image)
        self.ui.clear_image_button.clicked.connect(self.clear_image)
        self.ui.quit_app_button.clicked.connect(self.quit_application)

        # Face detection and recognition connections
        self.ui.detect_faces_btn.clicked.connect(self.detect_faces)
        self.ui.recognize_faces_btn.clicked.connect(self.recognize_faces)
        self.ui.eigen_components_slider.valueChanged.connect(self.update_eigen_components)
        self.ui.dataset_combo.currentIndexChanged.connect(self.on_dataset_changed)
        self.ui.color_mode_combo.currentIndexChanged.connect(self.on_color_mode_changed)

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
            print("Error: Processed image is None.")
            return

        self.srv.save_image(self.processed_image)

    def clear_image(self):
        if self.original_image is None:
            return

        self.srv.clear_image(self.ui.processed_groupBox)
        self.srv.clear_image(self.ui.original_groupBox)

    def quit_application(self):
        """Close the application."""
        remove_directories()
        self.app.quit()

    def detect_faces(self):
        """Handle face detection"""
        if not hasattr(self, 'current_image'):
            self.show_error("Please upload an image first")
            return
        # TODO: Implement face detection
        pass

    def train_pca_model(self):
        """Handle PCA model training"""
        if self.ui.dataset_combo.currentText() == "Custom Dataset":
            self.show_error("Please select a standard dataset for training")
            return
        # TODO: Implement PCA model training
        pass

    def recognize_faces(self):
        """Handle face recognition"""
        if not hasattr(self, 'current_image'):
            self.show_error("Please upload an image first")
            return
        # TODO: Implement face recognition
        pass

    def update_eigen_components(self, value):
        """Update the eigen components label"""
        self.ui.eigen_components_label.setText(f"Eigen Components: {value}")

    def on_dataset_changed(self, index):
        """Handle dataset selection change"""
        # TODO: Implement dataset loading
        pass

    def on_color_mode_changed(self, index):
        """Handle color mode change"""
        # TODO: Implement color mode switching
        pass

    def show_error(self, message):
        """Show error message to user"""
        QtWidgets.QMessageBox.critical(self, "Error", message)
