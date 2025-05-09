from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize

from app.design.tools.gui_utilities import GUIUtilities


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.screen_size = QtWidgets.QApplication.primaryScreen().size()
        MainWindow.resize(self.screen_size.width(), self.screen_size.height())

        # 1) Define style variables
        self.setupStyles()
        self.util = GUIUtilities()

        # 2) Apply main window style
        MainWindow.setStyleSheet(self.main_window_style)

        # 3) Central widget & main layout
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.main_vertical_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.main_vertical_layout.setSpacing(0)

        # 4) Create Title (icon + label) and Navbar (upload, reset, save, quit)
        self.setupTitleArea()
        self.setupNavbar()
        self.combineTitleAndNavbar()

        # Add the top bar (title+navbar) to the main vertical layout
        self.main_vertical_layout.addLayout(self.title_nav_layout)

        # 5) Create the main content: left sidebar + two group boxes on the right
        self.setupMainContent()
        self.main_vertical_layout.addLayout(self.main_content_layout)

        # 6) Finalize the main window
        MainWindow.setCentralWidget(self.centralwidget)

        # Menu bar & status bar (if needed)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.original_groupBox.show()
        self.processed_groupBox.show()

        # 7) Set window title, etc.
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # ----------------------------------------------------------------------
    # Styles
    # ----------------------------------------------------------------------
    def setupStyles(self):
        """Holds all style sheets in one place for easier modification."""
        # Main window: deep blue gradient with floating bubble effect
        self.main_window_style = """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a2342, stop:1 #1e3c72);
                background-image:
                    radial-gradient(circle 180px at 20% 30%, rgba(0,212,255,0.18) 0%, rgba(0,212,255,0) 80%),
                    radial-gradient(circle 120px at 80% 60%, rgba(255,0,255,0.12) 0%, rgba(255,0,255,0) 80%),
                    radial-gradient(circle 90px at 60% 20%, rgba(0,255,255,0.13) 0%, rgba(0,255,255,0) 80%),
                    radial-gradient(circle 140px at 40% 80%, rgba(0,255,255,0.10) 0%, rgba(0,255,255,0) 80%);
            }
        """

        # Group boxes: lighter, bubble-inspired gradient, glowing border
        self.groupbox_style = """
            QGroupBox {
                color: #e0f7fa;
                border: 2px solid #a084ee;
                border-radius: 22px;
                margin-top: 16px;
                font-weight: bold;
                padding-top: 16px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0,234,255,0.25), stop:0.5 rgba(160,132,238,0.18), stop:1 rgba(0,180,216,0.18));
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #00eaff;
                font-size: 17px;
                font-weight: bold;
                letter-spacing: 1px;
            }
        """

        # Buttons: bubble-inspired gradient, glossy, glowing border
        self.button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00eaff, stop:0.5 #a084ee, stop:1 #00b4d8);
                color: #fff;
                border: 2px solid #a084ee;
                border-radius: 18px;
                font-size: 15px;
                font-weight: bold;
                padding: 12px 28px;
                margin: 4px 2px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #a084ee, stop:1 #00eaff);
                color: #fff;
                border: 2px solid #00eaff;
            }
            QPushButton:pressed {
                background: #0a2342;
                color: #e0f7fa;
                border: 2px solid #00eaff;
            }
        """

        # Quit button: bubble pink glow
        self.quit_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffb6ea, stop:1 #ff5252);
                color: #fff;
                border: 2px solid #ffb6ea;
                border-radius: 18px;
                font-weight: bold;
                min-width: 40px;
                min-height: 40px;
                max-width: 40px;
                max-height: 40px;
            }
            QPushButton:hover {
                background: #ff5252;
                color: #fff;
            }
            QPushButton:pressed {
                background: #b71c1c;
                color: #fff;
            }
        """

        # Sliders: bubble-inspired groove and handle
        self.slider_style = """
            QSlider::groove:horizontal {
                border: 1px solid #a084ee;
                height: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #a084ee44, stop:1 #00eaff33);
                margin: 2px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.7, fx:0.5, fy:0.5, stop:0 #fff, stop:1 #a084ee);
                width: 28px;
                margin: -10px 0;
                border: 2px solid #00eaff;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00eaff, stop:1 #a084ee);
                border: 1px solid #00eaff;
                height: 12px;
                border-radius: 6px;
            }
        """

        # Labels: modern, bold, bubble color
        self.label_style = """
            QLabel {
                color: #e0f7fa;
                font-size: 15px;
                padding: 8px;
                font-weight: bold;
                letter-spacing: 1px;
            }
        """

        self.back_icon_path = "static/icons/Back-icon.png"

    # ----------------------------------------------------------------------
    # Title + Navbar
    # ----------------------------------------------------------------------
    def setupTitleArea(self):
        """Creates the title icon & label in a horizontal layout."""
        self.title_icon = QtWidgets.QLabel()
        self.title_icon.setMaximumSize(QtCore.QSize(80, 80))
        self.title_icon.setPixmap(QtGui.QPixmap("static/icons/icon.png"))
        self.title_icon.setScaledContents(True)
        self.title_icon.setObjectName("title_icon")

        self.title_label = self.util.createLabel(
            text="Face Recognition",
            style="color:white; padding:10px; padding-left:0; font-size: 32px; font-weight: bold",
            isHead=True
        )
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(32)
        font.setBold(True)
        self.title_label.setFont(font)

        # Vertical layout for title and subtitle
        title_layout = QtWidgets.QVBoxLayout()
        title_layout.addWidget(self.title_label)
        title_layout.setSpacing(0)

        # Horizontal layout for icon + title group
        self.title_layout = QtWidgets.QHBoxLayout()
        self.title_layout.addWidget(self.title_icon)
        self.title_layout.addLayout(title_layout)

    def setupNavbar(self):
        """Creates the Upload, Reset, Save, and Quit buttons."""
        self.upload_button = self.util.createButton("📁 Upload Image", self.button_style)
        self.reset_image_button = self.util.createButton("🔄 Reset", self.button_style)
        self.save_image_button = self.util.createButton("💾 Save", self.button_style)
        self.clear_image_button = self.util.createButton("🗑️ Clear", self.button_style)

        self.quit_app_button = self.util.createButton("X", self.quit_button_style)
        self.util.adjust_quit_button(self.quit_app_button)

        self.navbar_layout = QtWidgets.QHBoxLayout()
        self.navbar_layout.setSpacing(15)
        self.navbar_layout.addWidget(self.upload_button)
        self.navbar_layout.addWidget(self.reset_image_button)
        self.navbar_layout.addWidget(self.save_image_button)
        self.navbar_layout.addWidget(self.clear_image_button)
        self.navbar_layout.addWidget(self.quit_app_button)

    def combineTitleAndNavbar(self):
        """Combines the title & navbar in one horizontal layout."""
        self.title_nav_layout = QtWidgets.QHBoxLayout()
        self.title_nav_layout.addLayout(self.title_layout)
        self.title_nav_layout.addStretch(1)
        self.title_nav_layout.addLayout(self.navbar_layout)

    # ----------------------------------------------------------------------
    # Main Content: Sidebar + GroupBoxes
    # ----------------------------------------------------------------------
    def setupMainContent(self):
        """Creates the main content area with sidebar and image displays."""
        self.main_content_layout = QtWidgets.QHBoxLayout()
        self.main_content_layout.setContentsMargins(20, 20, 20, 20)
        self.main_content_layout.setSpacing(20)

        # Left sidebar with controls
        self.sidebar = QtWidgets.QWidget()
        self.sidebar.setMaximumWidth(300)
        self.sidebar.setMinimumWidth(250)
        self.sidebar.setStyleSheet("""
            QWidget {
                background: rgba(52, 152, 219, 0.1);
                border-radius: 16px;
                border: 2px solid #3498db;
            }
        """)
        
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(20, 20, 20, 20)
        self.sidebar_layout.setSpacing(20)

        # Add face detection and recognition controls
        self.setupFaceDetectionControls()
        
        # Right side with image displays
        self.right_side = QtWidgets.QWidget()
        self.right_side_layout = QtWidgets.QHBoxLayout(self.right_side)
        self.right_side_layout.setContentsMargins(0, 0, 0, 0)
        self.right_side_layout.setSpacing(20)

        # Setup image display group boxes
        self.setupImageGroupBoxes()
        
        # Add widgets to layouts
        self.main_content_layout.addWidget(self.sidebar)
        self.main_content_layout.addWidget(self.right_side)

    def setupFaceDetectionControls(self):
        """Creates controls for face detection and recognition."""
        # Dataset Selection Group
        dataset_group = QtWidgets.QGroupBox("Dataset Selection")
        dataset_group.setStyleSheet(self.groupbox_style)
        dataset_layout = QtWidgets.QVBoxLayout()
        
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.addItems([
            "AT&T Face Database",
            "Yale Face Database",
            "FERET Database",
            "Custom Dataset"
        ])
        self.dataset_combo.setStyleSheet("""
            QComboBox {
                background: transparent;
                color: white;
                border: 1px solid #00B4D8;
                border-radius: 8px;
                padding: 8px;
                min-height: 30px;
            }
            QComboBox:hover {
                border: 1px solid #00B4D8;
                background: rgba(0,180,216,0.08);
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: url(static/icons/down-arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background: rgba(10,35,66,0.85);
                color: white;
                border: 1px solid #00B4D8;
                selection-background-color: #00B4D8;
                selection-color: white;
                border-radius: 8px;
            }
        """)
        dataset_layout.addWidget(self.dataset_combo)
        dataset_group.setLayout(dataset_layout)

        # Face Detection Group
        detection_group = QtWidgets.QGroupBox("Face Detection")
        detection_group.setStyleSheet(self.groupbox_style)
        detection_layout = QtWidgets.QVBoxLayout()
        
        self.detect_faces_btn = QtWidgets.QPushButton("Detect Faces")
        self.detect_faces_btn.setStyleSheet(self.button_style)
        detection_layout.addWidget(self.detect_faces_btn)
        
        self.color_mode_combo = QtWidgets.QComboBox()
        self.color_mode_combo.addItems(["Color", "Grayscale"])
        self.color_mode_combo.setStyleSheet(self.dataset_combo.styleSheet())
        detection_layout.addWidget(self.color_mode_combo)
        detection_group.setLayout(detection_layout)

        # Face Recognition Group
        recognition_group = QtWidgets.QGroupBox("Face Recognition")
        recognition_group.setStyleSheet(self.groupbox_style)
        recognition_layout = QtWidgets.QVBoxLayout()
        

        self.recognize_faces_btn = QtWidgets.QPushButton("Recognize Faces")
        self.recognize_faces_btn.setStyleSheet(self.button_style)
        recognition_layout.addWidget(self.recognize_faces_btn)
        
        self.eigen_components_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.eigen_components_slider.setMinimum(1)
        self.eigen_components_slider.setMaximum(100)
        self.eigen_components_slider.setValue(50)
        self.eigen_components_slider.setStyleSheet(self.slider_style)
        
        self.eigen_components_label = QtWidgets.QLabel("Eigen Components: 50")
        self.eigen_components_label.setStyleSheet(self.label_style)
        
        recognition_layout.addWidget(self.eigen_components_label)
        recognition_layout.addWidget(self.eigen_components_slider)
        recognition_group.setLayout(recognition_layout)

        # Add all groups to sidebar
        self.sidebar_layout.addWidget(dataset_group)
        self.sidebar_layout.addWidget(detection_group)
        self.sidebar_layout.addWidget(recognition_group)
        self.sidebar_layout.addStretch()

    def setupImageGroupBoxes(self):
        """Creates the original and processed image group boxes."""
        # Original image group box
        self.original_groupBox = QtWidgets.QGroupBox("Original Image")
        self.original_groupBox.setStyleSheet(self.groupbox_style)
        self.original_layout = QtWidgets.QVBoxLayout(self.original_groupBox)
        self.original_layout.setContentsMargins(15, 25, 15, 15)
        self.original_layout.setSpacing(10)
        
        self.original_image_label = QtWidgets.QLabel()
        self.original_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_image_label.setStyleSheet("""
            QLabel {
                background: rgba(44, 62, 80, 0.3);
                border-radius: 8px;
                padding: 10px;
            }
        """)
        self.original_layout.addWidget(self.original_image_label)

        # Processed image group box
        self.processed_groupBox = QtWidgets.QGroupBox("Processed Image")
        self.processed_groupBox.setStyleSheet(self.groupbox_style)
        self.processed_layout = QtWidgets.QVBoxLayout(self.processed_groupBox)
        self.processed_layout.setContentsMargins(15, 25, 15, 15)
        self.processed_layout.setSpacing(10)
        
        self.processed_image_label = QtWidgets.QLabel()
        self.processed_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.processed_image_label.setStyleSheet("""
            QLabel {
                background: rgba(44, 62, 80, 0.3);
                border-radius: 8px;
                padding: 10px;
            }
        """)
        self.processed_layout.addWidget(self.processed_image_label)

        # Add group boxes to the right side layout
        self.right_side_layout.addWidget(self.original_groupBox)
        self.right_side_layout.addWidget(self.processed_groupBox)

    # ----------------------------------------------------------------------
    # Retranslate
    # ----------------------------------------------------------------------
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Face Recognition System"))

    # ----------------------------------------------------------------------
    #  Show/Hide Logic
    # ----------------------------------------------------------------------
    def show_main_buttons(self):
        """
        Shows the original and processed group boxes.
        """
        # Show the original and processed group boxes
        self.original_groupBox.show()
        self.processed_groupBox.show()
