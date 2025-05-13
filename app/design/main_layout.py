from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize

from app.design.tools.gui_utilities import GUIUtilities


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Get screen dimensions and calculate base sizes
        self.screen = QtWidgets.QApplication.primaryScreen()
        self.screen_geometry = self.screen.availableGeometry()
        self.base_width = self.screen_geometry.width()
        self.base_height = self.screen_geometry.height()

        # Calculate dynamic sizes
        self.window_width = int(self.base_width * 0.9)
        self.window_height = int(self.base_height * 0.85)
        self.min_width = int(self.base_width * 0.5)
        self.min_height = int(self.base_height * 0.5)

        # Font sizes based on screen dimensions
        self.title_font_size = max(24, int(self.base_height * 0.035))  # 3.5% of screen height
        self.label_font_size = max(12, int(self.base_height * 0.018))  # 1.8% of screen height
        self.button_font_size = max(14, int(self.base_height * 0.02))  # 2% of screen height

        # Set window properties
        MainWindow.resize(self.window_width, self.window_height)
        MainWindow.setMinimumSize(self.min_width, self.min_height)
        MainWindow.setMaximumSize(self.base_width, self.base_height)

        # 1) Define style variables with dynamic sizes
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
        self.menubar.setGeometry(QtCore.QRect(0, 0, self.window_width, int(self.base_height * 0.03)))
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
    # Styles with dynamic sizing
    # ----------------------------------------------------------------------
    def setupStyles(self):
        """Holds all style sheets with dynamic sizing based on screen dimensions."""
        # Calculate dynamic sizes
        button_padding = f"{int(self.base_height * 0.015)}px {int(self.base_width * 0.01)}px"
        button_radius = int(self.base_height * 0.02)
        slider_handle_size = int(self.base_height * 0.03)
        slider_groove_height = int(self.base_height * 0.01)
        groupbox_radius = int(self.base_height * 0.025)
        groupbox_title_size = max(14, int(self.base_height * 0.02))

        # Main window style remains the same
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

        # Group boxes with dynamic sizing
        self.groupbox_style = f"""
            QGroupBox {{
                color: #e0f7fa;
                border: 2px solid #a084ee;
                border-radius: {groupbox_radius}px;
                margin-top: {int(self.base_height * 0.02)}px;
                font-weight: bold;
                padding-top: {int(self.base_height * 0.02)}px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0,234,255,0.25), stop:0.5 rgba(160,132,238,0.18), stop:1 rgba(0,180,216,0.18));
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {int(self.base_width * 0.01)}px;
                padding: 0 {int(self.base_width * 0.008)}px;
                color: #00eaff;
                font-size: {groupbox_title_size}px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
        """

        # Buttons with dynamic sizing
        self.button_style = f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00eaff, stop:0.5 #a084ee, stop:1 #00b4d8);
                color: #fff;
                border: 2px solid #a084ee;
                border-radius: {button_radius}px;
                font-size: {self.button_font_size}px;
                font-weight: bold;
                padding: {button_padding};
                margin: {int(self.base_height * 0.005)}px {int(self.base_width * 0.002)}px;
                letter-spacing: 1px;
                min-width: {int(self.base_width * 0.1)}px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #a084ee, stop:1 #00eaff);
                color: #fff;
                border: 2px solid #00eaff;
            }}
            QPushButton:pressed {{
                background: #0a2342;
                color: #e0f7fa;
                border: 2px solid #00eaff;
            }}
        """

        # Quit button with dynamic sizing
        quit_btn_size = int(self.base_height * 0.04)
        self.quit_button_style = f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffb6ea, stop:1 #ff5252);
                color: #fff;
                border: 2px solid #ffb6ea;
                border-radius: {quit_btn_size}px;
                font-weight: bold;
                min-width: {quit_btn_size}px;
                min-height: {quit_btn_size}px;
                max-width: {quit_btn_size}px;
                max-height: {quit_btn_size}px;
                font-size: {int(self.button_font_size * 0.8)}px;
            }}
            QPushButton:hover {{
                background: #ff5252;
                color: #fff;
            }}
            QPushButton:pressed {{
                background: #b71c1c;
                color: #fff;
            }}
        """

        # Sliders with dynamic sizing
        self.slider_style = f"""
            QSlider::groove:horizontal {{
                border: 1px solid #4a4e69;
                height: {slider_groove_height}px;
                background: rgba(74, 78, 105, 0.3);
                margin: {int(slider_groove_height * 0.25)}px 0;
                border-radius: {int(slider_groove_height * 0.5)}px;
            }}
            QSlider::handle:horizontal {{
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.7, 
                    fx:0.5, fy:0.5, stop:0 #ffffff, stop:1 #4cc9f0);
                width: {slider_handle_size}px;
                height: {slider_handle_size}px;
                margin: -{int(slider_handle_size * 0.33)}px 0;
                border: 2px solid #4cc9f0;
                border-radius: {int(slider_handle_size * 0.5)}px;
            }}
            QSlider::sub-page:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4cc9f0, stop:1 #7209b7);
                border: 1px solid #4cc9f0;
                height: {slider_groove_height}px;
                border-radius: {int(slider_groove_height * 0.5)}px;
            }}
        """

        # Labels with dynamic font size
        self.label_style = f"""
            QLabel {{
                color: #e0f7fa;
                font-size: {self.label_font_size}px;
                padding: {int(self.base_height * 0.01)}px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
        """

        # Text box with dynamic sizing
        textbox_padding = int(self.base_height * 0.01)
        self.textbox_style = f"""
            QTextEdit {{
                background: rgba(10, 35, 66, 0.7);
                border: 1px solid #4cc9f0;
                border-radius: {int(self.base_height * 0.01)}px;
                color: #ffffff;
                padding: {textbox_padding}px;
                font-size: {int(self.label_font_size)}px;
                selection-background-color: #7209b7;
                selection-color: white;
            }}
            QTextEdit:focus {{
                border: 2px solid #7209b7;
            }}
        """

        self.back_icon_path = "static/icons/Back-icon.png"

    # ----------------------------------------------------------------------
    # Title + Navbar with dynamic sizing
    # ----------------------------------------------------------------------
    def setupTitleArea(self):
        """Creates the title icon & label with dynamic sizing."""
        title_icon_size = int(self.base_height * 0.08)
        self.title_icon = QtWidgets.QLabel()
        self.title_icon.setMaximumSize(QtCore.QSize(title_icon_size, title_icon_size))
        self.title_icon.setPixmap(QtGui.QPixmap("static/icons/icon.png"))
        self.title_icon.setScaledContents(True)
        self.title_icon.setObjectName("title_icon")

        self.title_label = self.util.createLabel(
            text="Face Recognition",
            style=f"color:white; padding:{int(self.base_height * 0.01)}px; padding-left:0; font-size: {self.title_font_size}px; font-weight: bold",
            isHead=True
        )
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(self.title_font_size)
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
        self.title_layout.setSpacing(int(self.base_width * 0.01))

    def setupNavbar(self):
        """Creates the navigation buttons with dynamic sizing."""
        btn_spacing = int(self.base_width * 0.01)
        self.upload_button = self.util.createButton("📁 Upload Image", self.button_style)
        self.reset_image_button = self.util.createButton("🔄 Reset", self.button_style)
        self.save_image_button = self.util.createButton("💾 Save", self.button_style)
        self.clear_image_button = self.util.createButton("🗑️ Clear", self.button_style)

        self.quit_app_button = self.util.createButton("X", self.quit_button_style)
        self.util.adjust_quit_button(self.quit_app_button)

        self.navbar_layout = QtWidgets.QHBoxLayout()
        self.navbar_layout.setSpacing(btn_spacing)
        self.navbar_layout.addWidget(self.upload_button)
        self.navbar_layout.addWidget(self.reset_image_button)
        self.navbar_layout.addWidget(self.save_image_button)
        self.navbar_layout.addWidget(self.clear_image_button)
        self.navbar_layout.addWidget(self.quit_app_button)

    def combineTitleAndNavbar(self):
        """Combines the title & navbar with proper spacing."""
        self.title_nav_layout = QtWidgets.QHBoxLayout()
        self.title_nav_layout.addLayout(self.title_layout)
        self.title_nav_layout.addStretch(1)
        self.title_nav_layout.addLayout(self.navbar_layout)
        self.title_nav_layout.setContentsMargins(
            int(self.base_width * 0.02),  # left
            int(self.base_height * 0.01),  # top
            int(self.base_width * 0.02),  # right
            int(self.base_height * 0.01)  # bottom
        )

    # ----------------------------------------------------------------------
    # Main Content with dynamic sizing
    # ----------------------------------------------------------------------
    def setupMainContent(self):
        """Creates the main content area with dynamic sizing."""
        main_margin = int(self.base_width * 0.02)
        main_spacing = int(self.base_width * 0.02)

        self.main_content_layout = QtWidgets.QHBoxLayout()
        self.main_content_layout.setContentsMargins(main_margin, main_margin, main_margin, main_margin)
        self.main_content_layout.setSpacing(main_spacing)

        # Left sidebar with dynamic width
        sidebar_min = int(self.base_width * 0.15)
        sidebar_max = int(self.base_width * 0.25)
        self.sidebar = QtWidgets.QWidget()
        self.sidebar.setMaximumWidth(sidebar_max)
        self.sidebar.setMinimumWidth(sidebar_min)
        self.sidebar.setStyleSheet(f"""
            QWidget {{
                background: rgba(52, 152, 219, 0.1);
                border-radius: {int(self.base_height * 0.02)}px;
                border: 2px solid #3498db;
            }}
        """)

        sidebar_margin = int(self.base_width * 0.015)
        sidebar_spacing = int(self.base_height * 0.02)
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(sidebar_margin, sidebar_margin, sidebar_margin, sidebar_margin)
        self.sidebar_layout.setSpacing(sidebar_spacing)

        # Add face detection and recognition controls
        self.setupFaceDetectionControls()

        # Add text box at the bottom left
        self.setupTextBox()

        # Right side with image displays
        self.right_side = QtWidgets.QWidget()
        self.right_side_layout = QtWidgets.QHBoxLayout(self.right_side)
        self.right_side_layout.setContentsMargins(0, 0, 0, 0)
        self.right_side_layout.setSpacing(int(self.base_width * 0.02))

        # Setup image display group boxes
        self.setupImageGroupBoxes()

        # Add widgets to layouts
        self.main_content_layout.addWidget(self.sidebar)
        self.main_content_layout.addWidget(self.right_side)

    def setupTextBox(self):
        """Creates a dynamically sized text box."""
        textbox_height = int(self.base_height * 0.15)
        self.info_textbox = QtWidgets.QTextEdit()
        self.info_textbox.setStyleSheet(self.textbox_style)
        self.info_textbox.setReadOnly(True)
        self.info_textbox.setPlaceholderText("Detection and recognition info will appear here...")
        self.info_textbox.setMaximumHeight(max(textbox_height, 100))  # Minimum height of 100px
        self.sidebar_layout.addWidget(self.info_textbox)

    def setupFaceDetectionControls(self):
        """Creates dynamically sized face detection controls."""
        # Dataset Selection Group
        dataset_group = QtWidgets.QGroupBox("Dataset Type")
        dataset_group.setStyleSheet(self.groupbox_style)
        dataset_layout = QtWidgets.QVBoxLayout()

        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.addItems(["RGB", "Grayscale"])
        combo_height = int(self.base_height * 0.04)
        self.dataset_combo.setMinimumHeight(combo_height)
        self.dataset_combo.setStyleSheet(f"""
            QComboBox {{
                background: transparent;
                color: white;
                border: 1px solid #00B4D8;
                border-radius: {int(combo_height * 0.25)}px;
                padding: {int(combo_height * 0.2)}px;
                min-height: {combo_height}px;
                font-size: {self.label_font_size}px;
            }}
            QComboBox:hover {{
                border: 1px solid #00B4D8;
                background: rgba(0,180,216,0.08);
            }}
            QComboBox::drop-down {{
                border: none;
                width: {combo_height}px;
            }}
            QComboBox::down-arrow {{
                image: url(static/icons/down-arrow.png);
                width: {int(combo_height * 0.5)}px;
                height: {int(combo_height * 0.5)}px;
            }}
            QComboBox QAbstractItemView {{
                background: rgba(10,35,66,0.85);
                color: white;
                border: 1px solid #00B4D8;
                selection-background-color: #00B4D8;
                selection-color: white;
                border-radius: {int(combo_height * 0.25)}px;
                font-size: {self.label_font_size}px;
            }}
        """)
        dataset_layout.addWidget(self.dataset_combo)
        dataset_group.setLayout(dataset_layout)

        # Face Detection Group
        detection_group = QtWidgets.QGroupBox("Face Detection")
        detection_group.setStyleSheet(self.groupbox_style)
        detection_layout = QtWidgets.QVBoxLayout()
        detection_layout.setSpacing(int(self.base_height * 0.01))

        # Add scale factor slider
        self.scale_factor_label = QtWidgets.QLabel("Scale Factor: 1.1")
        self.scale_factor_label.setStyleSheet(self.label_style)
        self.scale_factor_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_factor_slider.setMinimum(11)
        self.scale_factor_slider.setMaximum(20)
        self.scale_factor_slider.setValue(11)
        self.scale_factor_slider.setStyleSheet(self.slider_style)
        detection_layout.addWidget(self.scale_factor_label)
        detection_layout.addWidget(self.scale_factor_slider)

        # Add min neighbors slider
        self.min_neighbors_label = QtWidgets.QLabel("Min Neighbors: 5")
        self.min_neighbors_label.setStyleSheet(self.label_style)
        self.min_neighbors_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.min_neighbors_slider.setMinimum(1)
        self.min_neighbors_slider.setMaximum(10)
        self.min_neighbors_slider.setValue(5)
        self.min_neighbors_slider.setStyleSheet(self.slider_style)
        detection_layout.addWidget(self.min_neighbors_label)
        detection_layout.addWidget(self.min_neighbors_slider)

        # Add min size slider
        self.min_size_label = QtWidgets.QLabel("Min Size: 30")
        self.min_size_label.setStyleSheet(self.label_style)
        self.min_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.min_size_slider.setMinimum(10)
        self.min_size_slider.setMaximum(100)
        self.min_size_slider.setValue(30)
        self.min_size_slider.setStyleSheet(self.slider_style)
        detection_layout.addWidget(self.min_size_label)
        detection_layout.addWidget(self.min_size_slider)

        # Add detect faces button
        self.detect_faces_btn = QtWidgets.QPushButton("Detect Faces")
        self.detect_faces_btn.setStyleSheet(self.button_style)
        detection_layout.addWidget(self.detect_faces_btn)

        # Add color mode combo
        self.color_mode_combo = QtWidgets.QComboBox()
        self.color_mode_combo.addItems(["Color", "Grayscale"])
        self.color_mode_combo.setMinimumHeight(combo_height)
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

        recognition_group.setLayout(recognition_layout)

        # Add all groups to sidebar
        self.sidebar_layout.addWidget(detection_group)
        self.sidebar_layout.addWidget(recognition_group)

    def setupImageGroupBoxes(self):
        """Creates dynamically sized image group boxes."""
        # Original image group box
        self.original_groupBox = QtWidgets.QGroupBox("Original Image")
        self.original_groupBox.setStyleSheet(self.groupbox_style)
        self.original_layout = QtWidgets.QVBoxLayout(self.original_groupBox)
        groupbox_margin = int(self.base_height * 0.02)
        self.original_layout.setContentsMargins(groupbox_margin, groupbox_margin, groupbox_margin, groupbox_margin)
        self.original_layout.setSpacing(int(self.base_height * 0.01))

        self.original_image_label = QtWidgets.QLabel()
        self.original_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_image_label.setStyleSheet(f"""
            QLabel {{
                background: rgba(44, 62, 80, 0.3);
                border-radius: {int(self.base_height * 0.01)}px;
                padding: {int(self.base_height * 0.01)}px;
            }}
        """)
        self.original_layout.addWidget(self.original_image_label)

        # Processed image group box
        self.processed_groupBox = QtWidgets.QGroupBox("Processed Image")
        self.processed_groupBox.setStyleSheet(self.groupbox_style)
        self.processed_layout = QtWidgets.QVBoxLayout(self.processed_groupBox)
        self.processed_layout.setContentsMargins(groupbox_margin, groupbox_margin, groupbox_margin, groupbox_margin)
        self.processed_layout.setSpacing(int(self.base_height * 0.01))

        self.processed_image_label = QtWidgets.QLabel()
        self.processed_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.processed_image_label.setStyleSheet(f"""
            QLabel {{
                background: rgba(44, 62, 80, 0.3);
                border-radius: {int(self.base_height * 0.01)}px;
                padding: {int(self.base_height * 0.01)}px;
            }}
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
    # Controller Function
    # ----------------------------------------------------------------------
    def update_info_text(self, message):
        """Controller function to clear and update the text box with a message."""
        self.info_textbox.clear()  # Clear existing text
        self.info_textbox.append(message)  # Append new message

    # ----------------------------------------------------------------------
    # Show/Hide Logic
    # ----------------------------------------------------------------------
    def show_main_buttons(self):
        """Shows the original and processed group boxes."""
        self.original_groupBox.show()
        self.processed_groupBox.show()
