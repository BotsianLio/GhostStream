import sys
from PyQt5.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QDialog, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QComboBox
import cv2

# Import your new Worker and Selector
from capture.worker import VideoWorker
from gui.selector import CameraSelector

class AppWindow(QMainWindow):
    def __init__(self, camera_index):
        super().__init__()
        self.current_camera_index = camera_index
        self.worker = None

        self.setWindowTitle("GhostStream - Realtime Inpainting")
        self.resize(1280, 500) # Wide window for side-by-side view

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # --- 1. Top Options Bar (UPDATED) ---
        # We use a horizontal layout (QHBoxLayout) so the buttons sit next to each other
        self.top_bar_layout = QHBoxLayout()
        
        self.btn_reselect = QPushButton("Switch Camera")
        self.btn_reselect.setFixedWidth(150)
        self.btn_reselect.clicked.connect(self.open_reselect_dialog)
        self.top_bar_layout.addWidget(self.btn_reselect, alignment=Qt.AlignRight)

        self.btn_load_video = QPushButton("Load Video File")
        self.btn_load_video.setFixedWidth(150)
        self.btn_load_video.clicked.connect(self.load_video_file)
        self.top_bar_layout.addWidget(self.btn_load_video, alignment=Qt.AlignLeft)

        self.combo_method = QComboBox()
        self.combo_method.addItems(["RANSAC", "MAGSAC++"])
        self.combo_method.setFixedWidth(150)
        self.combo_method.currentTextChanged.connect(self.change_algorithm)
        self.top_bar_layout.addWidget(self.combo_method, alignment=Qt.AlignLeft)

        # CRITICAL: This line actually puts the buttons onto the screen
        self.layout.addLayout(self.top_bar_layout)

        # --- 2. Video Display ---
        self.video_label = QLabel("Initializing AI Pipeline...", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_label, stretch=1)

        # Start the Background Thread
        self.start_worker(camera_index)

    def start_worker(self, index):
        # Stop existing worker if running
        if self.worker is not None:
            self.worker.stop()

        # Create and start new worker
        self.worker = VideoWorker(index)
        self.worker.frame_processed.connect(self.update_display)
        self.worker.start()

    def update_display(self, frame):
        """Receives numpy array from worker thread"""
        # Convert BGR (OpenCV) to RGB (Qt)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit window
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def open_reselect_dialog(self):
        # Pause current worker
        if self.worker:
            self.worker.stop()

        selector = CameraSelector()
        if selector.exec_() == QDialog.Accepted:
            new_index = selector.selected_index
            print(f"Switching to Camera Index: {new_index}")
            self.current_camera_index = new_index
            self.start_worker(new_index)
        else:
            # Resume old camera if cancelled
            self.start_worker(self.current_camera_index)

    # --- 3. NEW VIDEO LOADER LOGIC ---
    def load_video_file(self):
        # Pause the live camera while the user is picking a file
        if self.worker:
            self.worker.stop()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Video File", 
            "", 
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)", 
            options=options
        )
        
        if file_path:
            print(f"Switching to Video File: {file_path}")
            self.current_camera_index = file_path # Save the string path
            self.start_worker(file_path) # Send the string path to the worker
        else:
            # If they cancel the file picker, resume the camera
            self.start_worker(self.current_camera_index)

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()

    def change_algorithm(self, method_name):
        print(f"GUI: Switching algorithm to {method_name}")
        if self.worker:
            self.worker.set_estimation_method(method_name)
