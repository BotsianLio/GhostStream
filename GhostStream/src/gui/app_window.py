import sys
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QDialog
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2

# Import your new Worker and Selector
from capture.worker import VideoWorker
from framesource.type import FrameSourceType

class AppWindow(QDialog):
    closed_signal = pyqtSignal(FrameSourceType, int, QDialog)

    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.worker = None

        self.setWindowTitle("GhostStream - Realtime Inpainting")
        self.resize(1280, 500) # Wide window for side-by-side view

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # --- 2. Video Display ---
        self.video_label = QLabel("Initializing AI Pipeline...", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_label, stretch=1)

        # Start the Background Thread
        self.start_worker()

    def start_worker(self):
        # Stop existing worker if running
        if self.worker is not None:
            self.worker.stop()

        # Create and start new worker
        self.worker = VideoWorker(self.camera_index, FrameSourceType.CAMERA)
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

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        self.closed_signal.emit(FrameSourceType.CAMERA, self.camera_index, self)
        event.accept()
