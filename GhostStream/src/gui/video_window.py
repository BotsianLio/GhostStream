import sys
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QDialog
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2

# Import your new Worker and Selector
from capture.worker import VideoWorker
from framesource.type import FrameSourceType

class VideoWindow(QDialog):
    closed_signal = pyqtSignal(FrameSourceType, str, QDialog)

    def __init__(self, frame_source, frame_rate):
        super().__init__()
        self.frame_source = frame_source
        self.frame_rate = frame_rate
        self.worker = None
        self.processed_images = []
        self.index = 0

        self.setWindowTitle("GhostStream - Realtime Inpainting")
        self.resize(1280, 500) # Wide window for side-by-side view
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # --- 2. Video Display ---
        self.video_label = QLabel("Initializing AI Pipeline...", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_label, stretch=1)
        
        # Checks
        if (self.frame_rate != "Default") and not (self.frame_rate.isnumeric() and int(self.frame_rate) > 0):
            raise Exception("Error")

        # Start the Background Thread
        self.start_worker()

    def start_worker(self):
        # Stop existing worker if running
        if self.worker is not None:
            self.worker.stop()

        # Create and start new worker
        if self.frame_rate == "Default":
            # If default treat like a camera display and update display when frames are processed
            self.worker = VideoWorker(self.frame_source, FrameSourceType.CAMERA)
            self.worker.frame_processed.connect(self.updateDisplay)
        elif self.frame_rate.isnumeric():
            self.worker = VideoWorker(self.frame_source, FrameSourceType.VIDEO)
            self.worker.frame_processed.connect(self.processFrames)
        self.worker.start()

    def processFrames(self, frames):
        """Receives numpy array from worker thread"""
        # Convert BGR (OpenCV) to RGB (Qt)
        print(frames.shape)
        for i in range(frames.shape[0]):
            frame = frames[i]
            print(frame.shape)
            print(frame)
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
            self.processed_images.append(scaled_pixmap)
        self.timer.start(int(1000 / int(self.frame_rate)))

    def updateFrame(self):
        if self.index < len(self.processed_images):
            self.video_label.setPixmap(self.processed_images[self.index])
            self.index += 1
        else:
            self.timer.stop()

    def updateDisplay(self, frame):
        """Receives numpy array from worker thread"""
        # Convert BGR (OpenCV) to RGB (Qt)
        print(frame.shape)
        print(frame.dtype)
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
        self.closed_signal.emit(FrameSourceType.VIDEO, self.frame_source, self)
        event.accept()
