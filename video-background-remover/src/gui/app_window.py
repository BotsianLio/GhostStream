from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
from capture.camera_stream import CameraStream # Needed for re-creating stream
from gui.selector import CameraSelector       # Needed for the pop-up


class AppWindow(QMainWindow):
    def __init__(self, camera_stream):
        super().__init__()
        self.camera_stream = camera_stream
        
        # self.pipeline = VideoPipeline() # <--- UNCOMMENT LATER

        self.setWindowTitle("DJI Action - Video Stream")
        self.setGeometry(100, 100, 1000, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # --- 1. Top Options Bar ---
        self.btn_reselect = QPushButton("Switch Camera")
        self.btn_reselect.setFixedWidth(150)
        self.btn_reselect.clicked.connect(self.open_reselect_dialog)
        self.layout.addWidget(self.btn_reselect, alignment=Qt.AlignCenter)

        # --- 2. Video Display ---
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        # Give the video label the most space (stretch factor 1)
        self.layout.addWidget(self.video_label, stretch=1)

        # Start the camera
        self.camera_stream.start_stream()

        # Setup timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def update_frame(self):
        # 1. Get raw frame
        frame = self.camera_stream.get_frame()

        if frame is not None:
            # --- PIPELINE PLACEHOLDER ---
            # processed_frame = self.pipeline.process(frame) 
            processed_frame = frame # Currently just showing raw video
            
            # 2. Convert to Qt
            h, w, ch = processed_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 3. Scale fit
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), 
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.video_label.setPixmap(scaled_pixmap)
        else:
            self.video_label.setText("Waiting for Camera...")

    def open_reselect_dialog(self):
        """Standard procedure to pause, choose new camera, and resume."""
        # 1. Stop Update Loop & Release hardware
        self.timer.stop()
        self.camera_stream.stop_stream()

        # 2. Show Selector
        selector = CameraSelector()
        if selector.exec_() == QDialog.Accepted:
            # 3. If new choice made, re-initialize
            new_index = selector.selected_index
            print(f"Switching to Camera Index: {new_index}")
            
            self.camera_stream = CameraStream(source=new_index)
            self.camera_stream.start_stream()
        else:
            # 4. If cancelled, just re-start old stream
            self.camera_stream.start_stream()

        # Resume Loop
        self.timer.start(33)

    def closeEvent(self, event):
        self.camera_stream.stop_stream()
        event.accept()