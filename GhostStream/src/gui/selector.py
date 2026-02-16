from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import platform

class CameraSelector(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Camera Source")
        self.setGeometry(200, 200, 600, 450)
        
        self.selected_index = None
        self.cap = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        # 1. Video Preview Area
        self.video_label = QLabel("Loading Camera Preview...", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000; color: #FFF; border: 1px solid #555;")
        self.video_label.setMinimumHeight(300)
        # Ensure it expands to fill space
        self.video_label.setSizePolicy(self.video_label.sizePolicy().Expanding, self.video_label.sizePolicy().Expanding)
        layout.addWidget(self.video_label)

        # 2. Controls Area
        controls = QHBoxLayout()
        
        self.combo = QComboBox()
        self.combo.currentIndexChanged.connect(self.change_camera_preview)
        controls.addWidget(self.combo, stretch=1)

        self.btn_confirm = QPushButton("Use This Camera")
        self.btn_confirm.clicked.connect(self.confirm_selection)
        controls.addWidget(self.btn_confirm)

        layout.addLayout(controls)

        # 3. Timer for updating the preview frame
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)

        # 4. Initial Scan
        self.scan_cameras()

    def scan_cameras(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY

        # Quickly check indices 0-9 to see which are valid
        for i in range(5):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.combo.addItem(f"Camera Index {i}", i)
                cap.release()

        self.combo.blockSignals(False)
        
        # Determine initial selection
        if self.combo.count() > 0:
            self.change_camera_preview()
        else:
            self.video_label.setText("No Cameras Found.\nCheck USB Connection.")
            self.btn_confirm.setEnabled(False)

    def change_camera_preview(self):
        # Stop previous camera
        if self.cap:
            self.cap.release()
            self.timer.stop()
        
        index = self.combo.currentData()
        if index is not None:
            backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(index, backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            
            self.timer.start(33) # 30 FPS

    def update_preview(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                
                qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Scale to fit label
                pix = QPixmap.fromImage(qt_img).scaled(
                    self.video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(pix)

    def confirm_selection(self):
        self.selected_index = self.combo.currentData()
        # Release resource so Main Window can pick it up
        if self.cap:
            self.cap.release()
        self.accept()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()