from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QFileDialog, QMainWindow, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import platform
from gui.app_window import AppWindow
from gui.video_window import VideoWindow
from framesource.type import FrameSourceType

class CameraSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Camera Source")
        self.setGeometry(200, 200, 600, 450)
        
        self.selected_index = None
        self.cap = None

        self.filepath = None

        self.dialogs = []

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

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
        
        # 3. Add Open Video Button
        #layout.addStretch()

        layout2 = QHBoxLayout()
        layout.addLayout(layout2)

        self.open_video_btn = QPushButton("Open Video")
        self.open_video_btn.clicked.connect(self.openFile)
        
        self.frame_rate_label = QLabel("Video Max Frame Rate:")
        self.frame_rate_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.frame_rate_input = QComboBox()
        self.frame_rate_input.addItems(['Default', '10', '20', '30', '40', '50', '60'])
        
        layout2.addWidget(self.open_video_btn)
        layout2.addWidget(self.frame_rate_label)
        layout2.addWidget(self.frame_rate_input)

        # 3. Timer for updating the preview frame
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)

        # 4. Initial Scan
        self.scan_cameras()

    def scan_cameras(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY

        active_cap = []
        for i in range(5):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.combo.addItem(f"Camera Index {i}", i)
                active_cap.append(cap)

        for cap in active_cap:
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
            ret, frame = self.cap.read()
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
        selected_index = self.combo.currentData()
        # Release resource so Main Window can pick it up
        if self.cap:
            self.cap.release()
        new_dialog = AppWindow(selected_index)
        new_dialog.closed_signal.connect(self.closeDialogEvent)
        self.dialogs.append(new_dialog)
        new_dialog.show()
    
    def closeDialogEvent(self, frame_source_type, frame_source, dialog):
        index = self.combo.currentData()
        if frame_source_type == FrameSourceType.CAMERA and index == frame_source:
            self.change_camera_preview()
        self.dialogs.remove(dialog)

    def closeEvent(self, event):
        for dialog in self.dialogs:
            dialog.close()
        # have no idea whether isOpened can be true if a dialog has it open
        # TODO double check this
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()

    def openFile(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "Open File", 
            "/home", 
            "Video files (*.mp4)"
        )

        if filename != "":
            new_dialog = VideoWindow(filename, self.frame_rate_input.currentText())
            new_dialog.closed_signal.connect(self.closeDialogEvent)
            self.dialogs.append(new_dialog)
            new_dialog.show()
        
