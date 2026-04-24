import sys
import os 
from datetime import datetime 
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QDialog
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2

from capture.worker import VideoWorker
from ghoststreamenums import FrameSourceType

class VideoWindow(QDialog):
    closed_signal = pyqtSignal(FrameSourceType, str, QDialog)

    def __init__(self, frame_source, setting, method):
        super().__init__()
        self.frame_source = frame_source
        self.setting = setting
        self.method = method
        self.video_writer = None        
        
        self.worker = None
        self.processed_images = []
        self.index = 0

        save_dir = os.path.join(os.getcwd(), "result_video")
        os.makedirs(save_dir, exist_ok=True) 

        if isinstance(self.frame_source, str) and self.frame_source.endswith('.mp4'):
            base_name = os.path.basename(self.frame_source)
            self.output_path = os.path.join(save_dir, f"ghost_{base_name}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = os.path.join(save_dir, f"live_capture_{timestamp}.mp4")
            
        print(f"Setup complete. Video will be saved to: {self.output_path}")

        self.setWindowTitle("GhostStream - Realtime Inpainting")
        self.resize(1280, 500) 
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.video_label = QLabel("Initializing AI Pipeline...", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_label, stretch=1)
        
        if (self.setting != "Default") and (self.setting != "Live"):
            raise Exception("Error")

        self.start_worker()

    def start_worker(self):
        if self.worker is not None:
            self.worker.stop()

        if self.setting == "Live":
            self.worker = VideoWorker(self.frame_source, FrameSourceType.CAMERA)
            self.worker.frame_processed.connect(self.updateDisplay)
        elif self.setting == "Default":
            self.worker = VideoWorker(self.frame_source, FrameSourceType.VIDEO)
            self.worker.video_processed.connect(self.processFrames)
        
        self.worker.set_estimation_method(self.method)
        self.worker.start()

    def processFrames(self, frames, frame_rate):
        """Receives numpy array from worker thread"""
        print(f"FPS reported by worker: {frame_rate}")
        
        if len(frames) > 0:
            h, combined_w = frames[0].shape[:2]
            
            single_w = combined_w // 3
            
            print(f"🎬 Saving ONLY the result video ({single_w}x{h})...")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = float(frame_rate) if frame_rate > 0 else 30.0
            
            writer = cv2.VideoWriter(self.output_path, fourcc, fps, (single_w, h))
            
            for i in range(frames.shape[0]):
                result_only = frames[i][:, single_w * 2:] 
                writer.write(result_only)
                
            writer.release()
            print(f"SUCCESSFULLY SAVED SINGLE RESULT: {self.output_path}")

        for i in range(frames.shape[0]):
            frame = frames[i]
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
        
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), 
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.processed_images.append(scaled_pixmap)
            
        self.timer.start(int(1000 / frame_rate) if frame_rate > 0 else 33)

    def updateFrame(self):
        if self.index < len(self.processed_images):
            self.video_label.setPixmap(self.processed_images[self.index])
            self.index += 1
        else:
            self.timer.stop()

    def updateDisplay(self, frame):
        """Receives numpy array from worker thread"""
        h, combined_w = frame.shape[:2]
        single_w = combined_w // 3
        
        result_only = frame[:, single_w * 2:]

        if self.video_writer is None:
            print(f"🎬 Starting live single-result recording ({single_w}x{h})...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, 30.0, (single_w, h))
        
        self.video_writer.write(result_only)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"LIVE STREAM SAVED: {self.output_path}")
            
        self.closed_signal.emit(FrameSourceType.VIDEO, self.frame_source, self)
        event.accept()
