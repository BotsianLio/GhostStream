from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
from processing.pipeline import VideoPipeline
from framesource.type import FrameSourceType

class VideoWorker(QThread):
    # Signals must be defined at class level
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, frame_source, frame_source_type):
        super().__init__()
        self.frame_source = frame_source
        self.frame_source_type = frame_source_type
        self.running = True
        self.pipeline = None

    def run(self):
        # Initialize the pipeline inside the thread (Thread Safety)
        self.pipeline = VideoPipeline()
        
        # Open Camera
        cap = cv2.VideoCapture(self.frame_source)
        
        while self.running:
            ret, frame = cap.read()
            if not ret: break

            # Pass to Pipeline (YOLO + Motion)
            result_frame = None
            if self.frame_source_type == FrameSourceType.CAMERA:
                result_frame = self.pipeline.process(frame)
            elif self.frame_source_type == FrameSourceType.VIDEO:
                result_frame = self.pipeline.process(frame)

            # Send result to GUI
            if result_frame is not None and self.frame_source_type == FrameSourceType.CAMERA:
                self.frame_processed.emit(result_frame)
            elif result_frame is not None and self.frame_source_type == FrameSourceType.VIDEO:
                self.frame_processed.emit(result_frame)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()
