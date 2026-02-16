from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
from processing.pipeline import VideoPipeline

class VideoWorker(QThread):
    # Signals must be defined at class level
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.pipeline = None

    def run(self):
        # Initialize the pipeline inside the thread (Thread Safety)
        self.pipeline = VideoPipeline()
        
        # Open Camera
        cap = cv2.VideoCapture(self.camera_index)
        
        while self.running:
            ret, frame = cap.read()
            if not ret: break

            # Pass to Pipeline (YOLO + Motion)
            result_frame = self.pipeline.process(frame)

            # Send result to GUI
            if result_frame is not None:
                self.frame_processed.emit(result_frame)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()