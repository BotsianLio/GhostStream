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
        
        if self.frame_source_type == FrameSourceType.VIDEO:
            processed_frames = np.empty((0, 360, 1920, 3), dtype=np.uint8)
        index = 0
        while self.running:
            ret, frame = cap.read()
            if not ret: break

            # Pass to Pipeline (YOLO + Motion)
            result_frame = self.pipeline.process(frame)

            # Send result to GUI
            if result_frame is not None and self.frame_source_type == FrameSourceType.CAMERA:
                self.frame_processed.emit(result_frame)
            elif result_frame is not None and self.frame_source_type == FrameSourceType.VIDEO:
                print(f"FRAME {index} PROCESSED")
                index += 1
                result_frame = result_frame.reshape((1, 360, 1920, 3))
                print(processed_frames.shape)
                print(result_frame.shape)
                processed_frames = np.append(processed_frames, result_frame, axis=0)

        if self.frame_source_type == FrameSourceType.VIDEO and processed_frames.size != 0:
            print("DONE, EMITTING FRAMES")
            self.frame_processed.emit(processed_frames)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()
