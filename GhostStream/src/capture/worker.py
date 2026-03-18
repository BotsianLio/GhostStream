from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import time
from processing.pipeline import VideoPipeline

class VideoWorker(QThread):
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, source): # <--- CHANGED: Accept 'source' (int or str)
        super().__init__()
        self.source = source
        self.running = True
        self.pipeline = None

    def set_estimation_method(self, method_name):
        if self.pipeline is not None:
            self.pipeline.set_estimation_method(method_name)

    def run(self):
        self.pipeline = VideoPipeline()
        
        # OpenCV magic: If source is 0, it opens webcam. If source is "video.mp4", it opens the file.
        cap = cv2.VideoCapture(self.source) 
        
        # Get the original video's FPS so we don't play it in fast-forward
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps if fps > 0 else 0.033 # Default to ~30fps if unknown
        
        while self.running:
            start_time = time.time()
            
            ret, frame = cap.read()
            # If a video file finishes, stop the loop
            if not ret: 
                break

            result_frame = self.pipeline.process(frame)

            if result_frame is not None:
                self.frame_processed.emit(result_frame)
                
            # --- CRITICAL FIX FOR VIDEO FILES ---
            # Webcams naturally wait for the next frame. Video files do not!
            # Without this delay, your M1 chip will process an mp4 at 200+ FPS 
            # and it will look like it's in extreme fast-forward.
            elapsed = time.time() - start_time
            sleep_time = delay - elapsed
            if sleep_time > 0 and isinstance(self.source, str):
                time.sleep(sleep_time)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()
