from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import time
from processing.pipeline import VideoPipeline
from ghoststreamenums import FrameSourceType

class VideoWorker(QThread):
    frame_processed = pyqtSignal(np.ndarray)
    video_processed = pyqtSignal(np.ndarray, float)

    def __init__(self, frame_source, frame_source_type,frame_skip=1): # <--- CHANGED: Accept 'source' (int or str)
        super().__init__()
        self.frame_source = frame_source
        self.frame_source_type = frame_source_type
        self.frame_skip = max(1, int(frame_skip))
        self.running = True
        self.pipeline = None

    def set_estimation_method(self, method_name):
        if self.pipeline is not None:
            self.pipeline.set_estimation_method(method_name)

    def run(self):
        self.pipeline = VideoPipeline()
        
        # OpenCV magic: If source is 0, it opens webcam. If source is "video.mp4", it opens the file.
        cap = cv2.VideoCapture(self.frame_source) 
        
        # Get the original video's FPS so we don't play it in fast-forward
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps) 
        if self.frame_source_type == FrameSourceType.VIDEO:
            processed_frames = np.empty((0, 360, 1920, 3), dtype=np.uint8)
        index = 0
        while self.running:
            start_time = time.time()
            
            ret, frame = cap.read()
            # If a video file finishes, stop the loop
            if not ret: 
                break
            if index % self.frame_skip != 0:
                index += 1
                continue
            result_frame = self.pipeline.process(frame)

            # Send result to GUI
            if result_frame is not None and self.frame_source_type == FrameSourceType.CAMERA:
                self.frame_processed.emit(result_frame)
            elif result_frame is not None and self.frame_source_type == FrameSourceType.VIDEO:
                print(f"FRAME {index} PROCESSED")
                result_frame = result_frame.reshape((1, 360, 1920, 3))
                processed_frames = np.append(processed_frames, result_frame, axis=0)
            index += 1

        if self.frame_source_type == FrameSourceType.VIDEO and processed_frames.size != 0:
            print("DONE, EMITTING FRAMES")
            self.video_processed.emit(processed_frames, fps)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()
