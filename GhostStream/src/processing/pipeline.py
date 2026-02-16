import cv2
import numpy as np
from segmentation.segmentation_engine import SegmentationEngine
from motion.motion_estimator import MotionEstimator
from processing.background_model import BackgroundModel

class VideoPipeline:
    def __init__(self):
        self.segmentation = SegmentationEngine()
        self.motion = MotionEstimator()
        self.background_model = BackgroundModel()

    def process(self, frame):
        if frame is None: return None
        
        working_frame = cv2.resize(frame, (640, 360))

        # 1. Mask
        mask, debug_yolo = self.segmentation.get_mask(working_frame)

        # 2. Motion
        H = self.motion.calculate_camera_motion(working_frame)

        # 3. Background Update
        # NOW RETURNS TWO IMAGES: The Result + The Memory
        final_result, internal_memory = self.background_model.update(working_frame, mask, H)

        # 4. Create the "Debug Strip"
        # We stack 3 images horizontally
        if internal_memory is None:
            internal_memory = np.zeros_like(working_frame)
            
        combined_result = np.hstack((debug_yolo, internal_memory, final_result))
        
        return combined_result