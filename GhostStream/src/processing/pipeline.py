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

    def set_estimation_method(self, method_name):
        self.motion.set_method(method_name)
        
    def process(self, frame):
        if frame is None: return None
        
        working_frame = cv2.resize(frame, (640, 360))

        mask, debug_yolo = self.segmentation.get_mask(working_frame)

        H = self.motion.calculate_camera_motion(working_frame)

        final_result, internal_memory = self.background_model.update(working_frame, mask, H)

        if internal_memory is None:
            internal_memory = np.zeros_like(working_frame)
            
        combined_result = np.hstack((debug_yolo, internal_memory, final_result))
        
        return combined_result
