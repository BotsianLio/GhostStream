from ultralytics import YOLO
import cv2
import numpy as np
import torch

class SegmentationEngine:
    def __init__(self, model_path='yolov8n-seg.pt'):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.target_classes = [0, 2] 

        self.dilation_kernel = np.ones((15, 15), np.uint8)

    def get_mask(self, frame):
        results = self.model.predict(
            source=frame, 
            conf=0.2, 
            imgsz=320,
            classes=self.target_classes,
            verbose=False,
            device=self.device
        )

        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if results[0].masks is not None:
            for m in results[0].masks.data.cpu().numpy():
                m_resized = cv2.resize(m, (w, h))
                mask = np.maximum(mask, (m_resized * 255).astype(np.uint8))
            
            mask = cv2.dilate(mask, self.dilation_kernel, iterations=1)

        return mask, results[0].plot()
