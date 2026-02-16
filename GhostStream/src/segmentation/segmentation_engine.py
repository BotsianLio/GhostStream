from ultralytics import YOLO
import cv2
import numpy as np
import torch

class SegmentationEngine:
    def __init__(self, model_path='yolov8n-seg.pt'):
        # Check for Apple Silicon (Metal) acceleration
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.target_classes = [0] # 0 is 'person' in COCO

    def get_mask(self, frame):
        # imgsz=320 makes it MUCH faster on M1
        results = self.model.predict(
            source=frame, 
            conf=0.4, 
            imgsz=320,
            classes=self.target_classes,
            verbose=False,
            device=self.device
        )

        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if results[0].masks is not None:
            # Combine all detected person masks into one
            for m in results[0].masks.data.cpu().numpy():
                m_resized = cv2.resize(m, (w, h))
                mask = np.maximum(mask, (m_resized * 255).astype(np.uint8))

        return mask, results[0].plot()