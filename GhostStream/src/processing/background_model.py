import cv2
import numpy as np

class BackgroundModel:
    def __init__(self):
        self.clean_background = None
        self.is_initialized = False
        
        # Dilation Kernel: Used to expand the mask slightly
        self.kernel = np.ones((10, 10), np.uint8) 

    def update(self, frame, mask, H):
        h, w = frame.shape[:2]

        # 1. Expand the mask (CRITICAL FIX)
        # This covers the "edge" of the person so we don't smudge their hair
        dilated_mask = cv2.dilate(mask, self.kernel, iterations=1)

        # 2. Initialize
        if not self.is_initialized:
            self.clean_background = frame.copy()
            self.is_initialized = True
            return frame, self.clean_background

        # 3. Warp History
        warped_clean_bg = cv2.warpPerspective(
            self.clean_background, H, (w, h), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        output_frame = frame.copy()

        # 4. Identify Valid History
        # Pixels where we have data (not black borders)
        history_valid_mask = np.any(warped_clean_bg != [0, 0, 0], axis=-1).astype(np.uint8) * 255
        
        # Zone where we can use History (Time Travel)
        temporal_zone = cv2.bitwise_and(dilated_mask, history_valid_mask)
        
        # Zone where we MUST smudge (No History)
        spatial_zone = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(history_valid_mask))

        # 5. Apply Inpainting
        # A. Prefer History
        if np.count_nonzero(temporal_zone) > 0:
            output_frame[temporal_zone == 255] = warped_clean_bg[temporal_zone == 255]

        # B. Fallback to Smudge (Spatial)
        if np.count_nonzero(spatial_zone) > 0:
            # We use a larger radius (5) to blur it more smoothly
            output_frame = cv2.inpaint(output_frame, dilated_mask, 5, cv2.INPAINT_TELEA)

        # 6. Update Memory
        # Only learn from areas that are NOT the person
        update_mask = cv2.bitwise_not(dilated_mask)
        
        self.clean_background = warped_clean_bg.copy()
        self.clean_background[update_mask == 255] = frame[update_mask == 255]

        # Return BOTH the result AND the internal memory (for debugging)
        return output_frame, self.clean_background