import cv2
import numpy as np
from PIL import Image

try:
    from patchmatch import patch_match
    HAS_PATCHMATCH = True
    print("PyPatchMatch successfully loaded!")
except ImportError:
    HAS_PATCHMATCH = False
    print("PyPatchMatch not found.")

try:
    import torch
    
    original_load = torch.jit.load
    def mac_safe_load(f, map_location=None, *args, **kwargs):
        return original_load(f, map_location='cpu', *args, **kwargs)
    torch.jit.load = mac_safe_load

    from simple_lama_inpainting import SimpleLama
    HAS_LAMA = True
    print("LaMa Deep Learning module found (Mac Patch Applied)!")
except ImportError:
    HAS_LAMA = False
    print("Simple LaMa not found. Run: pip install simple-lama-inpainting")


class BackgroundModel:
    def __init__(self, method='lama'):
        self.clean_background = None
        self.is_initialized = False
        self.kernel = np.ones((30, 30), np.uint8) 
        self.method = method

        if self.method == 'lama' and HAS_LAMA:
            print("⏳ Loading LaMa AI into Mac memory (This takes a few seconds)...")
            self.lama = SimpleLama()
            print("🚀 LaMa AI is ready!")

    def update(self, frame, mask, H):
        h, w = frame.shape[:2]

        dilated_mask = cv2.dilate(mask, self.kernel, iterations=1)

        if not self.is_initialized:
            self.clean_background = frame.copy()
            self.is_initialized = True
            return frame, self.clean_background

        warped_clean_bg = cv2.warpPerspective(
            self.clean_background, H, (w, h), 
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        output_frame = frame.copy()

        history_valid_mask = np.any(warped_clean_bg != [0, 0, 0], axis=-1).astype(np.uint8) * 255
        
        temporal_zone = cv2.bitwise_and(dilated_mask, history_valid_mask)
        spatial_zone = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(history_valid_mask))

        if np.count_nonzero(temporal_zone) > 0:
            output_frame[temporal_zone == 255] = warped_clean_bg[temporal_zone == 255]

        if np.count_nonzero(spatial_zone) > 0:
            
            if self.method == 'lama' and HAS_LAMA:
                frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                mask_pil = Image.fromarray(spatial_zone).convert('L')
                
                result_pil = self.lama(img_pil, mask_pil)
                
                inpainted_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                output_frame[spatial_zone == 255] = inpainted_bgr[spatial_zone == 255]

            elif self.method == 'patchmatch' and HAS_PATCHMATCH:
                frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                inpainted_rgb = patch_match.inpaint(frame_rgb, spatial_zone, patch_size=7)
                inpainted_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
                output_frame[spatial_zone == 255] = inpainted_bgr[spatial_zone == 255]
                
            else:
                output_frame = cv2.inpaint(output_frame, spatial_zone, 5, cv2.INPAINT_TELEA)

        update_mask = cv2.bitwise_not(dilated_mask)
        self.clean_background = warped_clean_bg.copy()
        self.clean_background[update_mask == 255] = frame[update_mask == 255]

        return output_frame, self.clean_background
