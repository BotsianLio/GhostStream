import cv2
import numpy as np

class MotionEstimator:
    def __init__(self):
        # ORB is the feature detector (fast on M1)
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # Matcher compares features between frames
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Storage for previous frame data
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

    def calculate_camera_motion(self, curr_frame, foreground_mask=None):
        """
        Calculates the Homography Matrix (H) aligning Previous Frame -> Current Frame.
        Accepts an optional foreground_mask to ignore the moving subject.
        """
        # 1. Convert to Grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # --- NEW MASK LOGIC START ---
        background_mask = None
        if foreground_mask is not None:
            # Ensure mask is a single channel 8-bit image (required by cv2)
            if len(foreground_mask.shape) == 3:
                foreground_mask = cv2.cvtColor(foreground_mask, cv2.COLOR_BGR2GRAY)
            
            # Threshold to be safe: ensure foreground is 255 (white) and rest is 0
            _, binary_mask = cv2.threshold(foreground_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Invert: We want background to be 255 (white) so ORB looks there,
            # and foreground object to be 0 (black) so ORB ignores it completely.
            background_mask = cv2.bitwise_not(binary_mask)
        # --- NEW MASK LOGIC END ---

        # 2. Detect Features (NOW USING THE MASK)
        kp, des = self.orb.detectAndCompute(curr_gray, mask=background_mask)

        # First frame check
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            self.prev_kp = kp
            self.prev_des = des
            return np.eye(3) # No movement

        # 3. Match Features
        # Safety check: if ORB found no features (e.g., staring at a blank wall)
        if des is None or self.prev_des is None or len(des) == 0 or len(self.prev_des) == 0:
            return np.eye(3)

        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        # We need good matches to calculate geometry
        good_matches = matches[:50] 
        
        H = np.eye(3)

        if len(good_matches) > 4:
            src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 4. Find Homography
            H_computed, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H_computed is not None:
                H = H_computed

        # Update history for next loop
        self.prev_gray = curr_gray
        self.prev_kp = kp
        self.prev_des = des
        
        return H
