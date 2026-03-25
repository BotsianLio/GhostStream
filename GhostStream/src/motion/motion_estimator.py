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
        
        # --- NEW: Track current algorithm ---
        self.current_method = "RANSAC"

    # --- NEW: Method to change the algorithm ---
    def set_method(self, method_name):
        self.current_method = method_name
        print(f"[MotionEstimator] Algorithm switched to: {method_name}")

    def calculate_camera_motion(self, curr_frame, foreground_mask=None):
        # 1. Convert to Grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        background_mask = None
        if foreground_mask is not None:
            if len(foreground_mask.shape) == 3:
                foreground_mask = cv2.cvtColor(foreground_mask, cv2.COLOR_BGR2GRAY)
            
            _, binary_mask = cv2.threshold(foreground_mask, 127, 255, cv2.THRESH_BINARY)
            background_mask = cv2.bitwise_not(binary_mask)

        # 2. Detect Features
        kp, des = self.orb.detectAndCompute(curr_gray, mask=background_mask)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            self.prev_kp = kp
            self.prev_des = des
            return np.eye(3) 

        if des is None or self.prev_des is None or len(des) == 0 or len(self.prev_des) == 0:
            return np.eye(3)

        # 3. Match Features
        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = matches[:50] 
        H = np.eye(3)

        if len(good_matches) > 4:
            src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # --- NEW: Choose RANSAC or MAGSAC++ ---
            if self.current_method == "MAGSAC++":
                flag = cv2.USAC_MAGSAC
                # MAGSAC++ often works better with a slightly tighter threshold, like 3.0
                threshold = 3.0 
            else:
                flag = cv2.RANSAC
                threshold = 5.0

            # 4. Find Homography using the selected flag
            H_computed, mask = cv2.findHomography(src_pts, dst_pts, flag, threshold)
            
            if H_computed is not None:
                H = H_computed

        self.prev_gray = curr_gray
        self.prev_kp = kp
        self.prev_des = des
        
        return H
