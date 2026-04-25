import cv2
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim

# Imports matched to your folder structure
from segmentation.segmentation_engine import SegmentationEngine
from motion.motion_estimator import MotionEstimator
from processing.background_model import BackgroundModel

def run_real_world_benchmark(test_video, clean_video, motion_method="MAGSAC++"):
    print(f"🚀 Starting Real-World Evaluation using {motion_method}...")
    
    cap_test = cv2.VideoCapture(test_video)
    cap_clean = cv2.VideoCapture(clean_video)

    seg_engine = SegmentationEngine() 
    motion_model = MotionEstimator()
    motion_model.set_method(motion_method) 
    bg_model = BackgroundModel(method='patchmatch')

    psnr_scores = []
    ssim_scores = []
    processing_times = []
    frame_count = 0

    while True:
        ret_test, frame_test = cap_test.read()
        ret_clean, frame_clean = cap_clean.read()

        if not ret_test or not ret_clean:
            break
            
        frame_test = cv2.resize(frame_test, (640, 360))
        frame_clean = cv2.resize(frame_clean, (640, 360))

        start_time = time.time()

        mask, debug_plot = seg_engine.get_mask(frame_test) 

        H_matrix = motion_model.calculate_camera_motion(frame_test, foreground_mask=mask)

        healed_frame, _ = bg_model.update(frame_test, mask, H_matrix)
        
        end_time = time.time()
        processing_times.append(end_time - start_time)

        current_psnr = cv2.PSNR(frame_clean, healed_frame)
        current_ssim = ssim(frame_clean, healed_frame, channel_axis=2, data_range=255)
        
        psnr_scores.append(current_psnr)
        ssim_scores.append(current_ssim)

        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Frame {frame_count} | PSNR: {np.mean(psnr_scores):.2f} dB | SSIM: {np.mean(ssim_scores):.3f}")

    cap_test.release()
    cap_clean.release()

    avg_time = np.mean(processing_times)
    fps = 1.0 / avg_time if avg_time > 0 else 0

    print("\n" + "="*40)
    print(f"🏆 {motion_method} + PatchMatch RESULTS 🏆")
    print("="*40)
    print(f"Frames Tested : {frame_count}")
    print(f"Average FPS   : {fps:.2f}")
    print(f"Average PSNR  : {np.mean(psnr_scores):.2f} dB")
    print(f"Average SSIM  : {np.mean(ssim_scores):.4f}")
    print("="*40)

if __name__ == "__main__":
    TEST_VIDEO = "walking_0.mp4"
    CLEAN_VIDEO = "clean_0.mp4"
    
    run_real_world_benchmark(TEST_VIDEO, CLEAN_VIDEO, motion_method="MAGSAC++")
