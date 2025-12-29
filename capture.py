import dxcam
import time
import cv2
import numpy as np

class ScreenCapture:
    def __init__(self, region=None, device_idx=0, output_color="RGB"):
        """
        Initialize the DXCAM capture.
        :param region: Tuple of (left, top, right, bottom). If None, captures full screen.
        """
        self.camera = dxcam.create(device_idx=device_idx, output_color=output_color)
        self.region = region
        self.is_capturing = False
        
    def capture_frame(self):
        """
        Captures a single frame.
        """
        if self.region:
            return self.camera.grab(region=self.region)
        else:
            return self.camera.grab()

    def start_high_speed_capture(self, target_fps=60):
        """
        Starts a continuous capture loop.
        """
        self.camera.start(target_fps=target_fps, region=self.region)
        self.is_capturing = True
        print(f"Started DXCAM capture at {target_fps} FPS")

    def get_latest_frame(self):
        return self.camera.get_latest_frame()

    def stop_capture(self):
        if self.is_capturing:
            self.camera.stop()
            self.is_capturing = False

if __name__ == "__main__":
    # Test capture
    cap = ScreenCapture()
    print("Testing capture speed for 100 frames with DXCAM...")
    start_time = time.time()
    count = 0
    while count < 100:
        frame = cap.capture_frame()
        if frame is not None:
            count += 1
    end_time = time.time()
    print(f"Captured 100 frames in {end_time - start_time:.4f} seconds")
    print(f"Effective FPS: {100 / (end_time - start_time):.2f}")
    
    # Show one frame to verify
    frame = cap.capture_frame()
    if frame is not None:
        cv2.imwrite("test_capture.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("Test frame saved to test_capture.jpg")
