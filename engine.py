import cv2
import numpy as np
import time

class RIFEEngine:
    def __init__(self, model_version="rife-v4"):
        """
        Using OpenCV DISOpticalFlow as a fast fallback that works in real-time.
        """
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        print("Optical Flow Engine (DIS) initialized.")

    def interpolate(self, frame1, frame2):
        """
        Interpolate between frame1 and frame2 using optical flow.
        """
        # Convert to grayscale for flow calculation
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Calculate flow
        # We can use a lower resolution for flow to speed it up
        scale = 0.5
        small1 = cv2.resize(gray1, (0, 0), fx=scale, fy=scale)
        small2 = cv2.resize(gray2, (0, 0), fx=scale, fy=scale)
        
        flow = self.dis.calc(small1, small2, None)
        
        # Scale flow back
        flow = flow * (1.0 / scale)
        flow = cv2.resize(flow, (frame1.shape[1], frame1.shape[0]))
        
        # Warp frames (mid-way)
        # For simplicity, we just warp frame1 by half-flow
        mid_flow = flow * 0.5
        
        h, w = frame1.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Shift maps by mid_flow
        map_x = (map_x + mid_flow[..., 0]).astype(np.float32)
        map_y = (map_y + mid_flow[..., 1]).astype(np.float32)
        
        # Remap
        inter_frame = cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR)
        
        return inter_frame

if __name__ == "__main__":
    # Test engine
    engine = RIFEEngine()
    f1 = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(f1, (100, 100), 50, (255, 0, 0), -1)
    f2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(f2, (200, 100), 50, (255, 0, 0), -1)
    
    print("Testing Optical Flow interpolation...")
    start = time.time()
    result = engine.interpolate(f1, f2)
    end = time.time()
    print(f"Interpolation took: {end - start:.4f} seconds ({1/(end-start):.2f} FPS)")
    cv2.imwrite("test_inter.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
