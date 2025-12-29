import cv2
import numpy as np
import time

class RIFEEngine:
    def __init__(self, model_version="rife-v4"):
        """
        Using OpenCV DISOpticalFlow as a fast fallback that works in real-time.
        """
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.dis.setFinestScale(1)
        self.dis.setGradientDescentIterations(10)
        self.dis.setVariationalRefinementIterations(0) # Disable for speed
        
        self.last_h = 0
        self.last_w = 0
        self.map_x = None
        self.map_y = None
        
        print("Optimized Optical Flow Engine (DIS ULTRAFAST) initialized.")

    def interpolate(self, frame1, frame2):
        """
        Interpolate between frame1 and frame2 using optical flow.
        """
        h, w = frame1.shape[:2]
        
        # Convert to grayscale 
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Calculate flow at lower resolution
        scale = 0.5
        small1 = cv2.resize(gray1, (0, 0), fx=scale, fy=scale)
        small2 = cv2.resize(gray2, (0, 0), fx=scale, fy=scale)
        
        flow = self.dis.calc(small1, small2, None)
        
        # Scale and resize flow back
        flow = flow * (1.0 / scale)
        flow = cv2.resize(flow, (w, h))
        
        # Warp frames (mid-way)
        mid_flow = flow * 0.5
        
        # Cache meshgrid to avoid reallocation
        if h != self.last_h or w != self.last_w:
            self.map_x, self.map_y = np.meshgrid(np.arange(w), np.arange(h))
            self.map_x = self.map_x.astype(np.float32)
            self.map_y = self.map_y.astype(np.float32)
            self.last_h, self.last_w = h, w
        
        # Shift maps by mid_flow
        m_x = self.map_x + mid_flow[..., 0]
        m_y = self.map_y + mid_flow[..., 1]
        
        # Remap
        inter_frame = cv2.remap(frame1, m_x, m_y, cv2.INTER_LINEAR)
        
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
