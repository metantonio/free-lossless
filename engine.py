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
        self.high_precision = False
        
        print("Optimized Optical Flow Engine (DIS ULTRAFAST) initialized.")

    def set_high_precision(self, enabled):
        self.high_precision = enabled
        if enabled:
            # Ultra Smooth Mode: Higher precision, way more CPU usage
            self.dis.setFinestScale(0) # 0 is finer than 1
            self.dis.setGradientDescentIterations(25)
            self.dis.setVariationalRefinementIterations(5)
            print("Engine set to ULTRA SMOOTH (High Precision)")
        else:
            # Standard Mode: Fast and efficient
            self.dis.setFinestScale(1)
            self.dis.setGradientDescentIterations(10)
            self.dis.setVariationalRefinementIterations(0)
            print("Engine set to STANDARD Precision")

    def interpolate(self, frame1, frame2):
        """
        Interpolate between frame1 and frame2 using optical flow.
        """
        h, w = frame1.shape[:2]
        
        # 1. Quick static check to save CPU
        # Sample a small region to see if anything moved
        if np.array_equal(frame1[::100, ::100], frame2[::100, ::100]):
            return frame1

        # 2. Convert to grayscale (reuse buffer if possible)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # 3. Calculate flow at even lower resolution (Quarter Res)
        scale = 0.25
        small1 = cv2.resize(gray1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small2 = cv2.resize(gray2, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        flow = self.dis.calc(small1, small2, None)
        
        # 4. Scale and resize flow back - INTER_NEAREST is faster and fine for flow maps
        flow = flow * (1.0 / scale)
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Warp frames (mid-way)
        mid_flow = flow * 0.5
        
        # Cache meshgrid to avoid reallocation
        if h != self.last_h or w != self.last_w:
            self.map_x, self.map_y = np.meshgrid(np.arange(w), np.arange(h))
            self.map_x = self.map_x.astype(np.float32)
            self.map_y = self.map_y.astype(np.float32)
            self.last_h, self.last_w = h, w
        
        # Shift maps by mid_flow
        # Text protection: skip warp for very small movements
        motion_mag_sq = mid_flow[..., 0]**2 + mid_flow[..., 1]**2
        mid_flow[motion_mag_sq < 0.25] = 0 # 0.5 pixel threshold squared
        
        m_x = self.map_x + mid_flow[..., 0]
        m_y = self.map_y + mid_flow[..., 1]
        
        # Remap - Linear is necessary for visual quality
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
