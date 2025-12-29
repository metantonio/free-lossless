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
        Interpolate between frame1 and frame2 using stabilized bilateral warping.
        """
        if frame1.shape != frame2.shape:
            return frame2

        h, w = frame1.shape[:2]
        
        # 1. Quick static check
        if np.array_equal(frame1[::100, ::100], frame2[::100, ::100]):
            return frame1

        # 2. Prepare Grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # 3. Low-res flow calculation
        scale = 0.25
        small1 = cv2.resize(gray1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small2 = cv2.resize(gray2, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        flow = self.dis.calc(small1, small2, None)
        
        # 4. Stabilize Flow (Median filter to remove 'jelly' noise)
        flow = cv2.medianBlur(flow, 3)
        
        # 5. Scale and resize flow
        flow = flow * (1.0 / scale)
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 6. Bilateral Warping Logic
        # Instead of 1.0 warp, we do 0.5 for both directions
        if h != self.last_h or w != self.last_w:
            self.map_x, self.map_y = np.meshgrid(np.arange(w), np.arange(h))
            self.map_x = self.map_x.astype(np.float32)
            self.map_y = self.map_y.astype(np.float32)
            self.last_h, self.last_w = h, w
        
        # Motion magnitude for adaptive blending
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Forward warp map (frame1 -> mid)
        m1_x = self.map_x + flow[..., 0] * 0.5
        m1_y = self.map_y + flow[..., 1] * 0.5
        
        # Backward warp map (frame2 -> mid)
        m2_x = self.map_x - flow[..., 0] * 0.5
        m2_y = self.map_y - flow[..., 1] * 0.5
        
        # Warp both
        inter1 = cv2.remap(frame1, m1_x, m1_y, cv2.INTER_LINEAR)
        inter2 = cv2.remap(frame2, m2_x, m2_y, cv2.INTER_LINEAR)
        
        # Adaptive Blending: 
        # In regions with extreme motion, favor cross-fade to hide artifacts
        # In low motion, use high warp weight
        blend_mask = np.clip(mag / 20.0, 0, 1) # Threshold 20px
        
        # Final combine
        # We blend inter1 and inter2 (Bilateral)
        # and then slightly blend with original cross-fade if motion is too high
        combined = cv2.addWeighted(inter1, 0.5, inter2, 0.5, 0)
        
        if np.max(blend_mask) > 0.1:
            cross_fade = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
            # Use blend_mask to choose between motion-compensated and cross-fade
            # (Simplified for CPU speed)
            mask_3c = cv2.merge([blend_mask, blend_mask, blend_mask])
            final = combined * (1.0 - mask_3c * 0.4) + cross_fade * (mask_3c * 0.4)
            return final.astype(np.uint8)
        
        return combined

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
