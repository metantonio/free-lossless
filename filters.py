import cv2
import numpy as np

class AMDFilters:
    @staticmethod
    def apply_cas(img, sharpness=0.5):
        """
        Improved AMD FidelityFX Contrast Adaptive Sharpening (CAS).
        Optimized implementation that uses local min/max for adaptive weighting.
        """
        if sharpness <= 0: return img
        
        img_f = img.astype(np.float32) / 255.0
        
        # We need to process each channel or luminance
        # Simplified but effective adaptive weighting
        
        # 3x3 local min/max (approx by erosions/dilations)
        kernel = np.ones((3,3), np.uint8)
        local_min = cv2.erode(img_f, kernel)
        local_max = cv2.dilate(img_f, kernel)
        
        # AMD CAS Weight logic: 
        # weight = sqrt(min(l, 1-max) / max) * sharpness
        # We'll use a fast version:
        # Avoid division by zero
        local_max = np.maximum(local_max, 1e-5)
        
        # Calculate adaptive weight per pixel
        # This weight is lower in high-contrast (edges) to prevent halos
        w = np.sqrt(np.minimum(local_min, 1.0 - local_max) / local_max) * (sharpness * 0.5)
        
        # Standard sharpening kernel based on the weight
        # [ 0 -w  0 ]
        # [-w 1+4w -w]
        # [ 0 -w  0 ]
        
        sharp_kernel = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        details = cv2.filter2D(img_f, -1, sharp_kernel)
        
        # Apply the adaptive weight to the high-frequency details
        # Broadcast w (H,W,C) if needed, but erode/dilate handles channels
        result = img_f + details * w
        
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def apply_easu(img, target_dim):
        """
        Simplified Edge-Adaptive Spatial Upsampling (EASU).
        Uses Lanczos4 with a pre-pass edge boost.
        """
        h, w = img.shape[:2]
        if (w, h) == target_dim: return img
        
        # Pass 1: High-quality Lanczos upscale
        upscaled = cv2.resize(img, target_dim, interpolation=cv2.INTER_LANCZOS4)
        
        # Pass 2: Light CAS to restore edge definition lost in scaling
        return AMDFilters.apply_cas(upscaled, sharpness=0.3)
