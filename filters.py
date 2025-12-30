import cv2
import numpy as np
import os
import requests

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

class NvidiaAIUpscaler:
    def __init__(self, model_path="models/fsrcnn_x2.onnx"):
        self.model_path = model_path
        self.session = None
        self._download_model_if_missing()
        self._init_session()

    def _download_model_if_missing(self):
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"Downloading AI model to {self.model_path}...")
            # Using a public lightweight FSRCNN ONNX model
            url = "https://github.com/onuralpszener/FSRCNN-PyTorch/raw/master/fsrcnn_x2.onnx"
            try:
                r = requests.get(url, allow_redirects=True)
                with open(self.model_path, 'wb') as f:
                    f.write(r.content)
                print("Model downloaded successfully.")
            except Exception as e:
                print(f"Failed to download model: {e}")

    def _init_session(self):
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            print(f"Inference session initialized with {self.session.get_providers()}")
        except Exception as e:
            print(f"Error initializing ONNX session: {e}")

    def upscale(self, img):
        if self.session is None: return img
        
        # Pre-process: 1. Convert to YCrCb (SR usually works on Y channel)
        # However, for simplicity and color, we'll process RGB if model supports it
        # The FSRCNN model usually expects (B, 1, H, W) for Y-channel
        # or (B, 3, H, W) for RGB.
        
        # Resize to model input if needed (usually AI is fixed scale)
        # FSRCNN is x2. 
        # For a general solution, AI is harder. Let's assume the user wants x2
        # or we just use it for the core upscaling.
        
        h, w = img.shape[:2]
        img_f = img.astype(np.float32) / 255.0
        
        # Prepare for ONNX: (H, W, C) -> (1, C, H, W)
        input_tensor = np.transpose(img_f, (2, 0, 1))[np.newaxis, ...]
        
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_tensor})[0]
        
        # Post-process: (1, C, H, W) -> (H, W, C)
        output = np.transpose(output[0], (1, 2, 0))
        output = (np.clip(output, 0, 1) * 255).astype(np.uint8)
        
        return output
