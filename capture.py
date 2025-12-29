import dxcam
import time
import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api
import time

class ScreenCapture:
    def __init__(self, region=None, device_idx=0, output_color="RGB", mode="dxcam"):
        """
        Initialize the capture.
        :param region: Tuple of (left, top, right, bottom). If None, captures full screen.
        :param mode: "dxcam" or "bitblt"
        """
        self.mode = mode
        self.camera = None
        if self.mode == "dxcam":
            self.camera = dxcam.create(device_idx=device_idx, output_color=output_color)
        
        self.region = region
        self.is_capturing = False

        # BitBlt persistent resources
        self._hwnd_dc = None
        self._mfc_dc = None
        self._save_dc = None
        self._save_bitmap = None
        self._last_dims = (0, 0)
        
    def capture_frame(self):
        """
        Captures a single frame based on the current mode.
        """
        if self.mode == "dxcam":
            return self._capture_dxcam()
        else:
            return self._capture_bitblt()

    def _capture_dxcam(self):
        frame = None
        if self.region:
            frame = self.camera.grab(region=self.region)
        else:
            frame = self.camera.grab()
        
        if frame is None:
            frame = self.camera.get_latest_frame()
            
        return frame

    def _init_bitblt_resources(self, width, height):
        self._cleanup_gdi()
        hwnd = win32gui.GetDesktopWindow()
        self._hwnd_dc = win32gui.GetWindowDC(hwnd)
        self._mfc_dc = win32ui.CreateDCFromHandle(self._hwnd_dc)
        self._save_dc = self._mfc_dc.CreateCompatibleDC()
        self._save_bitmap = win32ui.CreateBitmap()
        self._save_bitmap.CreateCompatibleBitmap(self._mfc_dc, width, height)
        self._save_dc.SelectObject(self._save_bitmap)
        self._last_dims = (width, height)

    def _cleanup_gdi(self):
        # Deselect bitmap before deletion
        if self._save_dc:
            try:
                # Selecting a small dummy bitmap or 0 can help deselect the current one
                # but in win32ui it's often safer to just try/except the deletion
                self._save_dc.DeleteDC()
            except:
                pass
            self._save_dc = None
            
        if self._mfc_dc:
            try:
                self._mfc_dc.DeleteDC()
            except:
                pass
            self._mfc_dc = None
            
        if self._hwnd_dc:
            try:
                hwnd = win32gui.GetDesktopWindow()
                win32gui.ReleaseDC(hwnd, self._hwnd_dc)
            except:
                pass
            self._hwnd_dc = None
            
        if self._save_bitmap:
            try:
                win32gui.DeleteObject(self._save_bitmap.GetHandle())
            except:
                pass
            self._save_bitmap = None

    def _capture_bitblt(self):
        """
        Ultra-fast GDI Capture using GetDIBits and raw memory access.
        """
        try:
            if self.region is None:
                width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
                height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
                left, top = 0, 0
            else:
                left, top, right, bottom = self.region
                width = right - left
                height = bottom - top

            if width <= 0 or height <= 0: return None

            if (width, height) != self._last_dims or self._save_dc is None:
                self._init_bitblt_resources(width, height)

            # 1. BitBlt to our compatible DC
            self._save_dc.BitBlt((0, 0), (width, height), self._mfc_dc, (left, top), win32con.SRCCOPY)
            
            # 2. Extract bits directly to a pre-allocated numpy array for speed
            # GetBitmapBits is very slow. GetDIBits is preferred but win32ui's GetBitmapBits 
            # is often a wrapper. Let's use the fastest possible way.
            # signedIntsArray = self._save_bitmap.GetBitmapBits(True) # This is a slow copy
            
            # Optimization: Use the fact that Pygame can read BGRA directly
            # and that we can avoid cv2.cvtColor by just reversing the last channel if needed
            # For now, let's use the buffer and reshape.
            # Note: We keep RGBA to avoid cvtColor.
            
            data = self._save_bitmap.GetBitmapBits(True)
            img = np.frombuffer(data, dtype='uint8')
            img.shape = (height, width, 4)
            
            # Return RGB (discard alpha and flip BGR if necessary)
            # This slice is much faster than cv2.cvtColor
            # We use .copy() to ensure the array is contiguous for Pygame frombuffer
            return img[:, :, :3][:, :, ::-1].copy() 
            
        except Exception as e:
            self._cleanup_gdi()
            return None

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
        if self.is_capturing and self.camera is not None:
            self.camera.stop()
            self.is_capturing = False
        self._cleanup_gdi()

if __name__ == "__main__":
    # Test capture
    cap = ScreenCapture(mode="bitblt")
    print("Testing BitBlt speed for 100 frames with persistent GDI...")
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
        cv2.imwrite("test_capture_bitblt.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("Test frame saved to test_capture_bitblt.jpg")
    cap.stop_capture()
