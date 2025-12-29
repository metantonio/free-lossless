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

    def _capture_bitblt(self):
        """
        Captures using GDI BitBlt. Better for old games like Mu Online.
        """
        if self.region is None:
            width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            left, top = 0, 0
        else:
            left, top, right, bottom = self.region
            width = right - left
            height = bottom - top

        hwnd = win32gui.GetDesktopWindow()
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)

        saveDC.BitBlt((0, 0), (width, height), mfcDC, (left, top), win32con.SRCCOPY)

        signedIntsArray = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

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
