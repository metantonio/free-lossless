import cv2
import numpy as np
import pygame
import threading
import time
import ctypes
from queue import Queue
from capture import ScreenCapture
from engine import RIFEEngine
from ui import GameSelectorUI
from selector import WindowSelector
import win32gui
import win32con
import win32api

class FrameGenerationApp:
    def __init__(self, target_fps=60):
        self.target_fps = target_fps
        self.capture = ScreenCapture()
        self.engine = RIFEEngine()
        self.running = False
        self.target_window = None
        
        # Queues for pipeline
        self.capture_queue = Queue(maxsize=3)
        self.display_queue = Queue(maxsize=5)
        
        # Stats
        self.frame_count = 0
        self.start_time = 0
        self.current_fps = 0
        
        # Window management
        self.last_rect = None
        self.show_fps = True
        self.hotkey_cooldown = 0
        self.scale_factor = 1.0
        self.upscale_algo = cv2.INTER_LINEAR
        self.sharpness = 0.3 # 0.0 to 1.0 ideally

    def capture_worker(self):
        print("Capture worker started")
        last_frame = None
        # Capture at half the target FPS because we interpolate 1 extra frame
        capture_interval = 1.0 / (self.target_fps / 2) if self.target_fps > 0 else 1.0/30.0
        
        while self.running:
            # Update region based on window position
            if self.target_window:
                try:
                    rect = WindowSelector.get_window_rect(self.target_window["hwnd"])
                    self.capture.region = rect # Update region
                except:
                    print("Lost window, stopping...")
                    self.running = False
                    break
            
            start_time = time.time()
            frame = self.capture.capture_frame()
            
            if frame is not None:
                # Check for duplicates (quick check)
                is_duplicate = False
                if last_frame is not None:
                    # Sample some pixels for speed
                    try:
                        if np.array_equal(frame[10:20, 10:20], last_frame[10:20, 10:20]):
                            # If samples match, do a full check or just trust the sample
                            # Full check is safer but slower. Let's do a slightly larger sample.
                            if np.array_equal(frame[::50, ::50], last_frame[::50, ::50]):
                                is_duplicate = True
                    except: pass
                
                if not is_duplicate:
                    if self.capture_queue.full():
                        try: self.capture_queue.get_nowait()
                        except: pass
                    self.capture_queue.put(frame)
                    last_frame = frame
                
            elapsed = time.time() - start_time
            sleep_time = max(0, capture_interval - elapsed)
            time.sleep(sleep_time)

    def processing_worker(self):
        print("Processing worker started")
        last_frame = None
        while self.running:
            if not self.capture_queue.empty():
                current_frame = self.capture_queue.get()
                
                if last_frame is not None:
                    inter_frame = self.engine.interpolate(last_frame, current_frame)
                    
                    if self.display_queue.full():
                        self.display_queue.get()
                    self.display_queue.put(inter_frame)
                    
                    if self.display_queue.full():
                        self.display_queue.get()
                    self.display_queue.put(current_frame)
                else:
                    self.display_queue.put(current_frame)
                
                last_frame = current_frame
            else:
                time.sleep(0.001)

    def select_game(self):
        ui = GameSelectorUI()
        self.target_window = ui.get_selection()
        if not self.target_window:
            return False
        
        # Re-initialize capture with correct mode
        self.capture = ScreenCapture(mode=self.target_window["mode"])
        self.target_fps = self.target_window["fps"]
        
        # Scaling config
        scale_val = self.target_window["scale"]
        if scale_val == "Fullscreen":
            self.scale_factor = -1 # Special flag for fullscreen
        else:
            self.scale_factor = float(scale_val)
            
        algo_val = self.target_window["algo"]
        if algo_val == "Bilinear": self.upscale_algo = cv2.INTER_LINEAR
        elif algo_val == "Bicubic": self.upscale_algo = cv2.INTER_CUBIC
        elif algo_val == "Lanczos": self.upscale_algo = cv2.INTER_LANCZOS4
        
        self.sharpness = self.target_window["sharpness"] / 100.0 * 2.0 # Scale 0-100 to 0.0-2.0

        # Initial region
        rect = WindowSelector.get_window_rect(self.target_window["hwnd"])
        self.capture.region = rect
        print(f"Targeting: {self.target_window['title']} | Mode: {self.target_window['mode']} | FPS: {self.target_fps} | Scale: {scale_val}")
        print("Press F11 to stop, F10 to toggle FPS.")
        return True

    def run(self):
        if not self.select_game():
            return False

        pygame.init()
        # Initial capture size
        rect = self.capture.region
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        
        # Calculate display size
        if self.scale_factor == -1: # Fullscreen
            info = pygame.display.Info()
            d_w, d_h = info.current_w, info.current_h
            display_flags = pygame.NOFRAME | pygame.FULLSCREEN
        else:
            d_w, d_h = int(w * self.scale_factor), int(h * self.scale_factor)
            display_flags = pygame.NOFRAME

        if d_w <= 0 or d_h <= 0:
            print("Invalid window dimensions.")
            return True

        # Setup Borderless Window
        screen = pygame.display.set_mode((d_w, d_h), display_flags)
        pygame.display.set_caption("FG Overlay")
        
        # FPS Font
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 24, bold=True)
        
        hwnd_pygame = pygame.display.get_wm_info()["window"]
        
        # 1. Exclude this window from screen capture (Win 10+)
        try:
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd_pygame, WDA_EXCLUDEFROMCAPTURE)
        except Exception as e:
            print(f"Capture exclusion not available: {e}")

        # 2. Make Click-Through
        ex_style = win32gui.GetWindowLong(hwnd_pygame, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(hwnd_pygame, win32con.GWL_EXSTYLE, ex_style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)

        # 3. Set to Always On Top
        if self.scale_factor == -1: # Fullscreen
            win32gui.SetWindowPos(hwnd_pygame, win32con.HWND_TOPMOST, 0, 0, d_w, d_h, win32con.SWP_SHOWWINDOW)
        else:
            win32gui.SetWindowPos(hwnd_pygame, win32con.HWND_TOPMOST, rect[0], rect[1], d_w, d_h, win32con.SWP_SHOWWINDOW)
        self.last_rect = rect

        # Increase process priority for better smoothness
        try:
            import psutil
            p = psutil.Process()
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        except: pass

        clock = pygame.time.Clock()
        self.running = True
        self.start_time = time.time()
        
        # Clear queues
        while not self.capture_queue.empty(): self.capture_queue.get()
        while not self.display_queue.empty(): self.display_queue.get()
        
        t_cap = threading.Thread(target=self.capture_worker, daemon=True)
        t_proc = threading.Thread(target=self.processing_worker, daemon=True)
        t_cap.start()
        t_proc.start()
        
        frame_interval = 1.0 / self.target_fps
        last_display_time = time.perf_counter()

        try:
            while self.running:
                # 4. Check for Global Hotkey (F11)
                # VK_F11 = 0x7A
                if win32api.GetAsyncKeyState(0x7A) & 0x8000:
                    print("Stop key pressed. Returning to menu...")
                    self.running = False
                    break

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                
                # Precision Pacing Logic
                now = time.perf_counter()
                if now - last_display_time < frame_interval:
                    # Not yet time for next frame
                    time.sleep(0) # Yield
                    continue

                if not self.display_queue.empty():
                    frame = self.display_queue.get()
                    last_display_time = now
                    
                    # Window sync (position/size)
                    try:
                        t_rect = WindowSelector.get_window_rect(self.target_window["hwnd"])
                        t_cap_w, t_cap_h = t_rect[2] - t_rect[0], t_rect[3] - t_rect[1]
                        if self.scale_factor == -1: t_w, t_h = screen.get_size()
                        else: t_w, t_h = int(t_cap_w * self.scale_factor), int(t_cap_h * self.scale_factor)
                        
                        if self.last_rect != t_rect:
                            if self.scale_factor != -1:
                                if screen.get_width() != t_w or screen.get_height() != t_h:
                                    screen = pygame.display.set_mode((t_w, t_h), pygame.NOFRAME)
                                    hwnd_p = pygame.display.get_wm_info()["window"]
                                    try:
                                        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd_p, 0x00000011)
                                        ex = win32gui.GetWindowLong(hwnd_p, win32con.GWL_EXSTYLE)
                                        win32gui.SetWindowLong(hwnd_p, win32con.GWL_EXSTYLE, ex | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)
                                    except: pass
                                win32gui.SetWindowPos(hwnd_p, win32con.HWND_TOPMOST, t_rect[0], t_rect[1], t_w, t_h, win32con.SWP_NOACTIVATE)
                            self.last_rect = t_rect
                        else:
                            win32gui.SetWindowPos(hwnd_pygame, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                                                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
                    except: pass

                    # Processing
                    if self.scale_factor != 1.0 or self.scale_factor == -1:
                        frame = cv2.resize(frame, (t_w, t_h), interpolation=self.upscale_algo)
                    if self.sharpness > 0:
                        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
                        frame = cv2.addWeighted(frame, 1.0 + self.sharpness, blurred, -self.sharpness, 0)

                    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    screen.blit(surface, (0, 0))
                    
                    # FPS & Overlay
                    if win32api.GetAsyncKeyState(0x79) & 0x8000:
                        if time.time() - self.hotkey_cooldown > 0.3:
                            self.show_fps = not self.show_fps
                            self.hotkey_cooldown = time.time()
                    
                    if self.show_fps:
                        fps_text = font.render(f"FPS: {self.current_fps:.1f}", True, (0, 255, 0))
                        bg_rect = fps_text.get_rect(topleft=(10, 10))
                        pygame.draw.rect(screen, (0, 0, 0), bg_rect.inflate(10, 5))
                        screen.blit(fps_text, (10, 10))

                    pygame.display.flip()
                    
                    self.frame_count += 1
                    if self.frame_count % 30 == 0:
                        t_now = time.time()
                        self.current_fps = 30 / (t_now - self.start_time)
                        self.start_time = t_now
                else:
                    # Queue empty, wait a tiny bit to avoid busy loop stutter
                    time.sleep(0.001)

            # Removed clock.tick to rely on perf_counter pacing
        finally:
            self.running = False
            self.capture.stop_capture()
            pygame.quit()
        
        return True

if __name__ == "__main__":
    app = FrameGenerationApp(target_fps=60)
    while True:
        if not app.run():
            break
        print("Waiting for next selection...")
        time.sleep(0.5)
