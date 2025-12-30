import cv2
import numpy as np
import pygame
import threading
import time
import ctypes
from queue import Queue
from capture import ScreenCapture
from engine import RIFEEngine, RIFEONNXEngine
from ui import GameSelectorUI
from selector import WindowSelector
from filters import AMDFilters, NvidiaAIUpscaler
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
        self.process_queue = Queue(maxsize=3)
        self.display_queue = Queue(maxsize=20) # Buffer for smooth display
        
        # Stats
        self.frame_count = 0
        self.start_time = 0
        self.current_fps = 0
        
        # Window & Performance management
        self.last_rect = None
        self.show_fps = True
        self.hotkey_cooldown = 0
        self.scale_factor = 1.0
        self.upscale_algo = cv2.INTER_LINEAR
        self.sharpness = 0.3
        self.internal_res = (800, 600) # Default target resolution for processing
        self.display_dim = (1280, 720) # Actual output dimensions
        self.fsr_mode = False # Toggle for AMD CAS/EASU
        self.ai_mode = False # Toggle for NVIDIA AI SuperRes
        self.ai_upscaler = None

    def capture_worker(self):
        print("Capture worker started")
        last_frame = None
        # Target capture is exactly half the display FPS
        capture_interval = 1.0 / (self.target_fps / 2) if self.target_fps > 0 else 1.0/30.0
        
        last_capture_time = time.perf_counter()
        
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
            
            now = time.perf_counter()
            if now - last_capture_time < capture_interval:
                time.sleep(0.001)
                continue
            
            last_capture_time = now
            frame = self.capture.capture_frame()
            
            if frame is not None:
                # Optimized duplicate check
                is_duplicate = False
                if last_frame is not None:
                    try:
                        # Slice check is very fast
                        if np.array_equal(frame[10:30:2, 10:30:2], last_frame[10:30:2, 10:30:2]):
                            if np.array_equal(frame[::60, ::60], last_frame[::60, ::60]):
                                is_duplicate = True
                    except: pass
                
                if not is_duplicate:
                    if self.capture_queue.full():
                        try: self.capture_queue.get_nowait()
                        except: pass
                    self.capture_queue.put(frame)
                    last_frame = frame

    def processing_worker(self):
        print("Processing worker started")
        last_frame = None
        while self.running:
            if not self.capture_queue.empty():
                current_frame = self.capture_queue.get()
                
                # Internal Scaling: Downscale frame if it's too large for processing
                h, w = current_frame.shape[:2]
                if w > self.internal_res[0] or h > self.internal_res[1]:
                    current_frame = cv2.resize(current_frame, self.internal_res, interpolation=cv2.INTER_LINEAR)

                if self.fg_enabled and last_frame is not None:
                    inter_frame = self.engine.interpolate(last_frame, current_frame)
                    
                    self.process_queue.put(inter_frame)
                    self.process_queue.put(current_frame)
                else:
                    self.process_queue.put(current_frame)
                
                last_frame = current_frame
            else:
                time.sleep(0.001)

    def post_processing_worker(self):
        print("Post-processing worker started")
        while self.running:
            if not self.process_queue.empty():
                frame = self.process_queue.get()
                
                # Push to display
                if self.ai_mode and self.ai_upscaler:
                    # AI Reconstruction
                    frame = self.ai_upscaler.upscale(frame)
                    # Final fit to display if AI output differs
                    if frame.shape[1] != self.display_dim[0] or frame.shape[0] != self.display_dim[1]:
                        frame = cv2.resize(frame, self.display_dim, interpolation=cv2.INTER_LINEAR)
                elif self.fsr_mode:
                    # High-Speed Resizing (EASU-style if FSR mode enabled)
                    if frame.shape[1] != self.display_dim[0] or frame.shape[0] != self.display_dim[1]:
                        frame = AMDFilters.apply_easu(frame, self.display_dim)
                    else:
                        frame = AMDFilters.apply_cas(frame, self.sharpness)
                else:
                    if frame.shape[1] != self.display_dim[0] or frame.shape[0] != self.display_dim[1]:
                        frame = cv2.resize(frame, self.display_dim, interpolation=self.upscale_algo)
                    
                    # Apply simple sharpening (Fallback)
                    if self.sharpness > 0:
                        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
                        frame = cv2.addWeighted(frame, 1.0 + self.sharpness, blurred, -self.sharpness, 0)

                if self.display_queue.full():
                    try: self.display_queue.get_nowait()
                    except: pass
                self.display_queue.put(frame)
            else:
                # Slight sleep to reduce CPU usage when idle
                time.sleep(0.0005)

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
        elif "FSR" in algo_val:
            self.fsr_mode = True
        elif "AI" in algo_val:
            self.ai_mode = True
            self.ai_upscaler = NvidiaAIUpscaler()
            
        self.fg_enabled = self.target_window.get("fg_enabled", True)
        
        self.sharpness = self.target_window["sharpness"] / 100.0 * 2.0 # Scale 0-100 to 0.0-2.0
        self.ultra_smooth = self.target_window.get("ultra_smooth", False)
        if self.target_window.get("engine_type") == "AI (RIFE ONNX)":
            self.engine = RIFEONNXEngine()
        else:
            self.engine = RIFEEngine()
            
        self.engine.set_high_precision(self.ultra_smooth) if hasattr(self.engine, 'set_high_precision') else None

        # Initial region
        rect = WindowSelector.get_window_rect(self.target_window["hwnd"])
        self.capture.region = rect
        
        # Performance tuning: Set internal resolution limit
        # Performance Mode (Alta Res) uses 1280x720, Standard uses 800x600
        max_w, max_h = (1280, 720) if self.target_window.get("performance_mode") else (800, 600)
        
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        if w > max_w: self.internal_res = (max_w, max_h)
        else: self.internal_res = (w, h)
        
        # Initial display dimensions
        if self.scale_factor == -1:
            info = win32api.GetSystemMetrics(win32con.SM_CXSCREEN), win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            self.display_dim = info
        else:
            self.display_dim = (int(w * self.scale_factor), int(h * self.scale_factor))

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
        while not self.process_queue.empty(): self.process_queue.get()
        while not self.display_queue.empty(): self.display_queue.get()
        
        t_cap = threading.Thread(target=self.capture_worker, daemon=True)
        t_proc = threading.Thread(target=self.processing_worker, daemon=True)
        t_post = threading.Thread(target=self.post_processing_worker, daemon=True)
        t_cap.start()
        t_proc.start()
        t_post.start()
        
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
                    # Busy-wait for the last 1ms for sub-ms precision
                    if frame_interval - (now - last_display_time) < 0.001:
                        pass # Busy wait
                    else:
                        time.sleep(0.0005)
                    continue

                # Buffer check: wait for at least 2 frames to be ready to absorb jitter
                # but only if we are not already in a high-latency state
                if self.display_queue.qsize() < 2 and self.frame_count > 0:
                    time.sleep(0.001)
                    continue

                if not self.display_queue.empty():
                    frame = self.display_queue.get()
                    last_display_time = now
                    
                    # Window sync (minimal overhead)
                    try:
                        t_rect = WindowSelector.get_window_rect(self.target_window["hwnd"])
                        if self.last_rect != t_rect:
                            t_w, t_h = t_rect[2] - t_rect[0], t_rect[3] - t_rect[1]
                            
                            # Update internal res cap immediately
                            max_w, max_h = (1280, 720) if self.target_window.get("performance_mode") else (800, 600)
                            if t_w > max_w: self.internal_res = (max_w, max_h)
                            else: self.internal_res = (t_w, t_h)

                            if self.scale_factor != -1:
                                d_w, d_h = int(t_w * self.scale_factor), int(t_h * self.scale_factor)
                                self.display_dim = (d_w, d_h)
                                if screen.get_width() != d_w or screen.get_height() != d_h:
                                    screen = pygame.display.set_mode((d_w, d_h), pygame.NOFRAME)
                                    hwnd_p = pygame.display.get_wm_info()["window"]
                                    ctypes.windll.user32.SetWindowDisplayAffinity(hwnd_p, 0x00000011)
                                    ex = win32gui.GetWindowLong(hwnd_p, win32con.GWL_EXSTYLE)
                                    win32gui.SetWindowLong(hwnd_p, win32con.GWL_EXSTYLE, ex | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)
                                win32gui.SetWindowPos(hwnd_p, win32con.HWND_TOPMOST, t_rect[0], t_rect[1], d_w, d_h, win32con.SWP_NOACTIVATE)
                            self.last_rect = t_rect
                    except: pass

                    # Blit and Flip (Now ultra-fast as frame is pre-processed)
                    surface = pygame.image.frombuffer(frame.tobytes(), self.display_dim, 'RGB')
                    screen.blit(surface, (0, 0))
                    
                    # Hotkeys & Stats
                    if win32api.GetAsyncKeyState(0x79) & 0x8000: # F10
                        if time.time() - self.hotkey_cooldown > 0.3:
                            self.show_fps = not self.show_fps
                            self.hotkey_cooldown = time.time()
                    
                    if win32api.GetAsyncKeyState(0x78) & 0x8000: # F9
                        if time.time() - self.hotkey_cooldown > 0.3:
                            self.fsr_mode = not self.fsr_mode
                            self.hotkey_cooldown = time.time()
                            print(f"FSR Mode: {'ON' if self.fsr_mode else 'OFF'}")
                    
                    if self.show_fps:
                        status_color = (0, 255, 0)
                        fps_text = font.render(f"FPS: {self.current_fps:.1f}", True, status_color)
                        fsr_text = font.render(f"(F9) FSR: {'ON' if self.fsr_mode else 'OFF'}", True, (255, 200, 0) if self.fsr_mode else (150, 150, 150))
                        
                        ai_text = font.render(f"AI SuperRes: {'ON' if self.ai_mode else 'OFF'}", True, (0, 255, 200) if self.ai_mode else (150, 150, 150))

                        # Extra status for new modes
                        mode_text_str = "STD"
                        if self.ultra_smooth: mode_text_str = "SMOOTH"
                        mode_text = font.render(f"Mode: {mode_text_str}", True, (0, 200, 255))

                        # Draw status box
                        bg_rect = pygame.Rect(10, 10, 200, 110)
                        pygame.draw.rect(screen, (0, 0, 0), bg_rect)
                        pygame.draw.rect(screen, (50, 50, 50), bg_rect, 2)
                        
                        screen.blit(fps_text, (20, 15))
                        screen.blit(fsr_text, (20, 40))
                        screen.blit(ai_text, (20, 65))
                        screen.blit(mode_text, (20, 90))

                    pygame.display.flip()
                    
                    self.frame_count += 1
                    if self.frame_count % 30 == 0:
                        t_now = time.time()
                        self.current_fps = 30 / (t_now - self.start_time)
                        self.start_time = t_now
                else:
                    time.sleep(0.0005)

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
