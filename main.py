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

    def capture_worker(self):
        print("Capture worker started")
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
            
            frame = self.capture.capture_frame()
            if frame is not None:
                if self.capture_queue.full():
                    self.capture_queue.get()
                self.capture_queue.put(frame)
            
            # FPS control for capture
            time.sleep(1/(self.target_fps)) 

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
        
        # Initial region
        rect = WindowSelector.get_window_rect(self.target_window["hwnd"])
        self.capture.region = rect
        print(f"Targeting: {self.target_window['title']} using {self.target_window['mode']} at {self.target_fps} FPS")
        print("Press F11 to stop generation and return to menu.")
        return True

    def run(self):
        if not self.select_game():
            return False

        pygame.init()
        # Initial size
        rect = self.capture.region
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        
        if w <= 0 or h <= 0:
            print("Invalid window dimensions.")
            return True

        # Setup Borderless Window
        screen = pygame.display.set_mode((w, h), pygame.NOFRAME)
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
        win32gui.SetWindowPos(hwnd_pygame, win32con.HWND_TOPMOST, rect[0], rect[1], w, h, win32con.SWP_SHOWWINDOW)
        self.last_rect = rect

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
                
                if not self.display_queue.empty():
                    frame = self.display_queue.get()
                    
                    # Sync with target window position/size
                    try:
                        t_rect = WindowSelector.get_window_rect(self.target_window["hwnd"])
                        t_w, t_h = t_rect[2] - t_rect[0], t_rect[3] - t_rect[1]
                        
                        # Only update window pos if target moved to avoid jitter/latency
                        if self.last_rect != t_rect:
                            if (screen.get_width() != t_w or screen.get_height() != t_h):
                                screen = pygame.display.set_mode((t_w, t_h), pygame.NOFRAME)
                                hwnd_pygame = pygame.display.get_wm_info()["window"]
                                # Re-apply styles
                                try:
                                    ctypes.windll.user32.SetWindowDisplayAffinity(hwnd_pygame, 0x00000011)
                                    ex_style = win32gui.GetWindowLong(hwnd_pygame, win32con.GWL_EXSTYLE)
                                    win32gui.SetWindowLong(hwnd_pygame, win32con.GWL_EXSTYLE, ex_style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)
                                except: pass
                            
                            win32gui.SetWindowPos(hwnd_pygame, win32con.HWND_TOPMOST, t_rect[0], t_rect[1], t_w, t_h, win32con.SWP_NOACTIVATE)
                            self.last_rect = t_rect
                        else:
                            # Re-assert TopMost
                            win32gui.SetWindowPos(hwnd_pygame, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                                                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
                    except:
                        pass

                    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    screen.blit(surface, (0, 0))
                    
                    # 5. Render FPS Overlay if enabled
                    # Toggle with F10 (VK_F10 = 0x79)
                    if win32api.GetAsyncKeyState(0x79) & 0x8000:
                        if time.time() - self.hotkey_cooldown > 0.3:
                            self.show_fps = not self.show_fps
                            self.hotkey_cooldown = time.time()
                    
                    if self.show_fps:
                        fps_text = font.render(f"FPS: {self.current_fps:.1f}", True, (0, 255, 0))
                        # Simple background for readability
                        bg_rect = fps_text.get_rect(topleft=(10, 10))
                        pygame.draw.rect(screen, (0, 0, 0), bg_rect.inflate(10, 5))
                        screen.blit(fps_text, (10, 10))
                    
                    pygame.display.flip()

                    self.frame_count += 1
                    if self.frame_count % 60 == 0:
                        elapsed = time.time() - self.start_time
                        self.current_fps = self.frame_count / elapsed
                        cap_q = self.capture_queue.qsize()
                        dis_q = self.display_queue.qsize()
                        print(f"Stats: FPS={self.current_fps:.2f}, CaptureQueue={cap_q}, DisplayQueue={dis_q}")
                
                clock.tick(self.target_fps * 2) 
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
