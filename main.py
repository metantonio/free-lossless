import cv2
import numpy as np
import pygame
import threading
import time
from queue import Queue
from capture import ScreenCapture
from engine import RIFEEngine
from ui import GameSelectorUI
from selector import WindowSelector

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
        
        # Initial region
        rect = WindowSelector.get_window_rect(self.target_window["hwnd"])
        self.capture.region = rect
        print(f"Targeting: {self.target_window['title']} using {self.target_window['mode']} at {rect}")
        return True

    def run(self):
        if not self.select_game():
            print("No game selected. Exiting.")
            return

        pygame.init()
        # Initial size from window
        rect = self.capture.region
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        
        # Ensure dimensions are valid
        if w <= 0 or h <= 0:
            print("Invalid window dimensions.")
            return

        screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        pygame.display.set_caption(f"FG: {self.target_window['title']}")
        clock = pygame.time.Clock()
        
        self.running = True
        self.start_time = time.time()
        
        t_cap = threading.Thread(target=self.capture_worker, daemon=True)
        t_proc = threading.Thread(target=self.processing_worker, daemon=True)
        t_cap.start()
        t_proc.start()
        
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                
                if not self.display_queue.empty():
                    frame = self.display_queue.get()
                    
                    # Handle window resizing
                    c_h, c_w, _ = frame.shape
                    if (screen.get_width() != c_w or screen.get_height() != c_h):
                        screen = pygame.display.set_mode((c_w, c_h), pygame.RESIZABLE)

                    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    screen.blit(surface, (0, 0))
                    pygame.display.flip()
                    
                    self.frame_count += 1
                    if self.frame_count % 60 == 0:
                        elapsed = time.time() - self.start_time
                        self.current_fps = self.frame_count / elapsed
                        
                        # Add queue sizes for debugging
                        cap_q = self.capture_queue.qsize()
                        dis_q = self.display_queue.qsize()
                        
                        pygame.display.set_caption(f"FG: {self.target_window['title']} - FPS: {self.current_fps:.2f} | Q: {cap_q}/{dis_q}")
                        print(f"Stats: FPS={self.current_fps:.2f}, CaptureQueue={cap_q}, DisplayQueue={dis_q}")
                
                clock.tick(self.target_fps * 1.5) # Allow display to be slightly faster than intake
        finally:
            self.running = False
            self.capture.stop_capture()
            pygame.quit()

if __name__ == "__main__":
    app = FrameGenerationApp(target_fps=60)
    app.run()
