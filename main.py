import cv2
import numpy as np
import pygame
import threading
import time
from queue import Queue
from capture import ScreenCapture
from engine import RIFEEngine

class FrameGenerationApp:
    def __init__(self, target_fps=60):
        self.target_fps = target_fps
        self.capture = ScreenCapture()
        self.engine = RIFEEngine()
        self.running = False
        
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
            frame = self.capture.capture_frame()
            if frame is not None:
                if self.capture_queue.full():
                    self.capture_queue.get()
                self.capture_queue.put(frame)
            time.sleep(1/self.target_fps) # Capture at half or full rate? Let's say half of target

    def processing_worker(self):
        print("Processing worker started")
        last_frame = None
        while self.running:
            if not self.capture_queue.empty():
                current_frame = self.capture_queue.get()
                
                if last_frame is not None:
                    # Generate intermediate frame
                    inter_frame = self.engine.interpolate(last_frame, current_frame)
                    
                    # Push Frame I and Frame B
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

    def run(self):
        pygame.init()
        # Create a window same size as screen (or region)
        # For prototype, let's just use 1280x720 or detect capture size
        test_frame = self.capture.capture_frame()
        if test_frame is None:
            print("Failed to capture initial frame. Ensure screen is active.")
            return
            
        h, w, c = test_frame.shape
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Lossless Frame Gen Proto")
        clock = pygame.time.Clock()
        
        self.running = True
        self.start_time = time.time()
        
        # Start threads
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
                    
                    # Convert RGB to BGR for pygame (or handle it)
                    # Pygame expects RGB usually, so we might be fine if capture is RGB
                    # Actually Pygame surface uses (Width, Height) and capture is (Height, Width)
                    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    screen.blit(surface, (0, 0))
                    pygame.display.flip()
                    
                    self.frame_count += 1
                    if self.frame_count % 60 == 0:
                        elapsed = time.time() - self.start_time
                        self.current_fps = self.frame_count / elapsed
                        pygame.display.set_caption(f"Lossless Frame Gen Proto - FPS: {self.current_fps:.2f}")
                
                clock.tick(self.target_fps)
        finally:
            self.running = False
            self.capture.stop_capture()
            pygame.quit()

if __name__ == "__main__":
    app = FrameGenerationApp(target_fps=60)
    app.run()
