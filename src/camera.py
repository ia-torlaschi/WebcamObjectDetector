import cv2
import threading
import time
import numpy as np
import os
from PIL import Image
from src.detector import YOLODetector
from src.visualizer import Visualizer
from src.utils import load_font

class Camera:
    def __init__(self, source=0, model_base="yolo11n", device='cpu'):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError("Could not start camera.")
        
        # Default Settings
        self.model_base = model_base # e.g., 'yolo11n'
        self.device = device
        self.task = "detect" # detect, segment, pose
        self.conf = 0.5
        
        # Load Components
        font = load_font()
        self.visualizer = Visualizer(font)
        
        # Initial Model Load
        self.lock = threading.Lock()
        self._load_model()
        
    def _resolve_model_path(self):
        """Constructs filename based on version and task."""
        if self.task == "detect":
            return f"{self.model_base}.pt"
        elif self.task == "segment":
            return f"{self.model_base}-seg.pt"
        elif self.task == "pose":
            return f"{self.model_base}-pose.pt"
        return f"{self.model_base}.pt"

    def _load_model(self):
        model_path = self._resolve_model_path()
        print(f"Loading model: {model_path} for task: {self.task}")
        
        # Check if file exists, else fallback or warn
        if not os.path.exists(model_path):
            print(f"WARNING: Model {model_path} not found.")
            # We could implement logic to fallback to detection if seg/pose missing
            # But for now let's try to load it and let YOLO handle (it might auto-download)
            
        self.detector = YOLODetector(model_path, self.device)
        self.model_path = model_path # Store actual path for info

    def set_settings(self, task=None, conf=None, model_base=None):
        with self.lock:
            needs_reload = False
            
            if task and task != self.task:
                self.task = task
                needs_reload = True
            
            if model_base and model_base != self.model_base:
                self.model_base = model_base
                needs_reload = True
                
            if conf: 
                self.conf = float(conf)
            
            if needs_reload:
                try:
                    self._load_model()
                except Exception as e:
                    print(f"Error switching model: {e}")
                    # Revert or handle error? For now just log.

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Inference
        with self.lock:
            current_task = self.task
            current_conf = self.conf
            # Inference inside lock
            try:
                results = self.detector.predict(frame, conf=current_conf)
            except Exception as e:
                print(f"Inference error: {e}")
                results = []

        # 2. Visualization
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            if results:
                if current_task == "detect":
                    pil_img = self.visualizer.draw_detections(pil_img, results)
                elif current_task == "segment":
                    pil_img = self.visualizer.draw_segmentation(pil_img, results)
                    pil_img = self.visualizer.draw_detections(pil_img, results)
                elif current_task == "pose":
                    pil_img = self.visualizer.draw_pose(pil_img, results)

            frame_final = np.array(pil_img)
            frame_final = cv2.cvtColor(frame_final, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Visualization error: {e}")
            frame_final = frame
        
        ret, jpeg = cv2.imencode('.jpg', frame_final)
        return jpeg.tobytes()

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()
    
    def __del__(self):
        self.stop()
