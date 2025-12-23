from ultralytics import YOLO
import torch

class YOLODetector:
    def __init__(self, model_path, device='cpu'):
        print(f"Loading model: {model_path} on device: {device}")
        try:
            self.model = YOLO(model_path)
            
            # Smart device selection
            if device != 'cpu' and not torch.cuda.is_available():
                print(f"[WARNING] GPU requested ('{device}') but CUDA is not available. Falling back to CPU.")
                self.device = 'cpu'
            else:
                self.device = device
                
            if self.device != 'cpu':
                print(f"Using Device: GPU ({torch.cuda.get_device_name(0)})")
            else:
                print("Using Device: CPU")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path}: {e}")

    def predict(self, frame, conf=0.5):
        """
        Runs inference on the frame.
        """
        # verbose=False to reduce console spam
        # stream=True is usually for generators, here we want immediate results for one frame
        results = self.model(frame, conf=conf, device=self.device, verbose=False)
        return results
