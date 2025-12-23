import cv2
import argparse
import sys
import numpy as np
from PIL import Image

from src.detector import YOLODetector
from src.visualizer import Visualizer
from src.utils import load_font

def parse_args():
    parser = argparse.ArgumentParser(description="Webcam Object Detector with YOLO11")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to YOLO model file")
    parser.add_argument("--task", type=str, choices=["detect", "segment", "pose"], default="detect", help="Task type")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0) or video file path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, 0, etc.)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize components
    try:
        detector = YOLODetector(args.model, args.device)
        font = load_font(size=18)
        visualizer = Visualizer(font)
    except Exception as e:
        print(f"Initialization Error: {e}")
        sys.exit(1)

    # Initialize Source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        sys.exit(1)

    print(f"Starting {args.task} on source {source}...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or error reading frame.")
            break

        # Inference
        results = detector.predict(frame, conf=args.conf)

        # Visualization
        # Convert to PIL for better drawing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        if args.task == "detect":
            pil_img = visualizer.draw_detections(pil_img, results)
        elif args.task == "segment":
            # Segment also needs detection boxes usually, or just overlay
            pil_img = visualizer.draw_segmentation(pil_img, results)
            pil_img = visualizer.draw_detections(pil_img, results) # Draw boxes on top
        elif args.task == "pose":
            pil_img = visualizer.draw_pose(pil_img, results)
            # visualizer.draw_detections(pil_img, results) # Optional: draw boxes too

        # Convert back to OpenCV
        frame_final = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        cv2.imshow(f"YOLO11 {args.task.capitalize()}", frame_final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()