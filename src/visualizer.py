import cv2
import numpy as np
from PIL import Image, ImageDraw

class Visualizer:
    def __init__(self, font):
        self.font = font
        # Colors
        self.bbox_color = (0, 255, 0)
        self.text_color = (0, 0, 0)
        self.text_bg_color = (0, 255, 0)
        self.mask_alpha = 0.5
        self.skeleton_color = (255, 0, 0)

    def draw_detections(self, image, results):
        """
        Draws bounding boxes and labels on the image.
        Args:
            image (PIL.Image): The image to draw on.
            results: YOLO results object.
        """
        draw = ImageDraw.Draw(image)
        
        for result in results:
            boxes = result.boxes
            if boxes:
                for box in boxes:
                    # Bounding Box
                    coords = box.xyxy.int().tolist()[0]
                    x1, y1, x2, y2 = coords
                    confidence = box.conf.item()
                    cls_id = int(box.cls.item())
                    label = f"{result.names[cls_id]}: {confidence:.2f}"

                    draw.rectangle([(x1, y1), (x2, y2)], outline=self.bbox_color, width=3)
                    
                    # Label Background and Text
                    text_bbox = draw.textbbox((0, 0), label, font=self.font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    
                    bg_x1 = x1
                    bg_y1 = y1 - text_h - 4
                    bg_x2 = x1 + text_w + 4
                    bg_y2 = y1

                    if bg_y1 < 0: # If label goes off top, move it inside box
                        bg_y1 = y1
                        bg_y2 = y1 + text_h + 4

                    draw.rectangle([(bg_x1, bg_y1), (bg_x2, bg_y2)], fill=self.text_bg_color)
                    draw.text((bg_x1 + 2, bg_y1), label, font=self.font, fill=self.text_color)
        return image

    def draw_segmentation(self, image, results):
        """
        Draws segmentation masks overlay.
        """
        # Convert PIL to Numpy for mask operations
        img_np = np.array(image)
        
        for result in results:
            if result.masks:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.cpu().numpy()
                
                # Resize masks to original image size
                img_h, img_w = img_np.shape[:2]
                
                for i, mask in enumerate(masks):
                    # Resize mask to image size using OpenCV
                    mask_resized = cv2.resize(mask, (img_w, img_h))
                    
                    # Create color overlay
                    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                    overlay = np.zeros_like(img_np, dtype=np.uint8)
                    overlay[mask_resized > 0.5] = color
                    
                    # Blend
                    img_np = cv2.addWeighted(img_np, 1.0, overlay, self.mask_alpha, 0)
        
        # Convert back to PIL for consistency with draw_detections (calls it internally often)
        # But actually, Ultralytics results.plot() is very good for this.
        # However, to maintain the Pillow text style we might want to mix.
        # For simple robust segment visualization, manual drawing is complex.
        # Let's use a simpler approach: Revert to PIL for consistency
        return Image.fromarray(img_np)

    def draw_pose(self, image, results):
        """
        Draws pose skeletons.
        """
        draw = ImageDraw.Draw(image)
        
        for result in results:
            keypoints = result.keypoints
            if keypoints is not None:
                kpts = keypoints.xy.cpu().numpy() # (N, 17, 2)
                
                # Edges for COCO keypoints
                skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], 
                            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
                            [2, 4], [3, 5], [4, 6], [5, 7]]

                for person_kpts in kpts:
                    # Draw lines
                    for sk in skeleton:
                        p1_idx, p2_idx = sk[0]-1, sk[1]-1
                        if p1_idx < len(person_kpts) and p2_idx < len(person_kpts):
                            x1, y1 = person_kpts[p1_idx]
                            x2, y2 = person_kpts[p2_idx]
                            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                                draw.line([(x1, y1), (x2, y2)], fill=self.skeleton_color, width=2)
                    
                    # Draw points
                    for x, y in person_kpts:
                        if x > 0 and y > 0:
                            draw.ellipse([(x-3, y-3), (x+3, y+3)], fill=(0,0,255))
        return image
