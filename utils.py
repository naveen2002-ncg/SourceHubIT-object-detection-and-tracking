import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Any
import time

# COCO dataset class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class Colors:
    """Generate random colors for bounding boxes."""
    
    def __init__(self):
        self.palette = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(80)]
        self.n = 0
    
    def __call__(self, bgr=False):
        color = self.palette[self.n % len(self.palette)]
        self.n += 1
        return (color[2], color[1], color[0]) if bgr else color

def draw_box(img: np.ndarray, box: List[float], label: str = '', color: Tuple[int, int, int] = (128, 128, 128), 
             txt_color: Tuple[int, int, int] = (255, 255, 255), line_thickness: int = 2) -> np.ndarray:
    """
    Draw a bounding box on the image.
    
    Args:
        img: Input image
        box: Bounding box coordinates [x1, y1, x2, y2]
        label: Label text
        color: Box color (BGR)
        txt_color: Text color (BGR)
        line_thickness: Line thickness
    
    Returns:
        Image with bounding box drawn
    """
    x1, y1, x2, y2 = box
    c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    
    if label:
        tf = max(line_thickness - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, txt_color, 
                    thickness=tf, lineType=cv2.LINE_AA)
    
    return img

def draw_tracks(img: np.ndarray, tracks: List[Dict[str, Any]], colors: Colors) -> np.ndarray:
    """
    Draw tracking information on the image.
    
    Args:
        img: Input image
        tracks: List of track dictionaries
        colors: Color generator
    
    Returns:
        Image with tracking information drawn
    """
    for track in tracks:
        box = track['bbox']
        track_id = track['track_id']
        class_id = track['class_id']
        confidence = track['confidence']
        
        # Generate color based on track ID
        color = colors()
        color_bgr = (color[2], color[1], color[0])
        
        # Create label
        label = f"ID:{track_id} {COCO_CLASSES[class_id]} {confidence:.2f}"
        
        # Draw bounding box
        img = draw_box(img, box, label, color_bgr)
        
        # Draw track ID at center
        center_x = int((box[0] + box[2]) / 2)
        center_y = int((box[1] + box[3]) / 2)
        cv2.circle(img, (center_x, center_y), 4, color_bgr, -1)
    
    return img

def draw_fps(img: np.ndarray, fps: float) -> np.ndarray:
    """
    Draw FPS counter on the image.
    
    Args:
        img: Input image
        fps: FPS value
    
    Returns:
        Image with FPS counter
    """
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    return img

def draw_stats(img: np.ndarray, num_detections: int, num_tracks: int) -> np.ndarray:
    """
    Draw detection and tracking statistics on the image.
    
    Args:
        img: Input image
        num_detections: Number of detections
        num_tracks: Number of active tracks
    
    Returns:
        Image with statistics
    """
    cv2.putText(img, f"Detections: {num_detections}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"Tracks: {num_tracks}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def resize_image(img: np.ndarray, max_size: int = 640) -> Tuple[np.ndarray, float]:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        img: Input image
        max_size: Maximum dimension size
    
    Returns:
        Resized image and scale factor
    """
    h, w = img.shape[:2]
    scale = min(max_size / w, max_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    if scale != 1:
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return img, scale

class FPS:
    """FPS counter utility."""
    
    def __init__(self):
        self.fps = 0
        self.fps_count = 0
        self.fps_start_time = time.time()
    
    def update(self):
        """Update FPS calculation."""
        self.fps_count += 1
        if self.fps_count >= 30:
            current_time = time.time()
            self.fps = self.fps_count / (current_time - self.fps_start_time)
            self.fps_count = 0
            self.fps_start_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps 