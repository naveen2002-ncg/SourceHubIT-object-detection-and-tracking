#!/usr/bin/env python3
"""
Object Detection and Tracking System
Real-time object detection and tracking using YOLOv8 and OpenCV
"""

import cv2
import numpy as np
import argparse
import time
import os
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
from tracker import ObjectTracker, SimpleTracker
from utils import Colors, draw_box, draw_tracks, draw_fps, draw_stats, FPS, COCO_CLASSES

class ObjectDetectionTracker:
    """Main class for object detection and tracking."""
    
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5, 
                 iou_threshold: float = 0.45, max_age: int = 30, min_hits: int = 3):
        """
        Initialize the object detection and tracking system.
        
        Args:
            model_path: Path to YOLO model or model name
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            max_age: Maximum age for tracks
            min_hits: Minimum hits to confirm a track
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        
        # Initialize YOLO model
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
        # Initialize tracker
        try:
            self.tracker = ObjectTracker(max_age=max_age, min_hits=min_hits)
            print("Using advanced tracker with Hungarian algorithm")
        except ImportError:
            print("scipy not available, using simple tracker")
            self.tracker = SimpleTracker(max_age=max_age, min_hits=min_hits)
        
        # Initialize utilities
        self.colors = Colors()
        self.fps_counter = FPS()
        
        # State variables
        self.tracking_enabled = True
        self.detection_enabled = True
        self.frame_count = 0
        
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the frame using YOLO.
        
        Args:
            frame: Input frame
        
        Returns:
            List of detection dictionaries
        """
        if not self.detection_enabled:
            return []
        
        # Run YOLO detection
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id,
                        'confidence': confidence
                    })
        
        return detections
    
    def track_objects(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Track objects across frames.
        
        Args:
            detections: List of detections from current frame
        
        Returns:
            List of tracked objects
        """
        if not self.tracking_enabled:
            return detections
        
        return self.tracker.update(detections)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for detection and tracking.
        
        Args:
            frame: Input frame
        
        Returns:
            Processed frame with detections and tracks
        """
        self.frame_count += 1
        self.fps_counter.update()
        
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Track objects
        tracks = self.track_objects(detections)
        
        # Draw results
        if self.tracking_enabled and tracks:
            frame = draw_tracks(frame, tracks, self.colors)
        elif detections:
            # Draw detections without tracking
            for detection in detections:
                bbox = detection['bbox']
                class_id = detection['class_id']
                confidence = detection['confidence']
                
                color = self.colors()
                color_bgr = (color[2], color[1], color[0])
                label = f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                
                frame = draw_box(frame, bbox, label, color_bgr)
        
        # Draw statistics
        frame = draw_fps(frame, self.fps_counter.get_fps())
        frame = draw_stats(frame, len(detections), len(tracks))
        
        return frame
    
    def process_video(self, source: str, output_path: Optional[str] = None):
        """
        Process video stream (webcam or video file).
        
        Args:
            source: Video source (0 for webcam, or path to video file)
            output_path: Optional output video path
        """
        # Open video capture
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
            print(f"Opening webcam {source}")
        else:
            cap = cv2.VideoCapture(source)
            print(f"Opening video file: {source}")
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output video will be saved to: {output_path}")
        
        print("\nControls:")
        print("Q - Quit")
        print("S - Save current frame")
        print("T - Toggle tracking")
        print("D - Toggle detection")
        print("R - Reset tracker")
        print("H - Show/hide help")
        
        help_visible = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Show help if requested
                if help_visible:
                    self._draw_help(processed_frame)
                
                # Display frame
                cv2.imshow('Object Detection and Tracking', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(processed_frame)
                elif key == ord('t'):
                    self.tracking_enabled = not self.tracking_enabled
                    print(f"Tracking {'enabled' if self.tracking_enabled else 'disabled'}")
                elif key == ord('d'):
                    self.detection_enabled = not self.detection_enabled
                    print(f"Detection {'enabled' if self.detection_enabled else 'disabled'}")
                elif key == ord('r'):
                    self.tracker.reset()
                    print("Tracker reset")
                elif key == ord('h'):
                    help_visible = not help_visible
                
                # Write frame to output video
                if writer:
                    writer.write(processed_frame)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print("Video processing completed")
    
    def process_image(self, image_path: str, output_path: Optional[str] = None):
        """
        Process a single image.
        
        Args:
            image_path: Path to input image
            output_path: Optional output image path
        """
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        print(f"Processing image: {image_path}")
        
        # Process frame
        processed_frame = self.process_frame(frame)
        
        # Save output
        if output_path:
            cv2.imwrite(output_path, processed_frame)
            print(f"Output saved to: {output_path}")
        
        # Display result
        cv2.imshow('Object Detection and Tracking', processed_frame)
        print("Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _save_frame(self, frame: np.ndarray):
        """Save current frame to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as: {filename}")
    
    def _draw_help(self, frame: np.ndarray):
        """Draw help information on frame."""
        help_text = [
            "Controls:",
            "Q - Quit",
            "S - Save frame",
            "T - Toggle tracking",
            "D - Toggle detection",
            "R - Reset tracker",
            "H - Hide help"
        ]
        
        y_offset = 150
        for text in help_text:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 25

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Object Detection and Tracking System")
    parser.add_argument("--source", type=str, default="0", 
                       help="Video source (0 for webcam, or path to video/image file)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                       help="YOLO model path or name (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold for detections")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold for NMS")
    parser.add_argument("--max-age", type=int, default=30,
                       help="Maximum age for tracks")
    parser.add_argument("--min-hits", type=int, default=3,
                       help="Minimum hits to confirm a track")
    parser.add_argument("--output", type=str, default=None,
                       help="Output video/image path")
    
    args = parser.parse_args()
    
    # Create detector/tracker
    detector = ObjectDetectionTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_age=args.max_age,
        min_hits=args.min_hits
    )
    
    # Process source
    if args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        detector.process_image(args.source, args.output)
    else:
        detector.process_video(args.source, args.output)

if __name__ == "__main__":
    main() 