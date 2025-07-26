#!/usr/bin/env python3
"""
Demo script for Object Detection and Tracking System
This script demonstrates various features and configurations
"""

import cv2
import numpy as np
import time
from object_detection_tracking import ObjectDetectionTracker

def create_test_video(output_path: str = "test_video.mp4", duration: int = 10, fps: int = 30):
    """
    Create a test video with moving objects for demonstration.
    
    Args:
        output_path: Path to save the test video
        duration: Duration of video in seconds
        fps: Frames per second
    """
    print(f"Creating test video: {output_path}")
    
    # Video properties
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create moving objects
    objects = [
        {'x': 100, 'y': 200, 'dx': 2, 'dy': 1, 'size': 50, 'color': (0, 255, 0)},  # Green circle
        {'x': 500, 'y': 150, 'dx': -1.5, 'dy': 2, 'size': 40, 'color': (255, 0, 0)},  # Red circle
        {'x': 300, 'y': 400, 'dx': 1, 'dy': -1.5, 'size': 60, 'color': (0, 0, 255)},  # Blue circle
    ]
    
    for frame_num in range(duration * fps):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update and draw objects
        for obj in objects:
            # Update position
            obj['x'] += obj['dx']
            obj['y'] += obj['dy']
            
            # Bounce off walls
            if obj['x'] <= obj['size'] or obj['x'] >= width - obj['size']:
                obj['dx'] *= -1
            if obj['y'] <= obj['size'] or obj['y'] >= height - obj['size']:
                obj['dy'] *= -1
            
            # Draw circle
            cv2.circle(frame, (int(obj['x']), int(obj['y'])), obj['size'], obj['color'], -1)
        
        # Add some text
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {output_path}")

def demo_webcam():
    """Demo with webcam."""
    print("=== Webcam Demo ===")
    print("This will open your webcam for real-time object detection and tracking.")
    print("Press 'Q' to quit, 'T' to toggle tracking, 'D' to toggle detection.")
    
    detector = ObjectDetectionTracker(
        model_path="yolov8n.pt",
        conf_threshold=0.5,
        iou_threshold=0.45
    )
    
    detector.process_video("0")

def demo_video_file(video_path: str):
    """Demo with video file."""
    print(f"=== Video File Demo: {video_path} ===")
    
    detector = ObjectDetectionTracker(
        model_path="yolov8n.pt",
        conf_threshold=0.5,
        iou_threshold=0.45
    )
    
    detector.process_video(video_path, "output_video.mp4")

def demo_different_models():
    """Demo with different YOLO models."""
    models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    
    print("=== Different Models Demo ===")
    print("This will test different YOLO models on the same input.")
    
    # Create test video if it doesn't exist
    test_video = "test_video.mp4"
    try:
        cap = cv2.VideoCapture(test_video)
        if not cap.isOpened():
            create_test_video(test_video)
        cap.release()
    except:
        create_test_video(test_video)
    
    for model in models:
        print(f"\nTesting model: {model}")
        try:
            detector = ObjectDetectionTracker(
                model_path=model,
                conf_threshold=0.5,
                iou_threshold=0.45
            )
            
            # Process first few frames only for demo
            cap = cv2.VideoCapture(test_video)
            frame_count = 0
            max_frames = 30  # Process only first 30 frames for demo
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = detector.process_frame(frame)
                cv2.imshow(f'Model: {model}', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error with model {model}: {e}")

def demo_tracking_comparison():
    """Demo comparing detection vs tracking."""
    print("=== Tracking Comparison Demo ===")
    print("This will show the difference between detection-only and tracking modes.")
    
    # Create test video if it doesn't exist
    test_video = "test_video.mp4"
    try:
        cap = cv2.VideoCapture(test_video)
        if not cap.isOpened():
            create_test_video(test_video)
        cap.release()
    except:
        create_test_video(test_video)
    
    detector = ObjectDetectionTracker(
        model_path="yolov8n.pt",
        conf_threshold=0.5,
        iou_threshold=0.45
    )
    
    print("\n1. Detection only (no tracking)")
    detector.tracking_enabled = False
    detector.detection_enabled = True
    
    cap = cv2.VideoCapture(test_video)
    frame_count = 0
    while frame_count < 60:  # Show first 2 seconds
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = detector.process_frame(frame)
        cv2.imshow('Detection Only', processed_frame)
        
        if cv2.waitKey(33) & 0xFF == ord('q'):  # ~30 FPS
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n2. Detection + Tracking")
    detector.tracking_enabled = True
    detector.detection_enabled = True
    detector.tracker.reset()
    
    cap = cv2.VideoCapture(test_video)
    frame_count = 0
    while frame_count < 60:  # Show first 2 seconds
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = detector.process_frame(frame)
        cv2.imshow('Detection + Tracking', processed_frame)
        
        if cv2.waitKey(33) & 0xFF == ord('q'):  # ~30 FPS
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

def demo_performance_test():
    """Demo for performance testing."""
    print("=== Performance Test Demo ===")
    print("This will test different confidence thresholds and measure performance.")
    
    # Create test video if it doesn't exist
    test_video = "test_video.mp4"
    try:
        cap = cv2.VideoCapture(test_video)
        if not cap.isOpened():
            create_test_video(test_video)
        cap.release()
    except:
        create_test_video(test_video)
    
    confidence_thresholds = [0.3, 0.5, 0.7, 0.9]
    
    for conf in confidence_thresholds:
        print(f"\nTesting confidence threshold: {conf}")
        
        detector = ObjectDetectionTracker(
            model_path="yolov8n.pt",
            conf_threshold=conf,
            iou_threshold=0.45
        )
        
        cap = cv2.VideoCapture(test_video)
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 30:  # Process 30 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = detector.process_frame(frame)
            frame_count += 1
        
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        
        print(f"Confidence {conf}: {fps:.1f} FPS")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main demo function."""
    print("Object Detection and Tracking System - Demo")
    print("=" * 50)
    
    while True:
        print("\nChoose a demo:")
        print("1. Webcam demo (real-time)")
        print("2. Video file demo")
        print("3. Different YOLO models comparison")
        print("4. Tracking vs Detection comparison")
        print("5. Performance test")
        print("6. Create test video")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            demo_webcam()
        elif choice == "2":
            video_path = input("Enter video file path (or press Enter for test video): ").strip()
            if not video_path:
                video_path = "test_video.mp4"
            demo_video_file(video_path)
        elif choice == "3":
            demo_different_models()
        elif choice == "4":
            demo_tracking_comparison()
        elif choice == "5":
            demo_performance_test()
        elif choice == "6":
            create_test_video()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 