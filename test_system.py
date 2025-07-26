#!/usr/bin/env python3
"""
Test script for Object Detection and Tracking System
This script tests all components to ensure they work correctly
"""

import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics imported successfully")
    except ImportError as e:
        print(f"✗ Ultralytics import failed: {e}")
        return False
    
    try:
        from utils import Colors, draw_box, draw_fps, FPS, COCO_CLASSES
        print("✓ Utils imported successfully")
    except ImportError as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    try:
        from tracker import ObjectTracker, SimpleTracker
        print("✓ Tracker imported successfully")
    except ImportError as e:
        print(f"✗ Tracker import failed: {e}")
        return False
    
    try:
        from object_detection_tracking import ObjectDetectionTracker
        print("✓ Main detector imported successfully")
    except ImportError as e:
        print(f"✗ Main detector import failed: {e}")
        return False
    
    return True

def test_yolo_model():
    """Test YOLO model loading and inference."""
    print("\nTesting YOLO model...")
    
    try:
        from ultralytics import YOLO
        
        # Check if model exists
        model_path = "yolov8n.pt"
        if not Path(model_path).exists():
            print(f"✗ Model {model_path} not found")
            return False
        
        # Load model
        model = YOLO(model_path)
        print("✓ YOLO model loaded successfully")
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(test_image, conf=0.5, verbose=False)
        print("✓ YOLO inference completed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLO test failed: {e}")
        return False

def test_tracker():
    """Test object tracker."""
    print("\nTesting object tracker...")
    
    try:
        from tracker import SimpleTracker
        
        # Create tracker
        tracker = SimpleTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        print("✓ Tracker created successfully")
        
        # Create test detections
        detections = [
            {'bbox': [100, 100, 200, 200], 'class_id': 0, 'confidence': 0.8},
            {'bbox': [300, 300, 400, 400], 'class_id': 1, 'confidence': 0.9},
        ]
        
        # Update tracker
        tracks = tracker.update(detections)
        print(f"✓ Tracker updated successfully, {len(tracks)} tracks created")
        
        # Test multiple updates
        for i in range(5):
            # Move detections slightly
            new_detections = [
                {'bbox': [100 + i*10, 100 + i*10, 200 + i*10, 200 + i*10], 'class_id': 0, 'confidence': 0.8},
                {'bbox': [300 + i*5, 300 + i*5, 400 + i*5, 400 + i*5], 'class_id': 1, 'confidence': 0.9},
            ]
            tracks = tracker.update(new_detections)
        
        print(f"✓ Tracker maintained {len(tracks)} tracks over multiple frames")
        
        return True
        
    except Exception as e:
        print(f"✗ Tracker test failed: {e}")
        return False

def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from utils import Colors, draw_box, draw_fps, FPS, calculate_iou
        
        # Test color generation
        colors = Colors()
        color1 = colors()
        color2 = colors()
        print("✓ Color generation works")
        
        # Test drawing functions
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image = draw_box(test_image, [100, 100, 200, 200], "Test", (0, 255, 0))
        test_image = draw_fps(test_image, 30.5)
        print("✓ Drawing functions work")
        
        # Test FPS counter
        fps_counter = FPS()
        for _ in range(30):
            fps_counter.update()
        fps = fps_counter.get_fps()
        print(f"✓ FPS counter works: {fps:.1f} FPS")
        
        # Test IoU calculation
        iou = calculate_iou([0, 0, 100, 100], [50, 50, 150, 150])
        print(f"✓ IoU calculation works: {iou:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False

def test_detector():
    """Test the main detector class."""
    print("\nTesting main detector...")
    
    try:
        from object_detection_tracking import ObjectDetectionTracker
        
        # Create detector
        detector = ObjectDetectionTracker(
            model_path="yolov8n.pt",
            conf_threshold=0.5,
            iou_threshold=0.45
        )
        print("✓ Detector created successfully")
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process frame
        processed_frame = detector.process_frame(test_frame)
        print("✓ Frame processing works")
        
        return True
        
    except Exception as e:
        print(f"✗ Detector test failed: {e}")
        return False

def test_video_capture():
    """Test video capture functionality."""
    print("\nTesting video capture...")
    
    try:
        # Test webcam access
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Webcam access works")
                cap.release()
                return True
            else:
                print("✗ Could not read from webcam")
                cap.release()
                return False
        else:
            print("✗ Could not open webcam")
            return False
            
    except Exception as e:
        print(f"✗ Video capture test failed: {e}")
        return False

def create_test_image():
    """Create a test image with simple shapes."""
    print("\nCreating test image...")
    
    # Create blank image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw some shapes
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(img, (400, 200), 50, (255, 0, 0), -1)  # Blue circle
    cv2.rectangle(img, (300, 300), (400, 400), (0, 0, 255), -1)  # Red rectangle
    
    # Save image
    cv2.imwrite("test_image.jpg", img)
    print("✓ Test image created: test_image.jpg")
    return True

def run_comprehensive_test():
    """Run all tests."""
    print("Object Detection and Tracking System - Comprehensive Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("YOLO Model", test_yolo_model),
        ("Tracker", test_tracker),
        ("Utils", test_utils),
        ("Detector", test_detector),
        ("Video Capture", test_video_capture),
        ("Test Image Creation", create_test_image),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python quick_test.py' for a quick webcam test")
        print("2. Run 'python demo.py' for interactive demos")
        print("3. Run 'python object_detection_tracking.py --source 0' for webcam detection")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed correctly.")
    
    return passed == total

def main():
    """Main test function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test mode
        print("Running quick test...")
        if test_imports() and test_yolo_model():
            print("✓ Quick test passed!")
            return True
        else:
            print("✗ Quick test failed!")
            return False
    else:
        # Full test mode
        return run_comprehensive_test()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 