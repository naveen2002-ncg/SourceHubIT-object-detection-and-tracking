#!/usr/bin/env python3
"""
Installation script for Object Detection and Tracking System
This script helps set up the environment and download required models
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        return False
    print(f"Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def download_yolo_models():
    """Download YOLO models if they don't exist."""
    models = [
        "yolov8n.pt",  # Nano model (fastest)
        "yolov8s.pt",  # Small model (balanced)
        "yolov8m.pt",  # Medium model (more accurate)
    ]
    
    base_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    
    for model in models:
        model_path = Path(model)
        if model_path.exists():
            print(f"✓ {model} already exists")
            continue
        
        print(f"Downloading {model}...")
        try:
            url = base_url + model
            urllib.request.urlretrieve(url, model)
            print(f"✓ {model} downloaded successfully")
        except Exception as e:
            print(f"✗ Error downloading {model}: {e}")
            print("You can download it manually from: https://github.com/ultralytics/ultralytics/releases")

def test_installation():
    """Test if the installation is working."""
    print("\nTesting installation...")
    
    try:
        # Test imports
        import cv2
        print("✓ OpenCV imported successfully")
        
        import torch
        print("✓ PyTorch imported successfully")
        
        from ultralytics import YOLO
        print("✓ Ultralytics imported successfully")
        
        # Test YOLO model loading
        model = YOLO("yolov8n.pt")
        print("✓ YOLO model loaded successfully")
        
        print("\n🎉 Installation completed successfully!")
        print("You can now run the object detection and tracking system.")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

def create_sample_script():
    """Create a sample script for quick testing."""
    sample_code = '''#!/usr/bin/env python3
"""
Quick test script for Object Detection and Tracking System
"""

from object_detection_tracking import ObjectDetectionTracker

def main():
    # Create detector
    detector = ObjectDetectionTracker(
        model_path="yolov8n.pt",
        conf_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Test with webcam
    print("Starting webcam test...")
    print("Press 'Q' to quit")
    detector.process_video("0")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_test.py", "w") as f:
        f.write(sample_code)
    
    print("✓ Created quick_test.py for testing")

def main():
    """Main installation function."""
    print("Object Detection and Tracking System - Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please check your internet connection and try again.")
        return
    
    # Download YOLO models
    download_yolo_models()
    
    # Test installation
    if test_installation():
        create_sample_script()
        
        print("\n" + "=" * 50)
        print("Installation Summary:")
        print("✓ Python environment set up")
        print("✓ Required packages installed")
        print("✓ YOLO models downloaded")
        print("✓ Quick test script created")
        
        print("\nNext steps:")
        print("1. Run 'python quick_test.py' to test with webcam")
        print("2. Run 'python demo.py' for interactive demos")
        print("3. Run 'python object_detection_tracking.py --help' for command line options")
        
        print("\nExample usage:")
        print("  python object_detection_tracking.py --source 0  # Webcam")
        print("  python object_detection_tracking.py --source video.mp4  # Video file")
        print("  python object_detection_tracking.py --source image.jpg  # Image file")
    else:
        print("\nInstallation failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 