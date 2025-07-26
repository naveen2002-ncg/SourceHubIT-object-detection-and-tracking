"""
Configuration file for Object Detection and Tracking System
Modify these settings to customize the system behavior
"""

# YOLO Model Configuration
YOLO_CONFIG = {
    # Model selection (choose one)
    'model_path': 'yolov8n.pt',  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    
    # Detection parameters
    'conf_threshold': 0.5,        # Confidence threshold (0.0 - 1.0)
    'iou_threshold': 0.45,        # IoU threshold for NMS (0.0 - 1.0)
    
    # Model size for inference
    'imgsz': 640,                 # Input image size
}

# Tracking Configuration
TRACKING_CONFIG = {
    # Tracker parameters
    'max_age': 30,                # Maximum frames to keep a track without updates
    'min_hits': 3,                # Minimum detections before confirming a track
    'iou_threshold': 0.3,         # IoU threshold for track association
    
    # Kalman filter parameters
    'process_noise': 0.03,        # Process noise for Kalman filter
    'measurement_noise': 1.0,     # Measurement noise for Kalman filter
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    # Display settings
    'show_fps': True,             # Show FPS counter
    'show_stats': True,           # Show detection/tracking statistics
    'show_track_ids': True,       # Show track IDs on bounding boxes
    'show_confidence': True,       # Show confidence scores
    
    # Drawing settings
    'line_thickness': 2,          # Bounding box line thickness
    'font_scale': 0.6,            # Text font scale
    'text_thickness': 2,          # Text thickness
    
    # Colors
    'fps_color': (0, 255, 0),    # FPS counter color (BGR)
    'stats_color': (255, 255, 255), # Statistics text color (BGR)
    'track_center_color': (255, 255, 255), # Track center point color (BGR)
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    # Processing settings
    'skip_frames': 0,             # Skip frames for faster processing (0 = process all frames)
    'max_fps': 30,                # Maximum FPS for display
    
    # Memory settings
    'max_tracks': 100,            # Maximum number of tracks to maintain
    'cleanup_interval': 10,       # Frames between cleanup operations
}

# Input/Output Configuration
IO_CONFIG = {
    # Video settings
    'default_source': '0',        # Default video source (0 for webcam)
    'output_format': 'mp4v',      # Output video codec
    'output_quality': 95,         # Output image quality (0-100)
    
    # File paths
    'output_dir': 'output',       # Output directory for saved files
    'log_file': 'detection_log.txt', # Log file path
}

# Advanced Configuration
ADVANCED_CONFIG = {
    # Detection filtering
    'class_filter': None,         # List of class IDs to detect (None = all classes)
    'min_area': 100,              # Minimum bounding box area
    'max_area': 10000,            # Maximum bounding box area
    
    # Tracking enhancements
    'use_kalman': True,           # Use Kalman filter for prediction
    'use_hungarian': True,        # Use Hungarian algorithm for association (requires scipy)
    
    # Debug settings
    'debug_mode': False,          # Enable debug output
    'save_debug_frames': False,   # Save debug frames
    'log_detections': False,      # Log detection results
}

# Class-specific configurations
CLASS_CONFIG = {
    # Person detection
    'person': {
        'min_confidence': 0.3,
        'tracking_priority': 1,
        'color': (0, 255, 0),  # Green
    },
    
    # Vehicle detection
    'car': {
        'min_confidence': 0.4,
        'tracking_priority': 2,
        'color': (255, 0, 0),  # Blue
    },
    
    # Animal detection
    'dog': {
        'min_confidence': 0.5,
        'tracking_priority': 3,
        'color': (0, 0, 255),  # Red
    },
}

# COCO class names for reference
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

def get_config():
    """Get the complete configuration dictionary."""
    return {
        'yolo': YOLO_CONFIG,
        'tracking': TRACKING_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'io': IO_CONFIG,
        'advanced': ADVANCED_CONFIG,
        'class_specific': CLASS_CONFIG,
        'coco_classes': COCO_CLASSES
    }

def validate_config(config):
    """Validate configuration parameters."""
    errors = []
    
    # Validate YOLO config
    if not (0 <= config['yolo']['conf_threshold'] <= 1):
        errors.append("conf_threshold must be between 0 and 1")
    
    if not (0 <= config['yolo']['iou_threshold'] <= 1):
        errors.append("iou_threshold must be between 0 and 1")
    
    # Validate tracking config
    if config['tracking']['max_age'] < 1:
        errors.append("max_age must be at least 1")
    
    if config['tracking']['min_hits'] < 1:
        errors.append("min_hits must be at least 1")
    
    # Validate performance config
    if config['performance']['skip_frames'] < 0:
        errors.append("skip_frames must be non-negative")
    
    if config['performance']['max_fps'] < 1:
        errors.append("max_fps must be at least 1")
    
    return errors

def print_config_summary():
    """Print a summary of the current configuration."""
    config = get_config()
    
    print("Configuration Summary:")
    print("=" * 30)
    print(f"Model: {config['yolo']['model_path']}")
    print(f"Confidence Threshold: {config['yolo']['conf_threshold']}")
    print(f"IoU Threshold: {config['yolo']['iou_threshold']}")
    print(f"Max Track Age: {config['tracking']['max_age']}")
    print(f"Min Hits: {config['tracking']['min_hits']}")
    print(f"Show FPS: {config['visualization']['show_fps']}")
    print(f"Debug Mode: {config['advanced']['debug_mode']}")
    print("=" * 30)

if __name__ == "__main__":
    # Validate and print configuration
    config = get_config()
    errors = validate_config(config)
    
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")
        print_config_summary() 