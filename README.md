# Object Detection and Tracking System

A real-time object detection and tracking system using YOLOv8 and OpenCV. This system can detect objects in video streams and track them across frames.

## Features

- **Real-time Object Detection**: Using YOLOv8 pre-trained models
- **Object Tracking**: Track objects across video frames
- **Bounding Box Visualization**: Display detected objects with labels and confidence scores
- **Multiple Input Sources**: Support for webcam and video files
- **Performance Metrics**: FPS counter and detection statistics

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Real-time Detection with Webcam
```bash
python object_detection_tracking.py --source 0
```

### Detection from Video File
```bash
python object_detection_tracking.py --source path/to/video.mp4
```

### Detection from Image
```bash
python object_detection_tracking.py --source path/to/image.jpg
```

## Controls

- **Q**: Quit the application
- **S**: Save current frame
- **T**: Toggle tracking mode
- **D**: Toggle detection mode

## Project Structure

```
├── object_detection_tracking.py    # Main application
├── tracker.py                      # Object tracking implementation
├── utils.py                        # Utility functions
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Technologies Used

- **Python**: Core programming language
- **OpenCV**: Computer vision and video processing
- **YOLOv8**: Real-time object detection
- **Ultralytics**: YOLO model management
- **PyTorch**: Deep learning framework 