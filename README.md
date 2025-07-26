# 🎯 Object Detection and Tracking System

A real-time object detection and tracking system built with **YOLOv8** and **OpenCV**. This system can detect objects in video streams, track them across frames, and provide real-time visualization with bounding boxes and unique tracking IDs.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **🔍 Real-time Object Detection**: Using state-of-the-art YOLOv8 models
- **🎯 Multi-Object Tracking**: Track objects across video frames with unique IDs
- **📊 Live Visualization**: Bounding boxes, labels, confidence scores, and tracking IDs
- **📹 Multiple Input Sources**: Webcam, video files, and images
- **⚡ Performance Metrics**: Real-time FPS counter and detection statistics
- **🎮 Interactive Controls**: Toggle detection/tracking, save frames, reset tracker
- **🔄 Advanced Tracking**: Kalman filter prediction and IoU-based association
- **🎨 Customizable**: Configurable parameters for detection and tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- Webcam (for real-time detection)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/naveen2002-ncg/SourceHubIT-object-detection-and-tracking.git
   cd SourceHubIT-object-detection-and-tracking
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO model** (automatically done on first run)
   ```bash
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

4. **Run the system**
   ```bash
   # Real-time webcam detection
   python object_detection_tracking.py --source 0
   ```

## 📖 Usage

### Basic Commands

```bash
# Webcam detection (real-time)
python object_detection_tracking.py --source 0

# Video file detection
python object_detection_tracking.py --source path/to/video.mp4

# Image detection
python object_detection_tracking.py --source path/to/image.jpg

# Save output video
python object_detection_tracking.py --source 0 --output output_video.mp4
```

### Advanced Options

```bash
# Use different YOLO model
python object_detection_tracking.py --source 0 --model yolov8s.pt

# Adjust confidence threshold
python object_detection_tracking.py --source 0 --conf 0.7

# Adjust IoU threshold
python object_detection_tracking.py --source 0 --iou 0.5

# Customize tracking parameters
python object_detection_tracking.py --source 0 --max-age 50 --min-hits 5
```

### Interactive Controls

When running the system, use these keyboard controls:

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `S` | Save current frame |
| `T` | Toggle tracking on/off |
| `D` | Toggle detection on/off |
| `R` | Reset tracker |
| `H` | Show/hide help |

## 🎮 Demo Mode

Run the interactive demo to explore different features:

```bash
python demo.py
```

**Demo Options:**
1. **Webcam Demo** - Real-time detection with your camera
2. **Video File Demo** - Process pre-recorded videos
3. **Model Comparison** - Compare different YOLO models
4. **Tracking vs Detection** - See the difference between detection-only and tracking modes
5. **Performance Test** - Benchmark different confidence thresholds
6. **Test Video Creation** - Generate test videos for experimentation

## 🏗️ Project Structure

```
SourceHubIT-object-detection-and-tracking/
├── 📄 object_detection_tracking.py  # Main application
├── 📄 tracker.py                    # Object tracking implementation
├── 📄 utils.py                      # Utility functions and visualization
├── 📄 config.py                     # Configuration settings
├── 📄 demo.py                       # Interactive demo script
├── 📄 test_system.py                # System testing and validation
├── 📄 install.py                    # Installation helper
├── 📄 requirements.txt              # Python dependencies
├── 📄 .gitignore                    # Git ignore rules
└── 📄 README.md                     # This file
```

## ⚙️ Configuration

Customize the system behavior by modifying `config.py`:

```python
# Detection settings
YOLO_CONFIG = {
    'model_path': 'yolov8n.pt',     # Model selection
    'conf_threshold': 0.5,          # Confidence threshold
    'iou_threshold': 0.45,          # IoU threshold for NMS
}

# Tracking settings
TRACKING_CONFIG = {
    'max_age': 30,                  # Max frames without updates
    'min_hits': 3,                  # Min detections to confirm track
    'iou_threshold': 0.3,           # IoU threshold for association
}
```

## 🔧 Available Models

The system supports different YOLOv8 models:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yolov8n.pt` | 6.3MB | ⚡ Fast | Good | Real-time applications |
| `yolov8s.pt` | 22.6MB | 🏃 Medium | Better | Balanced performance |
| `yolov8m.pt` | 52.2MB | 🐌 Slow | Best | High accuracy needed |
| `yolov8l.pt` | 87.7MB | 🐌 Slow | Excellent | Maximum accuracy |
| `yolov8x.pt` | 136.7MB | 🐌 Slow | Outstanding | Research/benchmarking |

## 📊 Performance

**Typical Performance (on CPU):**
- **YOLOv8n**: ~15-25 FPS
- **YOLOv8s**: ~8-15 FPS  
- **YOLOv8m**: ~3-8 FPS

**Performance (on GPU):**
- **YOLOv8n**: ~60+ FPS
- **YOLOv8s**: ~40+ FPS
- **YOLOv8m**: ~25+ FPS

*Performance varies based on hardware, input resolution, and number of detected objects.*

## 🧪 Testing

Run comprehensive tests to verify system functionality:

```bash
# Quick test
python test_system.py --quick

# Full system test
python test_system.py
```

## 🐛 Troubleshooting

### Common Issues

**1. Webcam not working**
```bash
# Try different camera index
python object_detection_tracking.py --source 1
```

**2. Low FPS**
```bash
# Use smaller model
python object_detection_tracking.py --source 0 --model yolov8n.pt

# Increase confidence threshold
python object_detection_tracking.py --source 0 --conf 0.7
```

**3. Model download issues**
```bash
# Manual model download
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**4. Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 implementation
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [PyTorch](https://pytorch.org/) for deep learning framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/naveen2002-ncg/SourceHubIT-object-detection-and-tracking/issues) page
2. Create a new issue with detailed description
3. Include system information and error logs

---

**⭐ Star this repository if you find it helpful!**

**🔗 Repository:** [https://github.com/naveen2002-ncg/SourceHubIT-object-detection-and-tracking](https://github.com/naveen2002-ncg/SourceHubIT-object-detection-and-tracking)
