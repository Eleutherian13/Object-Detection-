# Object Detection Project

## 📋 Overview

This project demonstrates real-time object detection using state-of-the-art YOLO (You Only Look Once) models. The project includes multiple implementations using:
- **YOLOv8** - Latest version with improved accuracy and speed
- **YOLOv11** - Most recent release with enhanced performance
- **PyTorch** - Deep learning framework for custom implementations

## 🎯 Objectives

1. Understand YOLO architecture and how it works for real-time object detection
2. Perform inference using pre-trained models for quick predictions
3. Analyze and extract detection results (bounding boxes, confidence scores, class names)
4. Learn both command-line and Python API approaches
5. Explore PyTorch implementations for deeper understanding

## 📁 Project Structure

```
object Detection/
├── README.md                          # This file - Project overview
├── SETUP.md                           # Setup and installation instructions
├── DOCUMENTATION.md                   # Detailed technical documentation
├── requirements.txt                   # Python dependencies
├── YOLOintroduction.ipynb            # Main notebook - YOLO basics and inference
├── PytorchIntroduction.ipynb         # PyTorch fundamentals
├── Pytorch_Demo.ipynb                # PyTorch YOLO implementation demo
├── yolov8n.pt                        # Pre-trained YOLOv8 nano model (~6.3 MB)
├── yolo11s.pt                        # Pre-trained YOLOv11 small model (~12 MB)
└── runs/                             # Results directory
    └── detect/
        └── predict/                  # Model inference outputs
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda package manager
- GPU (optional, recommended for faster inference)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd "object Detection"

# Install dependencies
pip install -r requirements.txt
```

### Run Inference

**Method 1: Using Command-line**
```bash
# Using YOLOv11 small model on a local image
yolo predict model=yolo11s.pt source="path/to/image.jpg"

# Using YOLOv8 nano model on a URL
yolo predict model=yolov8n.pt source="https://example.com/image.jpg"
```

**Method 2: Using Jupyter Notebooks**
```bash
# Launch Jupyter and open YOLOintroduction.ipynb
jupyter notebook YOLOintroduction.ipynb
```

## 📚 Files Description

### Notebooks

#### `YOLOintroduction.ipynb` ⭐ START HERE
- **Purpose**: Comprehensive introduction to YOLO object detection
- **Contents**:
  - Installation and environment setup
  - Command-line inference with YOLOv11
  - Python API usage with YOLOv8
  - Bounding box extraction and analysis
  - Multiple output formats (XYWH, XYXY, normalized variants)
- **Duration**: ~15-20 minutes
- **Difficulty**: Beginner to Intermediate

#### `PytorchIntroduction.ipynb`
- **Purpose**: Learn PyTorch fundamentals
- **Contents**:
  - PyTorch tensors and basic operations
  - Building neural networks
  - Training and evaluation loops
  - GPU utilization
- **Duration**: ~20-30 minutes
- **Difficulty**: Beginner

#### `Pytorch_Demo.ipynb`
- **Purpose**: Practical PyTorch implementation demo
- **Contents**:
  - Custom model architectures
  - Data loading and preprocessing
  - Model training and inference
  - Performance evaluation
- **Duration**: ~30-40 minutes
- **Difficulty**: Intermediate

### Pre-trained Models

#### `yolov8n.pt` (6.3 MB)
- **Model**: YOLOv8 Nano
- **Speed**: Fastest inference (1-2 ms on GPU)
- **Accuracy**: Good (80.4 mAP)
- **Use Case**: Real-time applications, edge devices, resource-constrained environments
- **Classes**: 80 (COCO dataset)

#### `yolo11s.pt` (12 MB)
- **Model**: YOLOv11 Small
- **Speed**: Fast inference (~4-5 ms on GPU)
- **Accuracy**: Better (39.5 mAP)
- **Use Case**: Balanced speed/accuracy, production environments
- **Classes**: 80 (COCO dataset)

## 🔑 Key Concepts

### YOLO Architecture
- **Single-stage detector**: Predictions made in one pass
- **Real-time**: Processes images at 30+ FPS on GPU
- **Multi-scale detection**: Detects objects of various sizes
- **End-to-end learning**: Entire detection pipeline trained together

### Bounding Box Formats

| Format | Definition | Use Case |
|--------|-----------|----------|
| **XYWH** | Center X, Center Y, Width, Height (pixels) | Original image coordinates |
| **XYWHN** | XYWH normalized to [0, 1] | Scale-independent comparison |
| **XYXY** | Top-left (X1,Y1), Bottom-right (X2,Y2) | Cropping, visualization |
| **XYXYN** | XYXY normalized to [0, 1] | Standardized format |

### Confidence Scores
- Range: 0.0 to 1.0
- Interpretation: Higher = more confident prediction
- Filtering: Typically threshold at 0.5 or higher
- Usage: Filter weak detections, improve precision

### Class Predictions
- COCO Dataset: 80 classes (people, cars, dogs, cats, etc.)
- Each detection includes predicted class name and confidence

## 📊 Output Structure

### Detection Results
```python
# Returns Results object with:
results[0].boxes      # Detected bounding boxes
  .xyxy              # Coordinates in XYXY format
  .conf              # Confidence scores
  .cls               # Class indices
results[0].names      # Class name mapping (dict)
results[0].img       # Annotated image
```

## 🎓 Learning Path

1. **Start with YOLOintroduction.ipynb**
   - Understand YOLO basics
   - Learn output formats
   - Run your first inference

2. **Move to PytorchIntroduction.ipynb**
   - Learn PyTorch fundamentals
   - Understand neural networks
   - Build simple models

3. **Explore Pytorch_Demo.ipynb**
   - See practical implementations
   - Train custom models
   - Evaluate performance

## 🔧 Configuration

### Model Selection
```python
from ultralytics import YOLO

# Nano (fastest)
model = YOLO("yolov8n.pt")

# Small (balanced)
model = YOLO("yolo11s.pt")

# Others: Medium (m), Large (l), Extra-Large (x)
```

### Inference Parameters
```python
# Adjust confidence threshold
results = model(image, conf=0.5)

# Adjust IOU threshold
results = model(image, iou=0.45)

# Save results
results = model(image, save=True)
```

## 📈 Performance Benchmarks

### YOLOv8n (Nano)
- **Inference Speed**: ~2ms (GPU), ~50ms (CPU)
- **Model Size**: 6.3 MB
- **Memory**: ~200 MB
- **mAP50**: 80.4

### YOLOv11s (Small)
- **Inference Speed**: ~4ms (GPU), ~100ms (CPU)
- **Model Size**: 12 MB
- **Memory**: ~400 MB
- **mAP50**: 39.5

## 🤝 Common Tasks

### Load and Run Inference
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("image.jpg")

# Access results
for result in results:
    print(result.boxes.xyxy)      # Bounding boxes
    print(result.boxes.conf)       # Confidence scores
    print(result.names)            # Class names
```

### Extract Specific Information
```python
for result in results:
    boxes = result.boxes
    
    # Get class names
    classes = [result.names[int(c)] for c in boxes.cls]
    
    # Get high-confidence detections
    high_conf = boxes.conf > 0.5
    high_conf_boxes = boxes.xyxy[high_conf]
```

### Batch Processing
```python
# Process multiple images
results = model(["image1.jpg", "image2.jpg", "image3.jpg"])

# Process video
results = model("video.mp4")

# Process from URL
results = model("https://example.com/image.jpg")
```

## 🐛 Troubleshooting

### Issue: Model download fails
- **Solution**: Manually place `.pt` files in project directory
- **Alternative**: Check internet connection, try using VPN

### Issue: Out of memory (OOM)
- **Solution**: Use smaller model (nano instead of large)
- **Alternative**: Reduce image size or batch size

### Issue: Slow inference on CPU
- **Solution**: Use GPU if available
- **Alternative**: Use nano model for faster CPU inference

## 📖 Additional Resources

### Official Documentation
- [Ultralytics YOLO Docs](https://docs.ultralytics.com)
- [PyTorch Official Guide](https://pytorch.org/tutorials/)
- [COCO Dataset Info](https://cocodataset.org)

### Research Papers
- YOLOv8: [Paper](https://arxiv.org/abs/2305.09972)
- YOLOv11: Latest improvements and benchmarks
- Original YOLO: [You Only Look Once](https://arxiv.org/abs/1506.02640)

### Community Resources
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [Computer Vision Stack Exchange](https://stackoverflow.com/questions/tagged/object-detection)

## 📝 Notes

- Pre-trained models are trained on COCO dataset (80 object classes)
- Models automatically download on first use (~100-500 MB)
- GPU recommended for production use (30+ FPS)
- CPU inference slower but suitable for low-frequency processing

## 🤖 Model Comparison

| Aspect | YOLOv8n | YOLOv11s | YOLO Large |
|--------|---------|----------|-----------|
| Speed | ⚡⚡⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡⚡ |
| Accuracy | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Model Size | 6 MB | 12 MB | 140+ MB |
| Memory | Low | Medium | High |
| Best For | Edge, Real-time | Production | Maximum Accuracy |

## 📄 License

Specify your license here (e.g., MIT, Apache 2.0)

## 👨‍💻 Author

Created for learning and demonstration purposes.

## 🔄 Version History

- **v1.0** (Jan 2026): Initial project setup
  - YOLOv8 nano model
  - YOLOv11 small model
  - Jupyter notebooks for learning
  - Command-line and Python API examples
