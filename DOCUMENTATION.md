# Technical Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [YOLO Models](#yolo-models)
3. [API Reference](#api-reference)
4. [Data Formats](#data-formats)
5. [Advanced Usage](#advanced-usage)
6. [Performance Optimization](#performance-optimization)
7. [Model Training](#model-training)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### YOLO (You Only Look Once)

YOLO is a revolutionary approach to object detection that treats it as a single regression problem, predicting bounding boxes and class probabilities directly from full images in one evaluation.

#### Key Characteristics
- **Single-stage detector**: Processes entire image once, unlike two-stage detectors
- **Real-time**: Achieves 30+ FPS on modern GPUs
- **Global reasoning**: Sees entire image context
- **Unified detection pipeline**: Bounding boxes, confidence, and class predictions simultaneously

#### Evolution
```
YOLO v1 (2016)
  ↓
YOLO v2 (2017) - Batch normalization, multi-scale predictions
  ↓
YOLO v3 (2018) - Feature pyramid, improved backbone
  ↓
YOLOv4 (2020) - CSPDarknet, advanced augmentation
  ↓
YOLOv5 (2021) - PyTorch implementation, improved accuracy
  ↓
YOLOv8 (2023) - Ultralytics, new backbone, detection head
  ↓
YOLOv11 (2024) - Latest improvements, better performance
```

### Network Architecture (YOLOv8 Example)

```
Input Image (RGB, 640x640)
    ↓
[Backbone] - Feature Extraction
    - CSPDarknet with FPN
    - 3 detection scales
    ↓
[Neck] - Feature Fusion
    - PANet pyramid
    - Multi-scale features
    ↓
[Head] - Detection
    - Bounding box regression
    - Class probability
    - Objectness score
    ↓
Output: Detections
    - Bounding boxes
    - Confidence scores
    - Class predictions
```

---

## YOLO Models

### Model Variants

#### Size Comparison

| Model | Size | Parameters | Speed (GPU) | mAP50 | Memory |
|-------|------|-----------|------------|-------|--------|
| **Nano (n)** | 6 MB | 3.2M | 2.0ms | 80.4 | 200MB |
| **Small (s)** | 21 MB | 11.2M | 4.1ms | 86.6 | 400MB |
| **Medium (m)** | 49 MB | 25.9M | 10.3ms | 88.6 | 600MB |
| **Large (l)** | 99 MB | 43.7M | 15.9ms | 89.0 | 800MB |
| **X-Large (x)** | 155 MB | 68.2M | 21.7ms | 90.2 | 1.2GB |

### Model Selection Guide

#### YOLOv8n (Nano) - Use when:
- ✅ Maximum speed required
- ✅ Running on edge devices/mobile
- ✅ Real-time video processing (30+ FPS)
- ✅ Limited GPU/CPU resources
- ❌ High accuracy critical

#### YOLOv8s (Small) - Use when:
- ✅ Balanced speed and accuracy
- ✅ Most production scenarios
- ✅ Good accuracy needed (85+ mAP)
- ✅ Reasonable inference time
- ✅ Standard GPU available

#### YOLOv8m (Medium) - Use when:
- ✅ Better accuracy needed
- ✅ More computation resources available
- ✅ Less real-time constraint
- ✅ Complex scenes with small objects

#### YOLOv8l/x (Large/XLarge) - Use when:
- ✅ Maximum accuracy needed
- ✅ Powerful GPU available (RTX 30+)
- ✅ Batch processing acceptable
- ✅ Research or production with no speed constraint

---

## API Reference

### Basic Usage

#### Load a Model

```python
from ultralytics import YOLO

# Load model (auto-downloads on first use)
model = YOLO("yolov8n.pt")

# Or specify path
model = YOLO("./models/yolov8n.pt")
```

#### Run Inference

```python
# Single image
results = model("image.jpg")

# Multiple images
results = model(["img1.jpg", "img2.jpg", "img3.jpg"])

# Video file
results = model("video.mp4")

# Webcam
results = model(0)  # 0 = default camera

# Image array (numpy/PIL)
import cv2
frame = cv2.imread("image.jpg")
results = model(frame)
```

#### Access Results

```python
# Iterate through results
for result in results:
    # Get boxes
    boxes = result.boxes
    
    # Get image
    img = result.orig_img
    
    # Get detections
    detections = result.summary()
    
    # Plot
    annotated_img = result.plot()
```

### Results Object

#### Boxes Attributes

```python
boxes = result.boxes

# Coordinates (different formats)
boxes.xyxy        # torch.Tensor of shape (N, 4) - [x1, y1, x2, y2]
boxes.xyxyn       # Normalized XYXY (0-1)
boxes.xywh        # [x_center, y_center, width, height]
boxes.xywhn       # Normalized XYWH (0-1)

# Confidence and Class
boxes.conf        # torch.Tensor of shape (N,) - confidence scores
boxes.cls         # torch.Tensor of shape (N,) - class indices
boxes.id          # torch.Tensor of shape (N,) - tracking IDs (if tracking)

# Number of detections
len(boxes)        # int - number of detections

# Get data as numpy
boxes.xyxy.cpu().numpy()  # Convert to numpy array
```

#### Result Methods

```python
result = results[0]

# Get summary
summary = result.summary()  # Returns list of dicts with detection info

# Get boxes as dataframe
df = result.pandas().xyxy  # Useful for analysis

# Plot annotated image
plot = result.plot()

# Save annotated image
result.save("output.jpg")

# Get speed
print(result.speed)  # Dict with inference/NMS time
```

### Inference Parameters

```python
# Run with custom parameters
results = model(
    "image.jpg",
    
    # Detection parameters
    conf=0.25,              # Confidence threshold (0-1)
    iou=0.45,               # NMS IoU threshold (0-1)
    max_det=300,            # Maximum detections per image
    
    # Image parameters
    imgsz=640,              # Image size (square)
    
    # Processing
    save=False,             # Save annotated images
    save_txt=False,         # Save results to txt
    save_conf=False,        # Save confidence to txt
    
    # Performance
    half=False,             # Use FP16 precision (faster on GPU)
    device=0,               # GPU device (0 = first GPU)
    
    # Augmentation
    augment=False,          # Test time augmentation
    visualize=False,        # Visualize features
    
    # Format output
    verbose=False,          # Verbose output
)
```

### Batch Processing

```python
# Process multiple images efficiently
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = model.predict(
    source=images,
    batch_size=16,          # Process 16 at once
    device=0                # GPU device
)

# Access individual results
for i, result in enumerate(results):
    print(f"Image {i}: {len(result.boxes)} detections")
```

---

## Data Formats

### Input Formats

#### Supported Image Formats
```
JPG, JPEG, PNG, BMP, TIFF, GIF, WebP, etc.
```

#### Supported Video Formats
```
MP4, MOV, AVI, MKV, FLV, WMV, etc.
```

#### Supported Sources
```python
# File paths
model("image.jpg")
model("/path/to/image.jpg")

# Multiple files
model(["img1.jpg", "img2.jpg"])

# Directories (processes all images)
model("/path/to/image/folder")

# Video files
model("video.mp4")

# URLs
model("https://example.com/image.jpg")

# Numpy arrays
import numpy as np
img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
model(img_array)

# PIL Images
from PIL import Image
pil_img = Image.open("image.jpg")
model(pil_img)

# Video frames (OpenCV)
import cv2
cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()
model(frame)
```

### Output Formats

#### Bounding Box Formats

```python
# Get different coordinate formats
boxes = result.boxes

# XYXY: Top-left and bottom-right corners
# Format: [x1, y1, x2, y2]
# x1, y1 = top-left corner
# x2, y2 = bottom-right corner
xyxy = boxes.xyxy[0]  # [640, 360, 800, 480]

# XYWH: Center + dimensions
# Format: [x_center, y_center, width, height]
# Useful for: Drawing on images, custom processing
xywh = boxes.xywh[0]  # [720, 420, 160, 120]

# Normalized versions (0-1 scale)
xyxyn = boxes.xyxyn[0]  # Normalized XYXY
xywhn = boxes.xywhn[0]  # Normalized XYWH
```

#### Converting Between Formats

```python
import torch

def xywh2xyxy(x):
    """Convert XYWH to XYXY format"""
    y = x.clone() if isinstance(x, torch.Tensor) else x.copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def xyxy2xywh(x):
    """Convert XYXY to XYWH format"""
    y = x.clone() if isinstance(x, torch.Tensor) else x.copy()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x_center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y_center
    y[..., 2] = x[..., 2] - x[..., 0]        # width
    y[..., 3] = x[..., 3] - x[..., 1]        # height
    return y

# Usage
xywh_box = torch.tensor([100, 100, 50, 50])
xyxy_box = xywh2xyxy(xywh_box)
```

---

## Advanced Usage

### Custom Detection Filtering

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("image.jpg")

for result in results:
    boxes = result.boxes
    
    # Filter by confidence
    high_conf_mask = boxes.conf > 0.7
    high_conf_boxes = boxes.xyxy[high_conf_mask]
    high_conf_classes = boxes.cls[high_conf_mask]
    
    # Filter by class
    # 0=person, 1=bicycle, 2=car, 16=dog, 17=cat, etc.
    car_mask = boxes.cls == 2
    car_boxes = boxes.xyxy[car_mask]
    
    # Filter by size (area)
    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
    large_obj_mask = areas > 5000  # pixels
    large_boxes = boxes.xyxy[large_obj_mask]
```

### Post-Processing Results

```python
def process_detections(result, conf_threshold=0.5, class_filter=None):
    """
    Post-process YOLO detections
    
    Args:
        result: YOLO result object
        conf_threshold: Confidence threshold
        class_filter: List of class indices to keep (None = all)
    
    Returns:
        List of detection dicts
    """
    detections = []
    boxes = result.boxes
    
    for i in range(len(boxes)):
        conf = boxes.conf[i].item()
        cls = int(boxes.cls[i].item())
        
        # Apply filters
        if conf < conf_threshold:
            continue
        if class_filter and cls not in class_filter:
            continue
        
        # Extract coordinates
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
        
        detection = {
            'class_id': cls,
            'class_name': result.names[cls],
            'confidence': float(conf),
            'bbox': {
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2),
                'width': float(x2 - x1),
                'height': float(y2 - y1)
            }
        }
        detections.append(detection)
    
    return detections

# Usage
results = model("image.jpg")
detections = process_detections(results[0], conf_threshold=0.6, class_filter=[0, 2])
```

### Image Cropping

```python
import cv2

def crop_detections(image_path, result):
    """
    Crop detected objects from image
    
    Args:
        image_path: Path to image
        result: YOLO result object
    """
    image = cv2.imread(image_path)
    
    for i, box in enumerate(result.boxes.xyxy):
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        
        # Crop
        crop = image[y1:y2, x1:x2]
        
        # Save
        class_name = result.names[int(result.boxes.cls[i])]
        cv2.imwrite(f"crop_{i}_{class_name}.jpg", crop)

# Usage
results = model("image.jpg")
crop_detections("image.jpg", results[0])
```

### Tracking (Multi-Object Tracking)

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

# Track in video
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run tracking
    results = model.track(frame, persist=True)
    
    # Get tracking IDs
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        for track_id in track_ids:
            print(f"Track ID: {track_id}")
    
    # Visualize
    annotated_frame = results[0].plot()
    cv2.imshow("Tracking", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Performance Metrics

```python
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU)"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)
    
    inter_area = max(0, inter_max_x - inter_min_x) * max(0, inter_max_y - inter_min_y)
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Usage
box1 = [100, 100, 200, 200]
box2 = [150, 150, 250, 250]
iou = calculate_iou(box1, box2)
print(f"IoU: {iou:.2f}")
```

---

## Performance Optimization

### Speed Optimization

```python
# 1. Use smaller model
model = YOLO("yolov8n.pt")  # Nano, fastest

# 2. Reduce image size
results = model("image.jpg", imgsz=320)  # Default 640

# 3. Use FP16 (half precision)
results = model("image.jpg", half=True)  # Faster on GPU

# 4. Batch processing
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = model(images, batch_size=16)

# 5. Enable GPU
model = model.to("cuda:0")

# Complete optimized inference
model = YOLO("yolov8n.pt").to("cuda:0")
results = model("image.jpg", imgsz=320, half=True, conf=0.7)
```

### Accuracy Optimization

```python
# 1. Use larger model
model = YOLO("yolov8x.pt")  # Extra-large, most accurate

# 2. Higher resolution
results = model("image.jpg", imgsz=960)  # Higher than default 640

# 3. Lower confidence threshold (detects more)
results = model("image.jpg", conf=0.25)  # Default 0.25

# 4. Test-time augmentation
results = model("image.jpg", augment=True)

# 5. Ensemble models
from ultralytics import YOLO
model1 = YOLO("yolov8l.pt")
model2 = YOLO("yolov8x.pt")

results1 = model1("image.jpg")
results2 = model2("image.jpg")
# Combine results...
```

### Memory Optimization

```python
import torch

# 1. Use smaller batch size
results = model(images, batch_size=8)  # Instead of 16

# 2. Process one image at a time
for image in images:
    result = model(image)
    # Process immediately, don't store all results

# 3. Clear CUDA cache
torch.cuda.empty_cache()

# 4. Use nano model
model = YOLO("yolov8n.pt")

# 5. Reduce image size
results = model(image, imgsz=320)
```

---

## Model Training

### Fine-tuning on Custom Dataset

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8n.pt")

# Train on custom dataset
results = model.train(
    data="dataset.yaml",        # Dataset configuration
    epochs=100,                 # Number of training epochs
    imgsz=640,                  # Image size
    device=0,                   # GPU device
    patience=20,                # Early stopping patience
    batch=16,                   # Batch size
    workers=4,                  # Data loading workers
    lr0=0.01,                   # Initial learning rate
    momentum=0.937,             # SGD momentum
    weight_decay=0.0005,        # L2 regularization
    hsv_h=0.015,               # HSV-Hue augmentation
    hsv_s=0.7,                 # HSV-Saturation augmentation
    hsv_v=0.4,                 # HSV-Value augmentation
    flipud=0.0,                # Flip up-down
    fliplr=0.5,                # Flip left-right
    perspective=0.0,            # Perspective augmentation
    mosaic=1.0,                # Mosaic augmentation
    mixup=0.0,                 # Mixup augmentation
    copy_paste=0.0,            # Copy-paste augmentation
)

# Use trained model
results = model("image.jpg")

# Save model
model.save("my_model.pt")
```

### Dataset Format (YAML)

```yaml
# dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 80  # Number of classes
names: ['person', 'bicycle', 'car', ...]  # Class names
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Not Found Error

```
FileNotFoundError: [Errno 2] No such file or directory: 'yolov8n.pt'
```

**Solution:**
```python
# Models download automatically, but you can:
from ultralytics import YOLO

# This will download the model
model = YOLO("yolov8n.pt")

# Or download manually:
# https://github.com/ultralytics/assets/releases
```

#### 2. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Use smaller model
model = YOLO("yolov8n.pt")

# Reduce batch size
results = model(images, batch_size=4)

# Reduce image size
results = model(image, imgsz=320)

# Use FP16
results = model(image, half=True)

# Clear cache
import torch
torch.cuda.empty_cache()
```

#### 3. Slow Inference

**Solution:**
- Use smaller model (nano)
- Use GPU instead of CPU
- Reduce image size
- Use FP16 precision
- Batch process images

#### 4. Low Detection Accuracy

**Solution:**
- Use larger model (medium, large, xlarge)
- Increase image resolution
- Lower confidence threshold
- Use test-time augmentation
- Fine-tune on custom dataset

#### 5. Model Not Using GPU

```python
# Verify GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Force GPU usage
model = YOLO("yolov8n.pt")
model = model.to("cuda:0")

# Or
results = model(image, device=0)
```

---

## COCO Dataset Classes

Total: 80 classes

```
0: person
1: bicycle
2: car
3: motorbike
4: aeroplane
5: bus
6: train
7: truck
8: boat
9: traffic light
10: fire hydrant
11: stop sign
12: parking meter
13: bench
14: cat
15: dog
16: horse
17: sheep
18: cow
19: elephant
20: bear
21: zebra
22: giraffe
23: backpack
24: umbrella
25: handbag
26: tie
27: suitcase
28: frisbee
29: skis
30: snowboard
... (up to 79)
```

---

## References

### Official Documentation
- [Ultralytics YOLO Docs](https://docs.ultralytics.com)
- [PyTorch Docs](https://pytorch.org/docs/)
- [OpenCV Docs](https://docs.opencv.org/)

### Research Papers
- YOLOv8: [Ultralytics Paper](https://arxiv.org/abs/2305.09972)
- Original YOLO: [You Only Look Once](https://arxiv.org/abs/1506.02640)
- COCO: [Common Objects in Context](https://arxiv.org/abs/1405.0312)

### Community Resources
- [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)
- [Discussions](https://github.com/ultralytics/ultralytics/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/yolo)

---

**Last Updated**: January 2026
**Version**: 1.0
