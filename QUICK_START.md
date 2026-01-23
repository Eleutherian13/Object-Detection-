# Quick Start Guide

Get up and running in 5 minutes!

## 📦 Installation (2 minutes)

### Step 1: Clone/Download Project
```bash
# Navigate to the project directory
cd "object Detection"
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Run Your First Detection (2 minutes)

### Option A: Command Line (Fastest)
```bash
# Download and detect objects in a single command
yolo predict model=yolov8n.pt source="https://images.pexels.com/photos/13872248/pexels-photo-13872248.jpeg"

# Check results in: runs/detect/predict/
```

### Option B: Jupyter Notebook (Recommended for Learning)
```bash
# Launch Jupyter
jupyter notebook

# Open YOLOintroduction.ipynb
# Run cells sequentially (Shift + Enter)
```

## 📝 Simple Python Example (1 minute)

Create a file `test_detection.py`:

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Run detection on image
results = model("image.jpg")

# Print results
for result in results:
    print(f"Found {len(result.boxes)} objects")
    for i, box in enumerate(result.boxes):
        class_name = result.names[int(box.cls)]
        confidence = float(box.conf)
        print(f"  {i+1}. {class_name} - {confidence:.2f}")
```

Run it:
```bash
python test_detection.py
```

## 🎯 Common Tasks

### Detect Objects in Your Own Image
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("my_photo.jpg")

# Save annotated result
results[0].save("output.jpg")
```

### Detect in Video
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("video.mp4", save=True)
```

### Webcam Detection (Real-time)
```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Filter High-Confidence Detections
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("image.jpg")

for result in results:
    boxes = result.boxes
    
    # Keep only detections with >70% confidence
    high_conf = boxes.conf > 0.7
    
    for i, is_high_conf in enumerate(high_conf):
        if is_high_conf:
            box = boxes.xyxy[i]
            conf = boxes.conf[i]
            cls = result.names[int(boxes.cls[i])]
            print(f"{cls}: {conf:.2f}")
```

### Detect Specific Objects Only
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("image.jpg")

# COCO class IDs: 0=person, 2=car, 15=dog, etc.
# See DOCUMENTATION.md for full list
TARGET_CLASSES = [0, 2, 15]  # person, car, dog

for result in results:
    boxes = result.boxes
    
    for i in range(len(boxes)):
        if int(boxes.cls[i]) in TARGET_CLASSES:
            class_name = result.names[int(boxes.cls[i])]
            confidence = boxes.conf[i]
            bbox = boxes.xyxy[i]
            print(f"{class_name}: {confidence:.2f} at {bbox}")
```

## 🎓 Next Steps

1. ✅ Complete Quick Start above
2. 📖 Read [README.md](README.md) for project overview
3. 📚 Open [YOLOintroduction.ipynb](YOLOintroduction.ipynb) in Jupyter
4. 🔧 Reference [SETUP.md](SETUP.md) for detailed installation
5. 📋 Check [DOCUMENTATION.md](DOCUMENTATION.md) for advanced usage

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'ultralytics'"
```bash
# Make sure venv is activated (should see (venv) in terminal)
pip install ultralytics
```

### "CUDA error" (GPU issues)
```python
# Use CPU instead
model = YOLO("yolov8n.pt")
results = model("image.jpg", device="cpu")
```

### Model download fails
```python
# Specify model path
import os
os.environ['YOLO_HOME'] = './'
model = YOLO("yolov8n.pt")
```

## 💡 Tips

- **Start with nano model** (`yolov8n.pt`) for speed
- **Use small model** (`yolov11s.pt`) for better accuracy
- **Enable GPU** for 10-50x faster inference
- **Lower image size** (320 instead of 640) for faster processing
- **Batch process** images for efficiency

## 📚 Model Comparison

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| yolov8n | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Real-time |
| yolo11s | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Production |
| yolov8m | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Balanced |
| yolov8l | ⚡⚡ | ⭐⭐⭐⭐⭐ | Accuracy |

## ❓ FAQ

**Q: Can I detect custom objects?**
A: Yes! Fine-tune models on your custom dataset. See DOCUMENTATION.md.

**Q: Do I need GPU?**
A: No, but it's 10-50x faster. CPU works fine for learning.

**Q: How do I save detected objects?**
A: Use XYXY coordinates to crop images (see DOCUMENTATION.md).

**Q: Can I track objects across frames?**
A: Yes! Use `model.track()` instead of `model.predict()`.

**Q: What classes can YOLO detect?**
A: 80 COCO classes (person, car, dog, cat, etc.). See DOCUMENTATION.md.

---

**Need help?** See [SETUP.md](SETUP.md) and [DOCUMENTATION.md](DOCUMENTATION.md) for detailed guides.

Happy detecting! 🎉
