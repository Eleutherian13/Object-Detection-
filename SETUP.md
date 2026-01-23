# Setup and Installation Guide

## System Requirements

### Minimum Requirements
- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 8 GB
- **Disk Space**: 2 GB (including models and dependencies)

### Recommended Requirements (for GPU acceleration)
- **GPU**: NVIDIA GPU with CUDA support (RTX 30 series or better)
- **CUDA**: 11.8 or 12.1
- **cuDNN**: 8.0+
- **GPU Memory**: 2+ GB VRAM

## Step-by-Step Installation

### Step 1: Prerequisites Installation

#### Option A: Using Python venv (Recommended)

```bash
# Navigate to project directory
cd "object Detection"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Verify activation (you should see (venv) in terminal)
```

#### Option B: Using Anaconda

```bash
# Create conda environment
conda create -n yolo_detection python=3.10

# Activate environment
conda activate yolo_detection
```

### Step 2: Install Dependencies

```bash
# Upgrade pip for latest package versions
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

**What gets installed:**
- `ultralytics` - YOLO models and tools
- `opencv-python` - Image processing
- `torch` - PyTorch framework
- `torchvision` - Computer vision utilities
- `jupyter` - Jupyter notebooks
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `pillow` - Image library

### Step 3: Verify Installation

```bash
# Check if ultralytics is installed
python -c "import ultralytics; print(ultralytics.__version__)"

# Run environment check
yolo checks

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
ultralytics 8.x.x
...PASSED...
PyTorch 2.x.x, CUDA available: True/False
```

## GPU Setup (Optional but Recommended)

### For NVIDIA GPU Users

#### Step 1: Install CUDA Toolkit

```bash
# Download from: https://developer.nvidia.com/cuda-downloads
# Choose your OS and follow installation instructions

# Verify CUDA installation
nvcc --version
```

#### Step 2: Install cuDNN

```bash
# Download from: https://developer.nvidia.com/cudnn
# Follow NVIDIA's installation guide for your OS
```

#### Step 3: Install GPU-enabled PyTorch

```bash
# Remove CPU-only PyTorch
pip uninstall torch torchvision torchaudio

# Install GPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Step 4: Verify GPU Access

```bash
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB' if torch.cuda.is_available() else 'N/A')
"
```

## Project Setup

### Step 1: Verify Project Structure

Ensure you have the following files in the `object Detection` folder:

```
object Detection/
├── README.md
├── SETUP.md
├── DOCUMENTATION.md
├── requirements.txt
├── YOLOintroduction.ipynb
├── PytorchIntroduction.ipynb
├── Pytorch_Demo.ipynb
├── yolov8n.pt (will download on first use)
├── yolo11s.pt (will download on first use)
└── runs/ (created on first inference)
```

### Step 2: Download Models (Optional)

Models download automatically on first use, but you can pre-download:

```bash
# Download YOLOv8 nano
yolo export model=yolov8n.pt format=torchscript

# Download YOLOv11 small
yolo export model=yolo11s.pt format=torchscript
```

### Step 3: Launch Jupyter Notebooks

```bash
# From the project directory with venv activated
jupyter notebook

# Or for JupyterLab
jupyter lab
```

Then:
1. Navigate to `YOLOintroduction.ipynb`
2. Click to open
3. Run cells sequentially (Shift + Enter)

## Troubleshooting

### Issue: "Python command not found"

**Windows:**
```bash
# Use python.exe or check PATH
where python
```

**macOS/Linux:**
```bash
# Check if Python 3 is installed
which python3
# Use: python3 instead of python
```

### Issue: "pip is not recognized"

**Solution:**
```bash
# Use python module directly
python -m pip install --upgrade pip

# Then install packages
python -m pip install -r requirements.txt
```

### Issue: Virtual environment not activating

**Windows:**
```bash
# Try with full path
.\venv\Scripts\activate.bat

# Or for PowerShell
.\venv\Scripts\Activate.ps1
```

**If PowerShell throws error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try activation again
```

### Issue: "ModuleNotFoundError" when importing

**Solution:**
```bash
# Make sure venv is activated (should see (venv) in terminal)

# Reinstall packages
pip install --force-reinstall -r requirements.txt

# Verify installation
python -c "import ultralytics, torch, cv2; print('All imports successful!')"
```

### Issue: Jupyter notebook kernel issues

**Solution:**
```bash
# Install ipykernel in your venv
pip install ipykernel

# Create kernel for Jupyter
python -m ipykernel install --user --name yolo_detection

# In Jupyter, select kernel: yolo_detection
```

### Issue: Model download fails

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'yolov8n.pt'
```

**Solution:**
```bash
# Set download directory
export YOLO_HOME=./models

# Or manually download from:
# https://github.com/ultralytics/assets/releases

# Place .pt files in the project directory
```

### Issue: Out of Memory (OOM)

**Solution:**
```bash
# Use smaller model
model = YOLO("yolov8n.pt")  # instead of large/xlarge

# Reduce image size
results = model(image, imgsz=416)  # instead of 640

# Process one image at a time instead of batch
```

### Issue: Slow inference on CPU

**Solution:**
```bash
# If GPU available, verify it's being used
# Check CUDA availability (see GPU Setup section)

# For CPU-only: use nano model
model = YOLO("yolov8n.pt")

# Reduce image resolution
results = model(image, imgsz=320)
```

### Issue: CUDA/GPU not detected

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with correct CUDA version
# Visit: https://pytorch.org/get-started/locally/
```

## Environment Variables

**Optional: Create .env file (for custom configurations)**

```bash
# .env file
YOLO_HOME=./models
YOLO_VERBOSE=False
YOLO_SETTINGS=default
```

## Next Steps

1. ✅ Complete all installation steps above
2. 📖 Read [README.md](README.md) for project overview
3. 🚀 Start with [YOLOintroduction.ipynb](YOLOintroduction.ipynb)
4. 📚 Reference [DOCUMENTATION.md](DOCUMENTATION.md) for technical details

## Getting Help

### Common Resources
- [Ultralytics Docs](https://docs.ultralytics.com)
- [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/yolo)

### Debug Commands

```bash
# Check Python version
python --version

# List installed packages
pip list

# Show environment info
python -c "import sys; print(f'Python {sys.version}')"

# Check available YOLO models
yolo model info yolov8n

# Test inference
yolo predict model=yolov8n.pt source=https://ultralytics.com/images/bus.jpg
```

## Performance Tips

### Speed Up Inference
```python
# Use nano model for speed
model = YOLO("yolov8n.pt")

# Lower resolution
results = model(image, imgsz=320)

# Batch processing
results = model(images_list)

# GPU utilization
model = model.to('cuda')
```

### Improve Accuracy
```python
# Use larger model
model = YOLO("yolov8l.pt")

# Higher resolution
results = model(image, imgsz=960)

# Lower confidence threshold
results = model(image, conf=0.25)
```

## Uninstalling / Cleanup

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
# Windows: rmdir /s venv
# macOS/Linux: rm -rf venv

# Remove cached models
rm -rf ~/.cache/ultralytics
```

## Version Information

### Tested Configuration
- Python 3.8 - 3.11
- PyTorch 2.0+
- CUDA 11.8 / 12.1
- Ultralytics 8.0+
- Windows 10/11, macOS 11+, Ubuntu 20.04+

### Known Issues
- Pre-trained models require ~500 MB on first download
- CUDA support varies by GPU model
- Some older GPUs may not support latest PyTorch

---

**Last Updated**: January 2026
**Version**: 1.0
