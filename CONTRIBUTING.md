# Contributing Guidelines

Thank you for your interest in contributing to the Object Detection project!

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [FAQ](#faq)

---

## Code of Conduct

### Our Pledge
- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Welcome contributors of all backgrounds

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Any form of abuse

---

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- Familiarity with object detection or deep learning (helpful but not required)

### Fork and Clone
```bash
# Fork the repository on GitHub

# Clone your fork
git clone https://github.com/YOUR_USERNAME/object-detection.git
cd "object Detection"

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/object-detection.git
```

---

## Development Setup

### Step 1: Create Development Environment
```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Development Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pytest-cov black flake8 mypy
```

### Step 3: Verify Setup
```bash
# Test imports
python -c "import ultralytics, torch, cv2; print('✓ All imports successful')"

# Run tests (if available)
pytest tests/
```

---

## Making Changes

### Step 1: Create Feature Branch
```bash
# Sync with upstream
git fetch upstream
git checkout upstream/master

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### Branch Naming
- Features: `feature/feature-name`
- Bug fixes: `fix/bug-description`
- Improvements: `improve/improvement-name`
- Documentation: `docs/doc-description`

### Step 2: Make Your Changes

#### For Code Changes
- Write clear, readable code
- Add comments for complex logic
- Follow existing code style
- Keep changes focused and minimal

#### For Documentation Changes
- Use proper Markdown formatting
- Include code examples where relevant
- Update table of contents if adding sections
- Spell-check before submitting

#### For Notebook Changes
- Add markdown cells explaining each section
- Include comments in code cells
- Test notebook execution before submitting
- Keep output clean (restart kernel before saving)

### Step 3: Test Your Changes
```bash
# Test code
python test_your_feature.py

# Test notebooks
jupyter notebook YOLOintroduction.ipynb
# Run all cells and verify no errors

# Code quality
black .
flake8 .
mypy .
```

---

## Submitting Changes

### Step 1: Commit Your Changes
```bash
# Review changes
git status
git diff

# Stage changes
git add .

# Commit with clear message
git commit -m "feat: add feature description"
# or
git commit -m "fix: resolve bug description"
# or
git commit -m "docs: update documentation"
```

### Commit Message Guidelines
- **Format**: `[type]: [description]`
- **Types**: feat, fix, docs, style, refactor, perf, test
- **Description**: Clear, imperative, lowercase
- **Length**: First line max 50 chars

**Examples:**
```
feat: add confidence filtering to detection results
fix: resolve CUDA memory error in batch processing
docs: add custom training section to DOCUMENTATION.md
```

### Step 2: Push and Create PR
```bash
# Push to your fork
git push origin feature/your-feature-name

# Go to GitHub and create Pull Request
# Target branch: master or development
```

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type
- [ ] Feature
- [ ] Bug Fix
- [ ] Documentation
- [ ] Performance Improvement

## Related Issues
Fixes #(issue number)

## Testing
Describe testing performed:
- [ ] Tested locally
- [ ] All notebooks run without errors
- [ ] No breaking changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests pass
- [ ] No new warnings generated
```

---

## Coding Standards

### Python Style
Follow PEP 8 with these tools:
```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .
```

### Code Example
```python
"""
Module docstring explaining purpose.
"""

from typing import List, Dict
import numpy as np
from ultralytics import YOLO


def detect_objects(image_path: str, confidence: float = 0.5) -> List[Dict]:
    """
    Detect objects in an image.
    
    Args:
        image_path: Path to image file
        confidence: Confidence threshold (0-1)
        
    Returns:
        List of detection dictionaries with keys:
        - class_name: str
        - confidence: float
        - bbox: dict with x1, y1, x2, y2
        
    Raises:
        FileNotFoundError: If image not found
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    model = YOLO("yolov8n.pt")
    results = model(image_path, conf=confidence)
    
    detections = []
    for result in results:
        for i, box in enumerate(result.boxes):
            detection = {
                'class_name': result.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': {
                    'x1': float(box.xyxy[0, 0]),
                    'y1': float(box.xyxy[0, 1]),
                    'x2': float(box.xyxy[0, 2]),
                    'y2': float(box.xyxy[0, 3]),
                }
            }
            detections.append(detection)
    
    return detections
```

### Docstring Format
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    One-line summary of function.
    
    Longer description if needed. Explain what the function does,
    how to use it, and any important notes.
    
    Args:
        param1: Description of param1 and type
        param2: Description of param2 and type
        
    Returns:
        Description of return value and type
        
    Raises:
        ExceptionType: When this exception occurs
        
    Examples:
        >>> result = function_name(1, 2)
        >>> print(result)
        output
    """
    pass
```

---

## Testing

### Writing Tests
```python
# tests/test_detection.py
import pytest
from ultralytics import YOLO


def test_model_load():
    """Test model loading."""
    model = YOLO("yolov8n.pt")
    assert model is not None


def test_inference():
    """Test inference on sample image."""
    model = YOLO("yolov8n.pt")
    results = model("test_image.jpg")
    assert len(results) > 0


@pytest.mark.slow
def test_batch_inference():
    """Test batch inference."""
    model = YOLO("yolov8n.pt")
    images = ["img1.jpg", "img2.jpg"]
    results = model(images)
    assert len(results) == 2
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test
pytest tests/test_detection.py::test_model_load

# Run with output
pytest -v
```

---

## Documentation

### Documentation Updates Required
- Update README.md if adding features
- Update DOCUMENTATION.md with API changes
- Update SETUP.md if changing requirements
- Add docstrings to new functions
- Update relevant notebooks

### Documentation Style
- Use clear, simple language
- Provide code examples
- Include relevant links
- Format properly with Markdown
- Include visuals when helpful

### Example Documentation
```markdown
## New Feature Name

### Overview
Brief explanation of feature.

### Usage
```python
# Code example
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| param1 | str | "default" | What this does |

### Returns
Description of return value

### Examples
Practical examples of usage
```

---

## FAQ

**Q: How do I get a feature approved?**
A: Create a detailed PR with tests and documentation. Maintainers will review and provide feedback.

**Q: What if I'm stuck?**
A: Open a discussion or create an issue. The community is here to help!

**Q: How long does review take?**
A: Usually 3-7 days depending on complexity. Be patient and responsive to feedback.

**Q: Can I contribute without coding?**
A: Yes! Documentation, testing, and reporting bugs are valuable contributions.

**Q: How do I report a bug?**
A: Create an issue with:
- Clear title
- Steps to reproduce
- Expected behavior
- Actual behavior
- System info (OS, Python version, etc.)

**Q: How often are releases made?**
A: Typically monthly for minor updates, quarterly for major releases.

---

## Contributors

We appreciate all contributors! See [GitHub Contributors](https://github.com/your-repo/graphs/contributors)

---

## License

By contributing, you agree to license your work under the project's license.

---

**Last Updated**: January 2026
**Version**: 1.0

Thank you for contributing! 🎉
