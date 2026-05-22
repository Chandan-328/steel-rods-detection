# Steel Rods Detection

<img width="1528" height="936" alt="Screenshot 2026-02-23 144012" src="https://github.com/user-attachments/assets/70a9f21c-c3f1-4f0a-9155-ddc94c262601" />

## Overview

A machine learning-based computer vision system for detecting and analyzing steel rods in images or video streams. This project uses advanced object detection techniques to identify, localize, and classify steel rods with high accuracy.

## Features

- **Real-time Detection**: Fast inference on images and video streams
- **High Accuracy**: Optimized detection model for various lighting and angles
- **Easy Integration**: Simple API for integration into existing pipelines
- **Visualization**: Built-in visualization tools for results

## Prerequisites

- Python 3.8+
- PyTorch or TensorFlow (depending on implementation)
- OpenCV
- NumPy

## Installation

```bash
# Clone the repository
git clone https://github.com/Chandan-328/steel-rods-detection.git
cd steel-rods-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Detection

```python
from steel_detector import SteelRodDetector

# Initialize detector
detector = SteelRodDetector(model_path='path/to/model')

# Detect rods in an image
results = detector.detect('image.jpg')

# Visualize results
detector.visualize(results)
```

### From Command Line

```bash
python detect.py --image path/to/image.jpg --output results/
```

## Model Details

- **Architecture**: [Specify your model - e.g., YOLOv8, Faster R-CNN, etc.]
- **Training Data**: [Number of images, dataset source]
- **Performance**: [mAP, precision, recall metrics]


## Project Structure

```
steel-rods-detection/
├── README.md
├── requirements.txt
├── src/
│   ├── detect.py
│   ├── model.py
│   └── utils.py
├── models/
│   └── steel_detector.pt
└── examples/
    └── sample_images/
```

