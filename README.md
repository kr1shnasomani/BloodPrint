<h1 align="center">BloodPrint</h1>
<p align="center" style="margin-top:30px;">
  <img src="https://github.com/user-attachments/assets/766cc9e1-6ab8-4bc8-82e8-c59d37135531" height="300cm"/>
</p>
This project detects blood groups from fingerprint images using deep learning models. It implements both PyTorch and TensorFlow solutions. The system leverages fingerprint patterns to predict blood types efficiently.

## Key Features
- **Multiple Frameworks**: PyTorch and TensorFlow implementations
- **Custom Architecture**: Specialized ResNet9 PyTorch model for blood group detection
- **Pre-trained Models**: EfficientNetB0, ResNet50, DenseNet121, InceptionV3, MobileNetV2, VGG16 (via TensorFlow ensemble)
- **Ready-to-use CLI**: Unified inference script (`scripts/predict.py`) for easy testing

## Repository Structure
```
BloodPrint/
├── src/
│   ├── __init__.py            # Package initialization
│   └── models.py              # PyTorch model architectures
├── scripts/
│   └── predict.py             # Unified CLI for inference (PyTorch & TF)
├── notebooks/                 # Source of truth for training & evaluation
│   ├── pytorch.ipynb
│   └── tensorflow.ipynb
├── model/                     # Trained Model Weights
│   ├── pytorch.pth            # Trained PyTorch model weights (25.1 MB)
│   └── tensorflow.h5          # Trained TensorFlow model weights (42.1 MB)
├── examples/                  # Sample test images
│   └── a+.bmp
├── papers/                    # Research papers and related documents
├── README.md                  # Project documentation (You are here)
└── requirements.txt           # Python dependencies
```

## Setup Instructions

### Prerequisites
- macOS / Linux / Windows
- Python 3.10 or 3.11 (TensorFlow is not yet fully supported on Python 3.12+ for all architectures)
- CUDA-compatible GPU (optional, recommended for training)

### 1. Clone the repository
```bash
git clone https://github.com/kr1shnasomani/BloodPrint.git
cd BloodPrint
```

### 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies and avoid version conflicts.

**For macOS/Linux:**
```bash
# Create the virtual environment using Python 3.11
python3.11 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

**For Windows:**
```cmd
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

### 3. Install Dependencies
Once the virtual environment is activated (you should see `(venv)` in your terminal prompt), install the required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage (Inference)

You can test the models using the provided `predict.py` script. The script automatically handles the correct image resizing required by each framework (128x128 for PyTorch, 64x64 for TensorFlow).

**Using PyTorch (Default):**
```bash
python scripts/predict.py examples/a+.bmp
```

**Using TensorFlow:**
```bash
python scripts/predict.py examples/a+.bmp --framework tensorflow
```

**Expected Output Example:**
```
File: examples/a+.bmp
Framework: Pytorch
Predicted Blood Group: A+ (Confidence: 99.85%)
```

## Results & Architecture

### PyTorch Custom Model (`model/pytorch.pth`)
- **Architecture**: Custom ResNet9
- **Input Size**: 128x128
- **Accuracy**: 85%

### TensorFlow Model (`model/tensorflow.h5`)
- **Architecture**: Ensemble of pre-trained models
- **Input Size**: 64x64
- **Validation Accuracy**: 83%

See `notebooks/pytorch.ipynb` and `notebooks/tensorflow.ipynb` for detailed classification reports and confusion matrices.
