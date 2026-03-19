<h1 align="center">BloodPrint</h1>
<p align="center" style="margin-top:30px;">
  <img src="https://github.com/user-attachments/assets/766cc9e1-6ab8-4bc8-82e8-c59d37135531" height="300cm"/>
</p>
This project detects blood groups from fingerprint images using deep learning models. It implements both PyTorch and TensorFlow solutions with multiple pre-trained architectures. The system leverages fingerprint patterns to predict blood types efficiently.

## Key Features:
- **Multiple Frameworks**: PyTorch and TensorFlow implementations
- **Pre-trained Models**: EfficientNetB0, ResNet50, DenseNet121, InceptionV3, MobileNetV2, VGG16
- **Custom Architecture**: Specialized model for blood group detection
- **Trained Models**: Ready-to-use `.pth` and `.h5` model files
- **Comprehensive Evaluation**: Accuracy metrics, confusion matrices, and classification reports

## Repository Structure:
The following is repository structure:
```
BloodPrint/
├── code/
│   ├── pytorch.ipynb          # PyTorch implementation with custom model
│   └── tensorflow.ipynb       # TensorFlow implementation with multiple models
├── dataset/
│   └── Dataset.md             # Dataset description and preprocessing info
├── model/
│   ├── pytorch.pth            # Trained PyTorch model weights (25.1 MB)
│   └── tensorflow.h5         # Trained TensorFlow model weights (42.1 MB)
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies

## Model Files:

### PyTorch Model (`model/pytorch.pth`)
- **Size**: 25.1 MB
- **Framework**: PyTorch
- **Architecture**: Custom CNN for blood group detection
- **Classes**: 8 blood types (A+, A-, B+, B-, AB+, AB-, O+, O-)

### TensorFlow Model (`model/tensorflow.h5`)
- **Size**: 42.1 MB  
- **Framework**: TensorFlow/Keras
- **Architecture**: Ensemble of pre-trained models
- **Classes**: 8 blood types (A+, A-, B+, B-, AB+, AB-, O+, O-)

## Installation & Usage:

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Setup
```bash
# Clone repository
git clone https://github.com/your-username/BloodPrint.git
cd BloodPrint

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebooks
jupyter notebook code/
```

### Quick Start
```python
# Load PyTorch model
import torch
model = torch.load('model/pytorch.pth')
model.eval()

# Load TensorFlow model  
import tensorflow as tf
model = tf.keras.models.load_model('model/tensorflow.h5')
```

## Results:

### Model Performance
- **PyTorch Custom Model**: 85% accuracy
- **TensorFlow High-Accuracy Model**: 83% validation accuracy

### Training Details
- **Dataset**: Fingerprint images for 8 blood types (A+, A-, B+, B-, AB+, AB-, O+, O-)
- **Image Size**: 64x64 pixels
- **Training Epochs**: 50
- **Batch Size**: 32

### Classification Report & Confusion Matrix
See `code/pytorch.ipynb` and `code/tensorflow.ipynb` for detailed classification reports and confusion matrices generated during model evaluation.

### Model Files
- `model/pytorch.pth`: Trained PyTorch model (25.1 MB)
- `model/tensorflow.h5`: Trained TensorFlow model (42.1 MB)
