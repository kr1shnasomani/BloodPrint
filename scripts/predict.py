import argparse
import os
import sys

# Constants based on training notebooks
CLASSES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
PYTORCH_IMG_SIZE = (128, 128)
TF_IMG_SIZE = (64, 64)

def predict_pytorch(image_path, model_path):
    import torch
    from torchvision.transforms import Compose, Resize, ToTensor
    from PIL import Image
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.models import load_pytorch_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_pytorch_model(model_path, len(CLASSES), device)

    transform = Compose([
        Resize(PYTORCH_IMG_SIZE),
        ToTensor()
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, preds = torch.max(output, dim=1)
        prob = torch.nn.functional.softmax(output, dim=1)
        confidence = prob[0][preds[0]].item()

    return CLASSES[preds[0]], confidence

def predict_tensorflow(image_path, model_path):
    import tensorflow as tf
    import numpy as np
    from PIL import Image

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = tf.keras.models.load_model(model_path)
    
    # Keras load_img equivalent
    img = Image.open(image_path).convert('RGB')
    img = img.resize(TF_IMG_SIZE)
    img_array = np.array(img)
    
    # TensorFlow models usually expect batch dimension and scaled inputs (0-1)
    # Checking how TF model was trained (assuming Rescaling layer inside or scaled externally)
    # The notebook implies preprocessing might be needed, we'll try standard scaling if it fails
    img_array = np.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array, verbose=0)
    
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    
    return CLASSES[predicted_class_idx], confidence

def main():
    parser = argparse.ArgumentParser(description="Predict Blood Group from Fingerprint Image")
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    parser.add_argument("--framework", type=str, choices=["pytorch", "tensorflow"], default="pytorch",
                        help="Which framework to use for inference (default: pytorch)")
    parser.add_argument("--model_path", type=str, help="Path to the trained model file. Defaults to models/pytorch.pth or models/tensorflow.h5 depending on framework.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        sys.exit(1)

    # Determine default model path if not provided
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    if args.model_path is None:
        if args.framework == 'pytorch':
            args.model_path = os.path.join(base_dir, 'model', 'pytorch.pth')
        else:
            args.model_path = os.path.join(base_dir, 'model', 'tensorflow.h5')

    try:
        if args.framework == 'pytorch':
            label, conf = predict_pytorch(args.image_path, args.model_path)
        else:
            label, conf = predict_tensorflow(args.image_path, args.model_path)
            
        print(f"File: {args.image_path}")
        print(f"Framework: {args.framework.capitalize()}")
        print(f"Predicted Blood Group: {label}")
        print(f"Confidence: {conf:.2%}")
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
