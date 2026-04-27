import streamlit as st
import os
import sys
import tempfile
from PIL import Image

# Add the root directory to the sys path so we can import from scripts
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from scripts.predict import predict_pytorch, predict_tensorflow

st.set_page_config(page_title="BloodPrint Predictor", page_icon="🩸", layout="centered")

st.title("🩸 BloodPrint Predictor")
st.write("Detect blood groups from fingerprint images using Deep Learning.")

st.sidebar.header("Settings")
framework = st.sidebar.radio("Select Framework", ("PyTorch", "TensorFlow"))

uploaded_file = st.file_uploader("Upload a Fingerprint Image", type=["bmp", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Fingerprint", use_column_width=True)

    if st.button("Predict Blood Group"):
        with st.spinner("Analyzing fingerprint..."):
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bmp") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_image_path = temp_file.name

            try:
                if framework == "PyTorch":
                    model_path = os.path.join(root_dir, 'model', 'pytorch.pth')
                    label, confidence = predict_pytorch(temp_image_path, model_path)
                else:
                    model_path = os.path.join(root_dir, 'model', 'tensorflow.h5')
                    label, confidence = predict_tensorflow(temp_image_path, model_path)

                st.success(f"**Predicted Blood Group:** {label}")
                st.info(f"**Confidence:** {confidence:.2%}")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
