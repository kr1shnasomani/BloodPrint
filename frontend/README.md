# BloodPrint Frontend (Demo)

This is a Streamlit-based web interface for the BloodPrint project, designed strictly for demonstration purposes to your panel members.

It is entirely self-contained inside the `frontend/` folder, so you can easily delete the entire folder once your minor project showcase is over.

## Setup Instructions

Make sure you have activated your project's main virtual environment (`venv`).

Then, install the UI dependency:

```bash
pip install -r frontend/requirements.txt
```

## Running the App

To start the user interface, run the following command from the **root directory** of your project (`BloodPrint/`):

```bash
streamlit run frontend/app.py
```

This will automatically open the Streamlit application in your default web browser. You can then select your model (PyTorch/TensorFlow), upload a fingerprint image from the `examples/` folder, and instantly view the predicted blood group!
