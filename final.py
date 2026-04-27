import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io


# Page config
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Theme CSS
st.markdown(
    """
    <style>
    /* File uploader (Image Upload option) */
    [data-testid="stFileUploader"] {
        background-color: #f4f7f9;  
        border-radius: 8px;
        padding: 1em;
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }

    /* File uploader label text */
    div[data-testid="stFileUploader"] > label {
        color: #1a73e8 !important;   /* Professional blue */
        font-weight: 600;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Uploader text */
    [data-testid="stFileUploader"] section {
        color: #1a73e8;
        font-weight: 500;
    }

    /* Drag & drop area hover effect */
    [data-testid="stFileUploader"] div:hover {
        background-color: #e8f0fe;
        border-color: #155ab6;
    }

    /* Warning / Alert box text */
    div[role="alert"] {
        color: #2c3e50 !important;   /* Dark grey text */
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        font-weight: 500;
    }

    


    /* App background */
    .stApp {
        background: linear-gradient(135deg, #f4f7f9, #e9eef2);
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
        color: #ecf0f1;
    }

    /* Sidebar radio buttons */
    .stRadio > label {
        color: #ecf0f1 !important;
        font-weight: 500;
    }

    /* Titles */
    h1, h2, h3 {
        color: #1a73e8;
        font-family: 'Segoe UI Semibold', sans-serif;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 6px;
        font-size: 15px;
        font-weight: 500;
        border: none;
        padding: 0.5em 1.2em;
    }
    .stButton>button:hover {
        background-color: #155ab6;
        color: #fff;
    }

    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background-color: #e8f0fe;
        color: #1a73e8;
        border-left: 5px solid #1a73e8;
    }

    /* Model Info summary text */
    div[data-testid="stMarkdownContainer"] pre {
        color: #2c3e50 !important;   /* Dark grey text */
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 14px;
        background-color: #f4f7f9;   /* हल्का background */
        border-radius: 6px;
        padding: 1em;
    }

    .st-emotion-cache-d8lm1x{
    color: #000
    }
    .st-emotion-cache-1fi65ho{
    color: #000
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load trained model
MODEL_PATH = "mask_detector_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

st.title("Face Mask Detection App")
st.sidebar.title("Options")
choice = st.sidebar.radio("Select Mode:", ["Image Upload", "Webcam Detection", "Model Info"])

# ---------------- IMAGE UPLOAD ----------------
if choice == "Image Upload":
    st.header("Upload an Image for Prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=500)

        # Preprocess
        img_array = np.array(image.convert("RGB"))
        img_resized = cv2.resize(img_array, (224, 224))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)

        # Prediction
        prediction = model.predict(img_expanded)
        result = "Mask Detected 😷" if prediction[0][0] < 0.5 else "No Mask Detected ❌"
        st.subheader("Prediction:")
        st.success(result)
        

# ---------------- WEBCAM DETECTION ----------------
elif choice == "Webcam Detection":
    st.header("Webcam Live Detection")
    st.write("Take a snapshot from your webcam")

    camera_image = st.camera_input("Capture Image")
    if camera_image:
        image = Image.open(camera_image)
        st.image(image, caption="Webcam Capture", width=250)

        # Preprocess frame
        img_array = np.array(image.convert("RGB"))
        img_resized = cv2.resize(img_array, (224, 224))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)

        prediction = model.predict(img_expanded)
        label = "Mask Detected 😷" if prediction[0][0] < 0.5 else "No Mask Detected ❌"

        st.success(label)


# ---------------- MODEL INFO ----------------
elif choice == "Model Info":
    st.header("CNN Model Summary")
    st.text("Architecture loaded from CNN.py")
    st.write("Model Name: sequential_4")
    st.write("Total Parameters: 2265285")
    st.write("✅ Train Accuracy: 0.95")
    st.write("ℹ️ Validation Accuracy: 0.93")
    st.write("⚠️ Test Accuracy: 0.92")



