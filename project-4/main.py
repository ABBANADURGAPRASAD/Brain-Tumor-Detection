import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained ResNet model
MODEL_PATH = "brain_tumor_model.h5"  # Use ResNet-trained model
model = load_model(MODEL_PATH)

# Class labels
TUMOR_CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
STAGE_CLASSES = ['Early Stage', 'Final Stage']

# Function to preprocess MRI image
def preprocess_image(image):
    image = cv2.resize(image, (224,224))  # Resize to match model input
    image = img_to_array(image)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Brain Tumor Detection System (ResNet)")
st.write("Upload an MRI image to detect tumor presence, type, and stage.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Predict tumor type
    predictions = model.predict(processed_image)
    tumor_index = np.argmax(predictions)
    tumor_type = TUMOR_CLASSES[tumor_index]
    
    # Predict tumor stage
    confidence = predictions[0][tumor_index]
    tumor_stage = STAGE_CLASSES[1] if confidence > 0.75 else STAGE_CLASSES[0]
    
    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Tumor Type:** {tumor_type}")
    st.write(f"**Tumor Stage:** {tumor_stage}")
