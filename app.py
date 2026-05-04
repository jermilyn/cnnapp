import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- CONFIGURATION ---
MODEL_PATH = 'herb_model_lab5.h5'
IMG_SIZE = (120, 120)
# Make sure this matches the order of your folders in DATA_DIR
CLASS_NAMES = [
    'Cardiospermum halicacabum (Balloon Vine)',
    'Curcuma longa (Turmeric)',
    'Mentha (Mint)',
    'Spinacia oleracea (Palak(Spinach))',
    'Zingiber officinale (Ginger)'
]

# --- APP LAYOUT ---
st.set_page_config(page_title="Medicinal Herb Identifier", layout="centered")
st.title("🌿 Medicinal Herb Classifier")
st.write("Upload an image of a herb, and the AI will identify its category.")

# --- LOAD MODEL ---
@st.cache_resource
def load_herb_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_herb_model()

if model is None:
    st.error(f"Model file '{MODEL_PATH}' not found. Please run your training script first.")
else:
    # --- IMAGE UPLOAD ---
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Herb Image', use_container_width=True)
        
        # --- PREDICTION LOGIC ---
        with st.spinner('Analyzing...'):
            # Preprocess the image to match model requirements
            img = image.resize(IMG_SIZE)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            # Generate prediction (Classification)
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0]) # Convert logits to probabilities
            
            result_class = CLASS_NAMES[np.argmax(score)]
            confidence = 100 * np.max(score)

        # --- DISPLAY RESULTS ---
        st.success(f"**Prediction:** {result_class}")
        st.info(f"**Confidence Level:** {confidence:.2f}%")
        
        # Progress bar for visual feedback
        st.progress(int(confidence))