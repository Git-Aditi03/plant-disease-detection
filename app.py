import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🌿 Plant Disease Detection System")
st.write("Upload a plant leaf image and the model will predict the disease.")

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_model_final.keras")
    return model

model = load_model()

# -----------------------------
# Class Labels (EDIT if needed)
# -----------------------------
class_names = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Healthy"
]

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Disease"):

        with st.spinner("Analyzing image..."):

            processed_image = preprocess_image(image)

            prediction = model.predict(processed_image)

            predicted_class = class_names[np.argmax(prediction)]

            confidence = np.max(prediction) * 100

        st.success("Prediction Complete")

        st.subheader("🩺 Prediction Result")
        st.write("Disease:", predicted_class)
        st.write(f"Confidence: {confidence:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("Developed for Plant Disease Detection using Deep Learning")
