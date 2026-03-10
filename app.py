import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Page configuration
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect plant disease")

# Load model
@st.cache_resource
def load_trained_model():
    model = load_model("model.h5")
    return model

model = load_trained_model()

# Example class labels (edit according to your dataset)
class_names = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Healthy Leaf"
]

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# File uploader
uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Disease"):

        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)

        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Prediction: {class_names[predicted_class]}")
        st.info(f"Confidence: {confidence*100:.2f}%")
