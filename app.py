import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("plant_disease_model_final.keras")

# Title
st.title("🌱 Plant Disease Detection")

st.write("Upload a plant leaf image to detect disease.")

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224,224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.success(f"Predicted class: {predicted_class}")
