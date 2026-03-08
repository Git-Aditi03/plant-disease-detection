import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("plant_disease_model_final.keras")

# Disease classes
class_names = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Apple Healthy",
    "Corn Gray Leaf Spot",
    "Corn Common Rust",
    "Corn Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]

# Streamlit UI
st.title("🌱 Plant Disease Detection")
st.write("Upload a plant leaf image and the AI model will predict the disease.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    disease_name = class_names[predicted_index]

    # Show results
    st.success(f"Predicted Disease: {disease_name}")
    st.info(f"Confidence: {confidence:.2f}%")
