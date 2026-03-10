import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

# Page configuration
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌿 Plant Disease Detection System")
st.write("Upload a plant leaf image and the model will detect the disease.")

# Load trained model
model = load_model("plant_disease_model_final.keras")

# Disease classes (edit according to your dataset)
classes = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Healthy Leaf"
]

# Disease descriptions
descriptions = {
    "Apple Scab": "A fungal disease causing dark lesions on leaves.",
    "Apple Black Rot": "Causes black rot spots on apple leaves and fruit.",
    "Apple Cedar Rust": "Fungal infection producing yellow-orange spots.",
    "Healthy Leaf": "The plant leaf is healthy."
}

# Treatment suggestions
treatments = {
    "Apple Scab": "Use fungicide and remove infected leaves.",
    "Apple Black Rot": "Prune infected branches and apply fungicide.",
    "Apple Cedar Rust": "Remove nearby cedar trees and spray fungicide.",
    "Healthy Leaf": "No treatment needed. Maintain proper plant care."
}

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    img = np.array(image)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Upload image
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    if st.button("🔍 Predict Disease"):

        img = preprocess_image(image)

        prediction = model.predict(img)

        index = np.argmax(prediction)

        confidence = np.max(prediction)

        disease = classes[index]

        st.success(f"Prediction: {disease}")

        st.info(f"Confidence: {confidence*100:.2f}%")

        st.subheader("📖 Disease Description")
        st.write(descriptions[disease])

        st.subheader("💊 Treatment Suggestion")
        st.write(treatments[disease])
