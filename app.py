import streamlit as st
import cv2
import joblib
import numpy as np

# Import feature functions from src/
from src.features import sky_ratio, vegetation_ratio, edge_density

# Load trained model
model = joblib.load("models/country_model.pkl")

# Page title
st.title("Visual Geolocation Inference")
st.write("Upload a street-view image and the model will guess the country.")

# Upload box
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "png", "jpeg"]
)

# Feature extraction function
def extract_features(img):
    sky = sky_ratio(img)
    veg = vegetation_ratio(img)
    edges = edge_density(img)

    return np.array([[sky, veg, edges]])

# Main prediction block
if uploaded_file is not None:

    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width=700)

    # Convert uploaded file into OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Extract features
    X = extract_features(img)

    # Predict probabilities
    probs = model.predict_proba(X)[0]
    classes = model.classes_

    # Sort top 3 predictions
    top3_idx = np.argsort(probs)[::-1][:3]

    # Display results
    st.subheader("Top Predictions")

    for rank, i in enumerate(top3_idx, start=1):
        st.write(f"**{rank}. {classes[i]}** — {probs[i]*100:.1f}%")

    # Best guess
    best_country = classes[top3_idx[0]]
    st.success(f"Final Prediction: {best_country}")
