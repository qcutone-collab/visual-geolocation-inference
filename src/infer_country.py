import sys
import cv2
import joblib
import numpy as np
from features import sky_ratio, vegetation_ratio, edge_density

# Load saved model
model = joblib.load("models/country_model.pkl")

def extract_features(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError("Could not load image: " + img_path)

    sky = sky_ratio(img)
    veg = vegetation_ratio(img)
    edges = edge_density(img)

    return np.array([[sky, veg, edges]])

if __name__ == "__main__":
    img_path = sys.argv[1]

    X = extract_features(img_path)

    prediction = model.predict(X)[0]

    print("\n--- GEOLOCATION GUESS ---")
    print("Predicted country:", prediction)
