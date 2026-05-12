import sys
import cv2
import joblib
import numpy as np
try:
    from src.country_features import extract_country_features
except ModuleNotFoundError:
    from country_features import extract_country_features  # type: ignore

# Load saved model
model = joblib.load("models/country_model.pkl")

def extract_features(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError("Could not load image: " + img_path)

    return np.array([extract_country_features(img)])

if __name__ == "__main__":
    img_path = sys.argv[1]

    X = extract_features(img_path)

    prediction = model.predict(X)[0]

    print("\n--- GEOLOCATION GUESS ---")
    print("Predicted country:", prediction)
