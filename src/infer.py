import sys
import cv2
import joblib
import numpy as np

from features import sky_ratio, vegetation_ratio, edge_density, road_brightness

model = joblib.load("models/country_model.pkl")

def extract(img):
    return np.array([[
        sky_ratio(img),
        vegetation_ratio(img),
        edge_density(img),
        road_brightness(img)
    ]])

if __name__ == "__main__":
    img_path = sys.argv[1]

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Could not load image")

    X = extract(img)

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    print("\nPrediction:", pred)
    print("Confidence:", round(float(max(probs)), 3))
