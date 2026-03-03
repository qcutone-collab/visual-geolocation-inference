import cv2
import numpy as np


# ----------------------------
# Feature 1: Sky Ratio
# ----------------------------
def sky_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([90, 20, 50])
    upper = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    return np.sum(mask > 0) / mask.size


# ----------------------------
# Feature 2: Vegetation Ratio
# ----------------------------
def vegetation_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    return np.sum(mask > 0) / mask.size


# ----------------------------
# Feature 3: Edge Density
# ----------------------------
def edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size


# ----------------------------
# Feature 4: Road Brightness
# ----------------------------
def road_brightness(img):
    h, w, _ = img.shape
    bottom = img[int(h * 0.6):, :]
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) / 255


# ----------------------------
# Feature 5: Yellow Line Ratio (NEW)
# ----------------------------
def yellow_line_ratio(img):
    """
    Detects yellow road markings (common in US/Canada).
    """
    h, w, _ = img.shape
    bottom = img[int(h * 0.6):, :]

    hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)

    lower = np.array([15, 80, 80])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    return np.sum(mask > 0) / mask.size


# ----------------------------
# Feature 6: Red Soil Ratio (NEW)
# ----------------------------
def red_soil_ratio(img):
    """
    Detects reddish ground tones (Brazil/Australia/etc).
    """
    h, w, _ = img.shape
    bottom = img[int(h * 0.6):, :]

    hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 50, 50])
    upper = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    return np.sum(mask > 0) / mask.size
