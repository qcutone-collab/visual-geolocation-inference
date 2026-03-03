import cv2
from features import sky_ratio, vegetation_ratio, edge_density

img = cv2.imread("data/raw/test.png")

print("Sky ratio:", sky_ratio(img))
print("Vegetation ratio:", vegetation_ratio(img))
print("Edge density:", edge_density(img))
