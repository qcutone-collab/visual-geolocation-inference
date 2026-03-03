import cv2
import csv
from pathlib import Path
from features import sky_ratio, vegetation_ratio, edge_density

raw_dir = Path("data/raw")
output_file = Path("data/features.csv")

images = list(raw_dir.glob("*.png"))

print("Found", len(images), "images")

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "sky", "vegetation", "edges"])

    for img_path in images:
        img = cv2.imread(str(img_path))

        sky = sky_ratio(img)
        veg = vegetation_ratio(img)
        edges = edge_density(img)

        writer.writerow([img_path.name, sky, veg, edges])

print("Saved dataset to", output_file)
