import csv
import random
from pathlib import Path
import cv2

from features import (
    sky_ratio,
    vegetation_ratio,
    edge_density,
    road_brightness,
    yellow_line_ratio,
    red_soil_ratio
)

# ============================
# SETTINGS
# ============================

ROOT_DIR = Path("data/geoworld")
OUTPUT_FILE = Path("data/country_features.csv")

MAX_IMAGES_PER_COUNTRY = 100
MIN_IMAGES_REQUIRED = 50


# ============================
# MAIN BUILDER
# ============================

def build_dataset():
    print("\nBuilding dataset from ALL country folders...")
    print(f"Max images per country: {MAX_IMAGES_PER_COUNTRY}")
    print(f"Skipping countries with fewer than {MIN_IMAGES_REQUIRED} images\n")

    rows = []
    included = 0
    skipped = 0
    total_images = 0

    for country_folder in sorted(ROOT_DIR.iterdir()):

        if not country_folder.is_dir():
            continue

        country = country_folder.name

        # Collect images
        images = list(country_folder.glob("*.jpg")) + list(country_folder.glob("*.png"))

        if len(images) < MIN_IMAGES_REQUIRED:
            skipped += 1
            continue

        # Sample up to max
        selected = random.sample(images, min(len(images), MAX_IMAGES_PER_COUNTRY))

        print(f"Including {country}: using {len(selected)} images")

        included += 1
        total_images += len(selected)

        for img_path in selected:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            rows.append([
                country,
                sky_ratio(img),
                vegetation_ratio(img),
                edge_density(img),
                road_brightness(img),
                yellow_line_ratio(img),
                red_soil_ratio(img)
            ])

    # Write CSV
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "country",
            "sky",
            "vegetation",
            "edges",
            "road",
            "yellow",
            "redsoil"
        ])

        writer.writerows(rows)

    print("\nDONE.")
    print(f"Countries included: {included}")
    print(f"Countries skipped: {skipped}")
    print(f"Total images used: {total_images}")
    print(f"Saved dataset to: {OUTPUT_FILE}\n")


# ============================
# RUN SCRIPT
# ============================

if __name__ == "__main__":
    build_dataset()
