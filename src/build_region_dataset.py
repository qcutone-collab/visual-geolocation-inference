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

ROOT_DIR = Path("data/geoworld")
OUTPUT_FILE = Path("data/region_features.csv")

MAX_IMAGES_PER_COUNTRY = 100
MIN_IMAGES_REQUIRED = 50

# Region mapping based on rough geography.
# This is a pragmatic mapping for an ML project, not a political statement.
EUROPE = {
    "Aland", "Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bulgaria", "Croatia",
    "Czechia", "Denmark", "Estonia", "Faroe Islands", "Finland", "France", "Germany",
    "Gibraltar", "Greece", "Greenland", "Hungary", "Iceland", "Ireland", "Isle of Man",
    "Italy", "Jersey", "Latvia", "Lithuania", "Luxembourg", "Malta", "Moldova",
    "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland",
    "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia",
    "Spain", "Svalbard and Jan Mayen", "Sweden", "Switzerland", "Ukraine",
    "United Kingdom"
}

ASIA = {
    "Armenia", "Azerbaijan", "Bangladesh", "Bhutan", "Cambodia", "China", "Hong Kong",
    "India", "Indonesia", "Japan", "Kazakhstan", "Kyrgyzstan", "Laos", "Lebanon",
    "Macao", "Malaysia", "Mongolia", "Myanmar", "Nepal", "North Korea", "Pakistan",
    "Palestine", "Philippines", "Singapore", "South Korea", "Sri Lanka", "Taiwan",
    "Thailand", "Turkey", "United Arab Emirates", "Vietnam", "Iraq", "Israel", "Jordan",
    "Qatar"
}

AFRICA = {
    "Egypt", "Eswatini", "Ghana", "Kenya", "Lesotho", "Madagascar", "Mozambique",
    "Nigeria", "Senegal", "South Africa", "South Sudan", "Tanzania", "Tunisia",
    "Uganda", "Botswana"
}

NORTH_AMERICA = {
    "Canada", "United States", "Mexico", "Guatemala", "Costa Rica", "Dominican Republic",
    "Puerto Rico", "Bermuda", "Curacao", "Martinique", "Reunion", "Guam",
    "Northern Mariana Islands", "US Virgin Islands", "American Samoa"
}

SOUTH_AMERICA = {
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Paraguay",
    "Peru", "Uruguay", "Venezuela"
}

OCEANIA = {
    "Australia", "New Zealand", "Pitcairn Islands"
}

POLAR = {
    "Antarctica", "South Georgia and South Sandwich Islands"
}

def country_to_region(country: str) -> str:
    if country in EUROPE:
        return "Europe"
    if country in ASIA:
        return "Asia"
    if country in AFRICA:
        return "Africa"
    if country in NORTH_AMERICA:
        return "NorthAmerica"
    if country in SOUTH_AMERICA:
        return "SouthAmerica"
    if country in OCEANIA:
        return "Oceania"
    if country in POLAR:
        return "Polar"
    return "Other"


def build_dataset():
    print("\nBuilding REGION dataset from data/geoworld...")
    print("Max images per country:", MAX_IMAGES_PER_COUNTRY)
    print("Skipping countries with fewer than", MIN_IMAGES_REQUIRED, "images\n")

    rows = []
    included_countries = 0
    skipped_countries = 0
    total_images = 0

    for country_folder in sorted(ROOT_DIR.iterdir()):
        if not country_folder.is_dir():
            continue

        country = country_folder.name
        region = country_to_region(country)

        image_paths = list(country_folder.glob("*.jpg")) + list(country_folder.glob("*.png"))

        if len(image_paths) < MIN_IMAGES_REQUIRED:
            skipped_countries += 1
            continue

        selected = random.sample(image_paths, min(len(image_paths), MAX_IMAGES_PER_COUNTRY))

        print(f"Including {country} → {region} with {len(selected)} images")

        included_countries += 1
        total_images += len(selected)

        for img_path in selected:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            rows.append([
                region,
                sky_ratio(img),
                vegetation_ratio(img),
                edge_density(img),
                road_brightness(img),
                yellow_line_ratio(img),
                red_soil_ratio(img)
            ])

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "sky", "vegetation", "edges", "road", "yellow", "redsoil"])
        writer.writerows(rows)

    print("\nDONE.")
    print("Countries included:", included_countries)
    print("Countries skipped:", skipped_countries)
    print("Total images used:", total_images)
    print("Saved dataset to:", OUTPUT_FILE, "\n")


if __name__ == "__main__":
    build_dataset()
