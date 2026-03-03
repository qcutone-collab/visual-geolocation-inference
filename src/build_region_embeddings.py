import csv
import random
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models

from text_features import detect_script

ROOT_DIR = Path("data/geoworld")
OUTPUT_FILE = Path("data/region_embeddings.csv")

MAX_IMAGES_PER_COUNTRY = 100
MIN_IMAGES_REQUIRED = 50
MAX_IMAGES_PER_REGION = 1200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# REGION MAP
# ----------------------------

EUROPE = {
    "Austria","Belgium","Bulgaria","Croatia","Czechia","Denmark","Estonia",
    "Finland","France","Germany","Greece","Hungary","Iceland","Ireland",
    "Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands","Norway",
    "Poland","Portugal","Romania","Russia","Serbia","Slovakia","Slovenia",
    "Spain","Sweden","Switzerland","Ukraine","United Kingdom"
}

ASIA = {
    "Bangladesh","Cambodia","China","Hong Kong","India","Indonesia","Japan",
    "Jordan","Kyrgyzstan","Laos","Malaysia","Mongolia","Nepal","Philippines",
    "Singapore","South Korea","Sri Lanka","Taiwan","Thailand","Turkey",
    "United Arab Emirates","Vietnam","Israel"
}

AFRICA = {
    "Botswana","Egypt","Eswatini","Ghana","Kenya","Lesotho","Madagascar",
    "Mozambique","Nigeria","Senegal","South Africa","South Sudan",
    "Tanzania","Tunisia","Uganda"
}

AMERICAS = {
    "Argentina","Bolivia","Brazil","Canada","Chile","Colombia","Costa Rica",
    "Dominican Republic","Ecuador","Guatemala","Mexico","Paraguay","Peru",
    "United States","Uruguay","Venezuela"
}

def country_to_region(country):
    if country in EUROPE:
        return "Europe"
    if country in ASIA:
        return "Asia"
    if country in AFRICA:
        return "Africa"
    if country in AMERICAS:
        return "Americas"
    return None


# ----------------------------
# CNN MODEL
# ----------------------------

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model.eval()
model.to(DEVICE)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def build_embeddings():
    rows_by_region = {
        "Europe": [],
        "Asia": [],
        "Africa": [],
        "Americas": []
    }

    for country_folder in ROOT_DIR.iterdir():
        if not country_folder.is_dir():
            continue

        region = country_to_region(country_folder.name)
        if region is None:
            continue

        images = list(country_folder.glob("*.jpg")) + list(country_folder.glob("*.png"))
        if len(images) < MIN_IMAGES_REQUIRED:
            continue

        selected = random.sample(images, min(len(images), MAX_IMAGES_PER_COUNTRY))

        for img_path in selected:
            if len(rows_by_region[region]) >= MAX_IMAGES_PER_REGION:
                break

            try:
                img = Image.open(img_path).convert("RGB")
                x = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    embedding = model(x).squeeze().cpu().numpy().tolist()

                has_text, script_id = detect_script(img_path)

                rows_by_region[region].append(
                    [region] + embedding + [has_text, script_id]
                )

            except Exception:
                continue

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = (
            ["region"]
            + [f"f{i}" for i in range(512)]
            + ["has_text", "script"]
        )
        writer.writerow(header)

        for region, rows in rows_by_region.items():
            print(region, len(rows))
            writer.writerows(rows)

    print("\nSaved embeddings to:", OUTPUT_FILE)


if __name__ == "__main__":
    build_embeddings()