import cv2
from pathlib import Path

raw_dir = Path("data/raw")
processed_dir = Path("data/processed")

processed_dir.mkdir(exist_ok=True)

images = list(raw_dir.glob("*.png"))
print("Found", len(images), "images")

for img_path in images[:50]:
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (224, 224))

    out_path = processed_dir / img_path.name
    cv2.imwrite(str(out_path), img)

print("Saved resized images into data/processed/")
