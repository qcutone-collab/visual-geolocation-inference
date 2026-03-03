import cv2
import sys
from pathlib import Path

def load_image(path):
    img = cv2.imread(str(path))

    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    img = cv2.resize(img, (224, 224))
    return img

if __name__ == "__main__":
    path = Path(sys.argv[1])

    img = load_image(path)

    print("Loaded image:", path)
    print("Image shape:", img.shape)

    cv2.imshow("Street View Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
