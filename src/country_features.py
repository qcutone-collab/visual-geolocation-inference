import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

try:
    from src.features import (
        sky_ratio,
        vegetation_ratio,
        edge_density,
        road_brightness,
        yellow_line_ratio,
        red_soil_ratio,
    )
except ModuleNotFoundError:
    from features import (  # type: ignore
        sky_ratio,
        vegetation_ratio,
        edge_density,
        road_brightness,
        yellow_line_ratio,
        red_soil_ratio,
    )

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_BACKBONE = None
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def _load_backbone():
    global _BACKBONE
    if _BACKBONE is None:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        model.eval()
        model.to(_DEVICE)
        _BACKBONE = model
    return _BACKBONE


def extract_handcrafted_features(img_bgr):
    return np.array(
        [
            sky_ratio(img_bgr),
            vegetation_ratio(img_bgr),
            edge_density(img_bgr),
            road_brightness(img_bgr),
            yellow_line_ratio(img_bgr),
            red_soil_ratio(img_bgr),
        ],
        dtype=np.float32,
    )


def extract_country_features(img_bgr):
    model = _load_backbone()
    handcrafted = extract_handcrafted_features(img_bgr)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    x = _TRANSFORM(pil_img).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        deep = model(x).squeeze(0).cpu().numpy().astype(np.float32)

    return np.concatenate([deep, handcrafted], axis=0)
