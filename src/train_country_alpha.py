import random
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

try:
    from src.country_features import extract_country_features
except ModuleNotFoundError:
    from country_features import extract_country_features  # type: ignore

ROOT_DIR = Path("data/geoworld")
MIN_IMAGES_REQUIRED = 60
MODEL_PATH = Path("models/country_model.pkl")
FEATURE_CACHE_PATH = Path("data/country_cnn_features.npz")
METRICS_PATH = Path("models/country_metrics.txt")


def build_country_training_matrix():
    X_rows = []
    y_rows = []

    countries = 0
    images_used = 0

    for country_folder in sorted(ROOT_DIR.iterdir()):
        if not country_folder.is_dir():
            continue

        images = list(country_folder.glob("*.jpg")) + list(country_folder.glob("*.png"))
        if len(images) < MIN_IMAGES_REQUIRED:
            continue

        selected = images
        countries += 1

        for image_path in selected:
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            try:
                feats = extract_country_features(img)
                X_rows.append(feats)
                y_rows.append(country_folder.name)
                images_used += 1
            except Exception:
                continue

    print("Countries included:", countries)
    print("Images used:", images_used)
    return np.array(X_rows, dtype=np.float32), np.array(y_rows)


def train_country_model():
    if FEATURE_CACHE_PATH.exists():
        cache = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
        X = cache["X"]
        y = cache["y"]
        print("Loaded cached features from", FEATURE_CACHE_PATH)
    else:
        X, y = build_country_training_matrix()
        np.savez_compressed(FEATURE_CACHE_PATH, X=X, y=y)
        print("Saved cached features to", FEATURE_CACHE_PATH)

    if len(X) == 0:
        raise RuntimeError("No training samples found. Check data/geoworld.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=900,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=700,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }

    best_name = None
    best_model = None
    best_top1 = -1.0
    best_top3 = -1.0

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)

        top1 = accuracy_score(y_test, preds)
        top3 = top_k_accuracy_score(y_test, probs, k=3, labels=model.classes_)
        print(f"\n{name} -> top1: {top1:.4f}, top3: {top3:.4f}")

        if top1 > best_top1:
            best_top1 = top1
            best_top3 = top3
            best_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("No model was trained.")

    print(f"\nBest model: {best_name}")
    print(f"Country Top-1 Accuracy: {best_top1:.4f}")
    print(f"Country Top-3 Accuracy: {best_top3:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print("Saved model to", MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"best_model={best_name}\n")
        f.write(f"top1={best_top1:.6f}\n")
        f.write(f"top3={best_top3:.6f}\n")
    print("Saved metrics to", METRICS_PATH)


if __name__ == "__main__":
    train_country_model()
