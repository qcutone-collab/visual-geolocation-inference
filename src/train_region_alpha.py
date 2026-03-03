import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

df = pd.read_csv("data/region_features.csv")

print("Dataset size:", len(df))
print("Regions:", sorted(df["region"].unique()))

X = df[["sky", "vegetation", "edges", "road", "yellow", "redsoil"]]
y = df["region"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=600,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("\nRegion Accuracy:", acc)
print("\nBreakdown:\n")
print(classification_report(y_test, preds))

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/region_model.pkl")
print("Saved model to models/region_model.pkl")
