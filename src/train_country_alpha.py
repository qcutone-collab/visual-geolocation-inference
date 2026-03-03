import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

df = pd.read_csv("data/country_features.csv")

print("Dataset size:", len(df))

X = df[["sky", "vegetation", "edges", "road", "yellow", "redsoil"]]
y = df["country"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=400,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("\nAccuracy with extra features:", acc)

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/country_model.pkl")

print("Saved model to models/country_model.pkl")
