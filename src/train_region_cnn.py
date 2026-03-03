import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

df = pd.read_csv("data/region_embeddings.csv")

X = df.drop(columns=["region"])
y = df["region"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = LogisticRegression(
    max_iter=4000,
    n_jobs=-1
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("\nCNN + TEXT Region Accuracy:", acc)
print("\nBreakdown:\n")
print(classification_report(y_test, preds))

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/region_cnn_text_model.pkl")
print("Saved model to models/region_cnn_text_model.pkl")