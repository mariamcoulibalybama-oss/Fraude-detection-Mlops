import joblib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    average_precision_score, roc_auc_score
)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/raw/ieee-fraud-detection/train_transaction.csv"

print("📦 Chargement du pipeline...")
pipeline = joblib.load(BASE_DIR / "models/pipeline_v1.pkl")
model = pipeline["model"]
feature_names = pipeline["feature_names"]

print("📦 Chargement des données...")
df = pd.read_csv(DATA_PATH)
df = df.select_dtypes(include=["number"])
df = df.loc[:, df.isnull().mean() < 0.5]
df = df.fillna(0)

X = df.drop(columns=["isFraud"])
y = df["isFraud"]
X = X.reindex(columns=feature_names, fill_value=0)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🔍 Calcul des métriques...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

recall    = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auprc     = average_precision_score(y_test, y_proba)
auroc     = roc_auc_score(y_test, y_proba)

print(f"\n📊 Métriques finales :")
print(f"   Recall    : {recall:.3f}  (cible ≥ 0.75) {'✅' if recall >= 0.75 else '❌'}")
print(f"   Precision : {precision:.3f}  (cible ≥ 0.60) {'✅' if precision >= 0.60 else '❌'}")
print(f"   F1        : {f1:.3f}")
print(f"   AUPRC     : {auprc:.3f}  (cible ≥ 0.75) {'✅' if auprc >= 0.75 else '❌'}")
print(f"   AUROC     : {auroc:.3f}")
