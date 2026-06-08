"""
Script pour créer un pipeline de test léger pour CI/CD
"""
import joblib
import json
import numpy as np
from xgboost import XGBClassifier
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Créer un modèle minimaliste pour les tests
feature_names = json.load(open(BASE_DIR / "config/feature_names.json"))

X = np.zeros((100, len(feature_names)))
y = np.array([0] * 97 + [1] * 3)

model = XGBClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

joblib.dump({
    "model": model,
    "feature_names": feature_names
}, BASE_DIR / "models/pipeline_v1.pkl")

print("✅ Pipeline de test créé")
