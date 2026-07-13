import pytest
import joblib
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

def test_pipeline_existe():
    """Vérifie que le pipeline est sauvegardé"""
    pipeline_path = BASE_DIR / "models/pipeline_v1.pkl"
    assert pipeline_path.exists(), "pipeline_v1.pkl introuvable"

def test_pipeline_contient_model_et_features():
    """Vérifie que le pipeline contient le modèle et les features"""
    pipeline = joblib.load(BASE_DIR / "models/pipeline_v1.pkl")
    assert "model" in pipeline, "Clé 'model' manquante"
    assert "feature_names" in pipeline, "Clé 'feature_names' manquante"

def test_nombre_features():
    """Vérifie qu on a bien 213 features (209 statiques + heure_transaction retiree du compte + 4 comportementales)"""
    pipeline = joblib.load(BASE_DIR / "models/pipeline_v1.pkl")
    assert len(pipeline["feature_names"]) == 213, "Nombre de features incorrect"

def test_feature_names_json_existe():
    """Vérifie que feature_names.json existe"""
    json_path = BASE_DIR / "config/feature_names.json"
    assert json_path.exists(), "feature_names.json introuvable"

def test_feature_names_coherents():
    """Vérifie que les features du pipeline et du JSON sont identiques"""
    pipeline = joblib.load(BASE_DIR / "models/pipeline_v1.pkl")
    with open(BASE_DIR / "config/feature_names.json") as f:
        features_json = json.load(f)
    assert pipeline["feature_names"] == features_json, "Features incohérentes"

def test_prediction_simple():
    """Vérifie que le modèle peut faire une prédiction"""
    pipeline = joblib.load(BASE_DIR / "models/pipeline_v1.pkl")
    model = pipeline["model"]
    feature_names = pipeline["feature_names"]

    # Créer une transaction vide
    df = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)
    score = model.predict_proba(df)[0][1]

    assert 0 <= score <= 1, "Score hors limite [0, 1]"

def test_score_transaction_normale():
    """Vérifie que le score d'une transaction normale est faible"""
    pipeline = joblib.load(BASE_DIR / "models/pipeline_v1.pkl")
    model = pipeline["model"]
    feature_names = pipeline["feature_names"]

    df = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)
    score = model.predict_proba(df)[0][1]

    assert score < 0.8, f"Score trop élevé pour une transaction vide : {score}"
