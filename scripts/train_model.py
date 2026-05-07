import pandas as pd
import joblib
import json
import mlflow
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data/raw/ieee-fraud-detection/train_transaction.csv"
MODEL_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("fraude-detection")

print("📦 Chargement du dataset...")
df = pd.read_csv(DATA_PATH)
df = df.select_dtypes(include=["number"])
df = df.loc[:, df.isnull().mean() < 0.5]
df = df.fillna(0)
print(f"   Shape : {df.shape}")

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

feature_names = list(X.columns)
with open(CONFIG_DIR / "feature_names.json", "w") as f:
    json.dump(feature_names, f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale = round((y_train == 0).sum() / (y_train == 1).sum())

with mlflow.start_run(run_name="xgboost_v1"):
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": scale,
    }
    model = XGBClassifier(**params, eval_metric="aucpr", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    mlflow.log_params(params)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("n_features", len(feature_names))

    print(f"\n📊 Résultats :")
    print(f"   Recall    : {recall:.3f}")
    print(f"   Precision : {precision:.3f}")
    print(f"   F1        : {f1:.3f}")
    print(f"   Run loggé dans MLflow ✅")

joblib.dump({
    "model": model,
    "feature_names": feature_names
}, MODEL_DIR / "pipeline_v1.pkl")

print("✅ Pipeline sauvegardé : models/pipeline_v1.pkl")
