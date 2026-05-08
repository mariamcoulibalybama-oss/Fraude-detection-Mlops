import pandas as pd
import joblib
import json
import time
from scipy import stats
from pymongo import MongoClient
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data/raw/ieee-fraud-detection/train_transaction.csv"
MODEL_PATH = BASE_DIR / "models/pipeline_v1.pkl"
CONFIG_DIR = BASE_DIR / "config"

DRIFT_THRESHOLD  = 0.05
CHECK_INTERVAL   = 60
COOLDOWN_AFTER_RETRAIN = 600  # 10 minutes après retraining
MIN_SAMPLES      = 50

def check_drift():
    client = MongoClient("mongodb://admin:changeme@localhost:27017/")
    collection = client["fraude_db"]["predictions"]
    docs = list(collection.find({}, {"_id": 0, "score": 1, "drift_status": 1}))

    if len(docs) < MIN_SAMPLES:
        print(f"   Pas assez de données ({len(docs)} < {MIN_SAMPLES})")
        return False, None

    df = pd.DataFrame(docs)
    df_normal = df[df["drift_status"] == "NORMAL"]["score"]
    df_drift  = df[df["drift_status"] == "DRIFT"]["score"]

    if len(df_normal) < 5 or len(df_drift) < 5:
        print("   Pas assez de données dans chaque groupe")
        return False, None

    _, p_value = stats.ks_2samp(df_normal, df_drift)
    print(f"   p-value : {p_value:.4f}")
    return p_value < DRIFT_THRESHOLD, p_value

def retrain():
    print("🚀 Chargement des données...")
    df = pd.read_csv(DATA_PATH)
    df = df.select_dtypes(include=["number"])
    df = df.loc[:, df.isnull().mean() < 0.5]
    df = df.fillna(0)

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    feature_names = list(X.columns)
    with open(CONFIG_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale = round((y_train == 0).sum() / (y_train == 1).sum())

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    print(f"   Recall    : {recall:.3f}")
    print(f"   Precision : {precision:.3f}")
    print(f"   F1        : {f1:.3f}")

    joblib.dump({
        "model": model,
        "feature_names": feature_names
    }, MODEL_PATH)

    print(f"✅ Nouveau modèle sauvegardé")
    return recall, precision, f1

def main():
    print("🔄 Auto-retrain démarré")
    print(f"   Seuil drift        : p-value < {DRIFT_THRESHOLD}")
    print(f"   Vérification       : toutes les {CHECK_INTERVAL}s")
    print(f"   Cooldown retrain   : {COOLDOWN_AFTER_RETRAIN//60} minutes")

    retrain_count = 0

    while True:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now}] Vérification du drift...")

        drift_detected, p_value = check_drift()

        if drift_detected:
            retrain_count += 1
            print(f"⚠️  DRIFT DÉTECTÉ — Retraining #{retrain_count} lancé !")
            retrain()
            print(f"   Cooldown de {COOLDOWN_AFTER_RETRAIN//60} min avant prochaine vérification...")
            time.sleep(COOLDOWN_AFTER_RETRAIN)
        else:
            print("   ✅ Pas de drift — prochaine vérification dans 60s")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
