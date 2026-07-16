"""
Comparaison des strategies de gestion du desequilibre :
scale_pos_weight (retenu) vs SMOTE (prevu dans le cadrage initial).
Meme split, memes features, pour une comparaison equitable.
"""
import gc
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             roc_auc_score, average_precision_score)
from imblearn.over_sampling import SMOTE
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data/raw/ieee-fraud-detection/train_transaction.csv"

def load():
    df = pd.read_csv(DATA)
    df = df.select_dtypes(include=["number"])
    df = df.loc[:, df.isnull().mean() < 0.5]
    df = df.fillna(0)
    df["heure_transaction"] = (df["TransactionDT"] // 3600) % 24
    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    y = df["isFraud"]
    return X, y

def evaluate(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n===== {label} =====")
    print(f"  Recall    : {recall_score(y_test, y_pred):.3f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.3f}")
    print(f"  F1        : {f1_score(y_test, y_pred):.3f}")
    print(f"  AUROC     : {roc_auc_score(y_test, y_proba):.3f}")
    print(f"  AUPRC     : {average_precision_score(y_test, y_proba):.3f}")

def main():
    X, y = load()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Strategie 1 : scale_pos_weight (retenue) ---
    scale = round((y_tr == 0).sum() / (y_tr == 1).sum())
    m1 = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                       scale_pos_weight=scale, eval_metric="aucpr",
                       random_state=42, n_jobs=-1)
    m1.fit(X_tr, y_tr)
    evaluate(m1, X_te, y_te, "SCALE_POS_WEIGHT (retenu)")
    del m1; gc.collect()

    # --- Strategie 2 : SMOTE (prevu au cadrage) ---
    print("\nApplication de SMOTE sur le train...")
    sm = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
    print(f"  Train apres SMOTE : {len(X_tr_sm)} tx (fraude {y_tr_sm.mean()*100:.1f}%)")
    m2 = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                       eval_metric="aucpr", random_state=42, n_jobs=-1)
    m2.fit(X_tr_sm, y_tr_sm)
    evaluate(m2, X_te, y_te, "SMOTE")

if __name__ == "__main__":
    main()
