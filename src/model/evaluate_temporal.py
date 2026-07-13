"""
Evaluation comparative : split aleatoire vs split temporel.
En contexte de drift, le split temporel (train = passe, test = futur)
est la seule evaluation realiste. Calcule aussi l'AUPRC, metrique
de pilotage recommandee en contexte fortement desequilibre.
"""
import gc
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score
)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/raw/ieee-fraud-detection/train_transaction.csv"


def load_and_engineer():
    print("Chargement des donnees...")
    df = pd.read_csv(DATA_PATH)
    df = df.select_dtypes(include=["number"])
    df = df.loc[:, df.isnull().mean() < 0.5]
    df = df.fillna(0)

    df["heure_transaction"] = (df["TransactionDT"] // 3600) % 24

    print("Calcul des features comportementales...")
    df = df.sort_values(["card1", "TransactionDT"]).reset_index(drop=True)
    df["event_dt"] = pd.to_datetime(df["TransactionDT"], unit="s")

    grouped = df.set_index("event_dt").groupby("card1")
    df["nb_tx_1h"] = grouped["TransactionAmt"].rolling("1h").count().reset_index(level=0, drop=True).values
    df["montant_cumule_1h"] = grouped["TransactionAmt"].rolling("1h").sum().reset_index(level=0, drop=True).values
    df["temps_depuis_derniere_tx"] = df.groupby("card1")["TransactionDT"].diff().fillna(-1)

    df["montant_moyen_hist"] = df.groupby("card1")["TransactionAmt"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["montant_moyen_hist"] = df["montant_moyen_hist"].fillna(df["TransactionAmt"])
    df["ecart_montant_vs_moyenne_carte"] = (
        (df["TransactionAmt"] - df["montant_moyen_hist"]) / df["montant_moyen_hist"].replace(0, np.nan)
    ).fillna(0)
    df = df.drop(columns=["montant_moyen_hist", "event_dt"])
    return df


def train_and_eval(X_train, X_test, y_train, y_test, label):
    scale = round((y_train == 0).sum() / (y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale, eval_metric="aucpr",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {label} =====")
    print(f"  Train : {len(X_train)} tx | Test : {len(X_test)} tx")
    print(f"  Taux fraude train : {y_train.mean()*100:.2f}% | test : {y_test.mean()*100:.2f}%")
    print(f"  Recall    : {recall_score(y_test, y_pred):.3f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.3f}")
    print(f"  F1        : {f1_score(y_test, y_pred):.3f}")
    print(f"  AUROC     : {roc_auc_score(y_test, y_proba):.3f}")
    print(f"  AUPRC     : {average_precision_score(y_test, y_proba):.3f}")

    del model
    gc.collect()


def main():
    df = load_and_engineer()

    # ── 1. Split ALEATOIRE (methode initiale, pour comparaison) ──
    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    y = df["isFraud"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_and_eval(X_tr, X_te, y_tr, y_te, "SPLIT ALEATOIRE (stratifie)")

    del X_tr, X_te, y_tr, y_te, X, y
    gc.collect()

    # ── 2. Split TEMPOREL : train = 80% les plus anciens, test = 20% les plus recents ──
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    cutoff = int(len(df) * 0.8)
    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    y = df["isFraud"]
    del df
    gc.collect()

    X_tr_t, X_te_t = X.iloc[:cutoff], X.iloc[cutoff:]
    y_tr_t, y_te_t = y.iloc[:cutoff], y.iloc[cutoff:]
    del X, y
    gc.collect()

    train_and_eval(X_tr_t, X_te_t, y_tr_t, y_te_t, "SPLIT TEMPOREL (train=passe, test=futur)")


if __name__ == "__main__":
    main()
