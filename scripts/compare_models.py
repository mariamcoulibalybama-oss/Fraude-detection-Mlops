import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data/raw/ieee-fraud-detection/train_transaction.csv"

# ── Chargement et preprocessing ───────────────────────────
# Exactement le meme preprocessing que train_model.py
# pour que la comparaison soit equitable entre les 3 modeles
print("📦 Chargement du dataset...")
df = pd.read_csv(DATA_PATH)
df = df.select_dtypes(include=["number"])
df = df.loc[:, df.isnull().mean() < 0.5]
df = df.fillna(0)

# Feature engineering : meme transformation que le pipeline final
df["heure_transaction"] = (df["TransactionDT"] // 3600) % 24

# Retirer les colonnes non pertinentes (meme logique que train_model.py)
X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
y = df["isFraud"]

print(f"   Shape : {X.shape}")
print(f"   Fraudes : {y.sum()} ({y.mean()*100:.2f}%)")

# ── Split train/test strategifie ──────────────────────────
# random_state=42 pour reproductibilite
# stratify=y pour conserver le ratio fraude/normal dans les deux sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale_pos_weight pour gerer le desequilibre (commune a XGBoost)
scale = round((y_train == 0).sum() / (y_train == 1).sum())
print(f"   scale_pos_weight : {scale}")

# ── Definition des 3 modeles ──────────────────────────────
# Meme logique de gestion du desequilibre pour chaque modele :
# class_weight='balanced' pour LR et RF (equivalent au scale_pos_weight de XGBoost)
modeles = {
    "Regression Logistique": LogisticRegression(
        class_weight="balanced",  # gere le desequilibre comme scale_pos_weight
        max_iter=1000,            # augmente les iterations pour convergence
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,         # 100 arbres (moins que XGBoost pour la vitesse)
        class_weight="balanced",  # gere le desequilibre
        random_state=42,
        n_jobs=-1                 # utilise tous les coeurs disponibles
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale,   # equivalent a class_weight='balanced'
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )
}

# ── Entrainement et evaluation de chaque modele ───────────
print("\n🚀 Entrainement et evaluation des 3 modeles...\n")

resultats = []

for nom, modele in modeles.items():
    print(f"   Entrainement : {nom}...")
    modele.fit(X_train, y_train)

    # Predictions binaires (0 ou 1)
    y_pred = modele.predict(X_test)

    # Probabilites pour AUPRC et AUROC
    y_proba = modele.predict_proba(X_test)[:, 1]

    # Calcul des metriques
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auprc     = average_precision_score(y_test, y_proba)
    auroc     = roc_auc_score(y_test, y_proba)

    # Nombre de fausses alertes sur 10 000 transactions
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fa_per_10k = round(fp / len(y_test) * 10000)

    resultats.append({
        "Modele": nom,
        "Recall": round(recall, 3),
        "Precision": round(precision, 3),
        "F1": round(f1, 3),
        "AUPRC": round(auprc, 3),
        "AUROC": round(auroc, 3),
        "FA/10k": fa_per_10k
    })

    print(f"   ✅ {nom} termine")

# ── Affichage du tableau comparatif ───────────────────────
print("\n📊 TABLEAU COMPARATIF DES 3 MODELES\n")
print(f"{'Modele':<25} {'Recall':<8} {'Precision':<11} {'F1':<8} {'AUPRC':<8} {'AUROC':<8} {'FA/10k'}")
print("-" * 80)
for r in resultats:
    print(f"{r['Modele']:<25} {r['Recall']:<8} {r['Precision']:<11} {r['F1']:<8} {r['AUPRC']:<8} {r['AUROC']:<8} {r['FA/10k']}")

print("\n📌 Cible minimale : Recall >= 0.75, AUPRC >= 0.75")
print("📌 FA/10k = Fausses Alertes pour 10 000 transactions")
