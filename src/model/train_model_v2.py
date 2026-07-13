import pandas as pd
import joblib
import json
import mlflow
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score, roc_auc_score
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
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
print(f"   Shape brut : {df.shape}")

# ── Feature engineering : heure de la journee ─────────────
# Meme transformation que dans train_model.py (v1)
df["heure_transaction"] = (df["TransactionDT"] // 3600) % 24

# ── Feature engineering comportemental (Niveau 1) ─────────
# Etape 1 : trier par ordre chronologique reel
# Indispensable avant tout calcul glissant
print("📦 Tri chronologique...")
df = df.sort_values("TransactionDT").reset_index(drop=True)

# Etape 2 : creer le pseudo-identifiant client
# card1+card2+card3+card5+addr1+addr2 ensemble designent
# probablement la meme carte/le meme client
print("📦 Creation du pseudo-identifiant client...")
df["card_id"] = (
    df["card1"].astype(str) + "_" +
    df["card2"].fillna(-999).astype(str) + "_" +
    df["card3"].fillna(-999).astype(str) + "_" +
    df["card5"].fillna(-999).astype(str) + "_" +
    df["addr1"].fillna(-999).astype(str) + "_" +
    df["addr2"].fillna(-999).astype(str)
)

# Etape 3 : calculer les features comportementales causales
# shift(1) garantit qu'on n'utilise que le passe, jamais le present
print("📦 Calcul des features comportementales...")

# Montant moyen des transactions precedentes de ce client
df["montant_moyen_client"] = (
    df.groupby("card_id")["TransactionAmt"]
      .transform(lambda x: x.shift(1).expanding().mean())
)

# Ecart-type des montants precedents de ce client
df["montant_std_client"] = (
    df.groupby("card_id")["TransactionAmt"]
      .transform(lambda x: x.shift(1).expanding().std())
)

# Nombre de transactions precedentes vues pour ce client
# cumcount() est naturellement causal (compte 0,1,2,3...)
df["nb_transactions_client"] = df.groupby("card_id").cumcount()

# Etape 4 : gerer le cas du premier client jamais vu
# Pour la toute premiere transaction d'un client, pas d'historique
# On remplace NaN par les valeurs globales du dataset (valeur neutre)
moyenne_globale = df["TransactionAmt"].mean()
std_globale = df["TransactionAmt"].std()

df["montant_moyen_client"] = df["montant_moyen_client"].fillna(moyenne_globale)
df["montant_std_client"] = df["montant_std_client"].fillna(std_globale)

# Etape 5 : calculer l'ecart (z-score) par rapport a l'habitude
# +1 au denominateur pour eviter division par zero
# si un client n'a que des montants identiques (std = 0)
df["ecart_montant_client"] = (
    (df["TransactionAmt"] - df["montant_moyen_client"]) /
    (df["montant_std_client"] + 1)
)

print(f"   Nouvelles features : montant_moyen_client, montant_std_client,")
print(f"                        nb_transactions_client, ecart_montant_client")

# ── Preparation X et y ────────────────────────────────────
# On retire card_id (identifiant textuel, pas une feature numerique)
# TransactionID, TransactionDT retires comme dans v1
X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT", "card_id"])
y = df["isFraud"]

print(f"   Shape final : {X.shape}")

feature_names = list(X.columns)
with open(CONFIG_DIR / "feature_names_v2.json", "w") as f:
    json.dump(feature_names, f)
print(f"   {len(feature_names)} features sauvegardees dans feature_names_v2.json")

# ── Split train/test ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale = round((y_train == 0).sum() / (y_train == 1).sum())

# ── Entrainement XGBoost v2 ────────────────────────────────
print("🚀 Entrainement XGBoost v2...")
with mlflow.start_run(run_name="xgboost_v2_comportemental"):
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": scale,
    }
    model = XGBClassifier(**params, eval_metric="aucpr", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auprc     = average_precision_score(y_test, y_proba)
    auroc     = roc_auc_score(y_test, y_proba)

    mlflow.log_params(params)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auprc", auprc)
    mlflow.log_metric("auroc", auroc)
    mlflow.log_metric("n_features", len(feature_names))

    print(f"\n📊 Resultats XGBoost v2 (avec features comportementales) :")
    print(f"   Recall    : {recall:.3f}  (cible >= 0.75)")
    print(f"   Precision : {precision:.3f}  (cible >= 0.60)")
    print(f"   F1        : {f1:.3f}")
    print(f"   AUPRC     : {auprc:.3f}  (cible >= 0.75)")
    print(f"   AUROC     : {auroc:.3f}")

    print(f"\n📊 Comparaison v1 vs v2 :")
    print(f"   {'Metrique':<12} {'v1 (sans profil)':<20} {'v2 (avec profil)'}")
    print(f"   {'Recall':<12} {'0.811':<20} {recall:.3f}")
    print(f"   {'Precision':<12} {'0.237':<20} {precision:.3f}")
    print(f"   {'F1':<12} {'0.367':<20} {f1:.3f}")
    print(f"   {'AUPRC':<12} {'0.633':<20} {auprc:.3f}")
    print(f"   {'AUROC':<12} {'0.932':<20} {auroc:.3f}")

# ── Sauvegarde pipeline v2 ─────────────────────────────────
# On sauvegarde sous pipeline_v2.pkl pour ne pas ecraser v1
# qui reste la reference de production actuelle
joblib.dump({
    "model": model,
    "feature_names": feature_names
}, MODEL_DIR / "pipeline_v2.pkl")

print("\n✅ Pipeline v2 sauvegarde : models/pipeline_v2.pkl")
print("✅ Features sauvegardees : config/feature_names_v2.json")
