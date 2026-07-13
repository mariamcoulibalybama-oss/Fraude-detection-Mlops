"""
Quantification de l'impact de la feature parity partielle (remarque G1).
Mesure quelle part de l'importance totale du modele est couverte par
les features reellement alimentees en production (producer + derivees
+ feature store Redis), les autres etant remplies a 0 par le reindex.
"""
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models/pipeline_v1.pkl"

# Colonnes numeriques envoyees par le producer (hors ID/DT/label/categorielles)
COLS_PRODUCER = [
    "TransactionAmt",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "C1", "C2", "C5", "C8", "C11", "C13", "C14",
    "D1", "D4", "D10",
    "V70", "V82", "V91",
    "dist1",
]

# Features calculees cote consumer (derivees + feature store Redis)
COLS_CALCULEES = [
    "heure_transaction",
    "nb_tx_1h",
    "montant_cumule_1h",
    "temps_depuis_derniere_tx",
    "ecart_montant_vs_moyenne_carte",
]


def main():
    pipeline = joblib.load(MODEL_PATH)
    model = pipeline["model"]
    feature_names = pipeline["feature_names"]

    importances = model.feature_importances_
    total_imp = importances.sum()

    alimentees = set(COLS_PRODUCER) | set(COLS_CALCULEES)
    mask = np.array([f in alimentees for f in feature_names])

    n_alim = int(mask.sum())
    imp_alim = importances[mask].sum()

    print(f"Features du modele              : {len(feature_names)}")
    print(f"Features reellement alimentees  : {n_alim} ({n_alim/len(feature_names)*100:.1f}%)")
    print(f"Features remplies a zero        : {len(feature_names)-n_alim}")
    print()
    print(f"Part de l'importance totale couverte par les features alimentees :")
    print(f"  {imp_alim/total_imp*100:.1f}%")
    print()

    # Top 20 features par importance, avec statut
    order = np.argsort(importances)[::-1]
    print("Top 20 features par importance (modele) :")
    for i in order[:20]:
        statut = "ALIMENTEE" if mask[i] else "A ZERO"
        print(f"  {feature_names[i]:35s} {importances[i]:.4f}  [{statut}]")


if __name__ == "__main__":
    main()
