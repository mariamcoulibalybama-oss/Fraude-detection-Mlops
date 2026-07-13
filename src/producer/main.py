import json
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from kafka import KafkaProducer
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/raw/ieee-fraud-detection/train_transaction.csv"

COLS_ESSENTIELLES = [
    "TransactionID", "TransactionDT", "TransactionAmt", "isFraud",
    "card1", "card2", "card3", "card5", "card6",
    "addr1", "addr2",
    "C1", "C2", "C5", "C8", "C11", "C13", "C14",
    "D1", "D4", "D10",
    "V70", "V82", "V91",
    "dist1", "DeviceType"
]

def create_producer():
    return KafkaProducer(
        bootstrap_servers="127.0.0.1:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all"
    )

def load_data():
    """
    Charge un échantillon représentatif du dataset IEEE-CIS.
    On prend ~1000 transactions par mois pour couvrir les 6 mois
    tout en restant léger en mémoire et rapide pour la démo.
    Le drift naturel est préservé car on échantillonne de façon
    stratifiée — proportionnellement aux fraudes réelles de chaque mois.
    """
    logger.info("Chargement du dataset...")

    df_cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
    cols_a_charger = [c for c in COLS_ESSENTIELLES if c in df_cols]

    df = pd.read_csv(DATA_PATH, usecols=cols_a_charger).fillna(0)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Calculer le mois relatif (0 à 5)
    dt_min = df["TransactionDT"].min()
    df["mois_relatif"] = ((df["TransactionDT"] - dt_min) // (30*24*3600)).astype(int)

    # Échantillonner 1000 transactions par mois de façon stratifiée
    # (proportionnellement au ratio fraude/normal de chaque mois)
    echantillons = []
    for mois in sorted(df["mois_relatif"].unique()):
        df_mois = df[df["mois_relatif"] == mois]
        n = min(5000, len(df_mois))
        # Stratifié sur isFraud pour préserver le taux de fraude naturel
        fraudes = df_mois[df_mois["isFraud"] == 1]
        normaux = df_mois[df_mois["isFraud"] == 0]
        n_fraudes = min(len(fraudes), int(n * len(fraudes) / len(df_mois)))
        n_normaux = n - n_fraudes
        sample = pd.concat([
            fraudes.sample(n=n_fraudes, random_state=42),
            normaux.sample(n=n_normaux, random_state=42)
        ]).sort_values("TransactionDT")
        echantillons.append(sample)
        logger.info(f"Mois {mois} : {len(sample)} transactions "
                   f"({n_fraudes} fraudes, taux {n_fraudes/len(sample)*100:.1f}%)")

    df_final = pd.concat(echantillons).reset_index(drop=True)
    logger.info(f"Total : {len(df_final)} transactions représentatives")
    return df_final, dt_min

def calculate_simulated_dates(df, dt_min):
    """Dates calendaires simulées à partir du 1er janvier 2025"""
    date_reference = datetime(2026, 1, 1)
    df["date_simulee"] = df["TransactionDT"].apply(
        lambda dt: date_reference + timedelta(seconds=int(dt - dt_min))
    )
    df["mois_simule"] = df["date_simulee"].dt.strftime("%Y-%m")
    logger.info(f"Période : {df['date_simulee'].min().strftime('%Y-%m-%d')} "
                f"→ {df['date_simulee'].max().strftime('%Y-%m-%d')}")
    return df

def get_drift_phase(mois_simule):
    """
    Drift basé sur le taux de fraude NATUREL mesuré dans IEEE-CIS :
    Jan 2025 (mois 0) : 2.53% → NORMAL
    Fév-Mar 2025      : ~4.00% → DRIFT_NATUREL (+58% vs janvier)
    Avr-Juin 2025     : ~3.5-4.0% → DRIFT_STABILISE
    """
    if mois_simule <= "2026-01":
        return "NORMAL", "NORMAL"
    elif mois_simule <= "2026-03":
        return "DRIFT", "DRIFT_NATUREL"
    else:
        return "DRIFT", "DRIFT_STABILISE"

def main():
    logger.info("Producer démarré — échantillon représentatif, drift naturel")

    df, dt_min = load_data()
    df = calculate_simulated_dates(df, dt_min)
    producer = create_producer()

    total = len(df)
    logger.info(f"Envoi de {total} transactions (~{total/config.producer.transactions_per_second/60:.1f} minutes)")

    for i, row_data in df.iterrows():
        row = row_data.to_dict()

        mois_simule = row.get("mois_simule", "2026-01")
        drift_status, drift_phase = get_drift_phase(mois_simule)

        row["event_time"] = row_data["date_simulee"].isoformat()
        row["drift_status"] = drift_status
        row["drift_phase"] = drift_phase
        row["drift_level"] = 0 if drift_status == "NORMAL" else 1
        row["mois_simule"] = mois_simule
        row.pop("date_simulee", None)
        row.pop("mois_relatif", None)

        producer.send(config.kafka.topic_transactions, value=row)

        if i % 500 == 0:
            logger.info(f"[{drift_phase}] {mois_simule} — {i}/{total} "
                       f"({i/total*100:.1f}%)")

        time.sleep(1 / config.producer.transactions_per_second)

    logger.info("Toutes les transactions ont été envoyées.")

if __name__ == "__main__":
    main()
