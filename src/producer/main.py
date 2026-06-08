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

def create_producer():
    return KafkaProducer(
        bootstrap_servers="127.0.0.1:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all"
    )

def load_data():
    df = pd.read_csv(DATA_PATH, nrows=3000).fillna(0)
    return df

# Timeline 2025-2026 — cycle complet sur 2 ans
TIMELINE = [
    # (debut, fin, phase, niveau, date_debut, date_fin)
    # ── 2025 ──────────────────────────────────────────────
    (0,    500,  "NORMAL",         0, "2025-01-01", "2025-02-28"),
    (500,  800,  "DRIFT_AMT",      1, "2025-03-01", "2025-03-31"),
    (800,  1100, "DRIFT_CARD",     2, "2025-04-01", "2025-04-30"),
    (1100, 1400, "DRIFT_BEHAVIOR", 3, "2025-05-01", "2025-05-31"),
    (1400, 1600, "DRIFT_FORT",     4, "2025-06-01", "2025-06-30"),
    (1600, 2000, "NOUVEAU_NORMAL", 0, "2025-07-01", "2025-12-31"),
    # ── 2026 ──────────────────────────────────────────────
    (2000, 2300, "NORMAL",         0, "2026-01-01", "2026-02-28"),
    (2300, 2500, "DRIFT_AMT",      1, "2026-03-01", "2026-03-31"),
    (2500, 2650, "DRIFT_CARD",     2, "2026-04-01", "2026-04-30"),
    (2650, 2800, "DRIFT_BEHAVIOR", 3, "2026-05-01", "2026-05-31"),
    (2800, 2900, "DRIFT_FORT",     4, "2026-06-01", "2026-06-30"),
    (2900, 3000, "NOUVEAU_NORMAL", 0, "2026-07-01", "2026-12-31"),
]

def get_phase_and_date(i):
    idx = i % 3000

    for debut, fin, phase, niveau, date_debut, date_fin in TIMELINE:
        if debut <= idx < fin:
            d_debut = datetime.strptime(date_debut, "%Y-%m-%d")
            d_fin = datetime.strptime(date_fin, "%Y-%m-%d")
            total_jours = (d_fin - d_debut).days
            progression = (idx - debut) / (fin - debut)
            date_simulee = d_debut + timedelta(days=int(progression * total_jours))
            return phase, niveau, date_simulee

    return "NORMAL", 0, datetime(2025, 1, 1)

def apply_drift(row, phase):
    if phase in ["NORMAL", "NOUVEAU_NORMAL"]:
        return row

    if phase == "DRIFT_AMT":
        if "TransactionAmt" in row:
            row["TransactionAmt"] *= 1.2

    elif phase == "DRIFT_CARD":
        if "TransactionAmt" in row:
            row["TransactionAmt"] *= 1.2
        if "card1" in row and row["card1"] != 0:
            row["card1"] = row["card1"] * 1.5
        if "card2" in row and row["card2"] != 0:
            row["card2"] = row["card2"] * 1.3

    elif phase == "DRIFT_BEHAVIOR":
        if "TransactionAmt" in row:
            row["TransactionAmt"] *= 1.5
        if "card1" in row and row["card1"] != 0:
            row["card1"] = row["card1"] * 1.5
        if "C1" in row and row["C1"] != 0:
            row["C1"] = row["C1"] * 0.3
        if "C14" in row and row["C14"] != 0:
            row["C14"] = row["C14"] * 0.3

    elif phase == "DRIFT_FORT":
        if "TransactionAmt" in row:
            row["TransactionAmt"] *= 3.0
        if "card1" in row and row["card1"] != 0:
            row["card1"] = row["card1"] * 2.0
        if "card2" in row and row["card2"] != 0:
            row["card2"] = row["card2"] * 1.8
        if "C1" in row and row["C1"] != 0:
            row["C1"] = row["C1"] * 0.1
        if "C14" in row and row["C14"] != 0:
            row["C14"] = row["C14"] * 0.1
        if "V70" in row and row["V70"] != 0:
            row["V70"] = row["V70"] * -2.0
        if "V17" in row and row["V17"] != 0:
            row["V17"] = row["V17"] * -1.5

    return row

def main():
    logger.info("Producer demarre — simulation timeline 2025-2026")
    df = load_data()
    producer = create_producer()
    i = 0

    while True:
        row = df.iloc[i % len(df)].to_dict()

        phase, level, date_simulee = get_phase_and_date(i)
        row = apply_drift(row, phase)

        row["event_time"] = date_simulee.isoformat()
        row["drift_status"] = "NORMAL" if phase in ["NORMAL", "NOUVEAU_NORMAL"] else "DRIFT"
        row["drift_phase"] = phase
        row["drift_level"] = level
        row["mois_simule"] = date_simulee.strftime("%Y-%m")

        producer.send(config.kafka.topic_transactions, value=row)

        if i % 100 == 0:
            logger.info(f"[{phase}] {date_simulee.strftime('%Y-%m')} — sent {i}")

        i += 1
        time.sleep(1 / config.producer.transactions_per_second)

if __name__ == "__main__":
    main()
