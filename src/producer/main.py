import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from kafka import KafkaProducer
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/raw/ieee-fraud-detection/train_transaction.csv"
DRIFT_FLAG = BASE_DIR / "config/drift_flag.json"

def create_producer():
    return KafkaProducer(
        bootstrap_servers="127.0.0.1:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all"
    )

def load_data():
    df = pd.read_csv(DATA_PATH, nrows=5000).fillna(0)
    return df

def is_drift_active():
    """Lit le fichier de contrôle pour savoir si drift actif"""
    try:
        if DRIFT_FLAG.exists():
            with open(DRIFT_FLAG) as f:
                return json.load(f).get("drift", False)
    except:
        pass
    return False

def apply_drift(row):
    if "TransactionAmt" in row:
        row["TransactionAmt"] *= 3
    return row

def main():
    logger.info("Producer demarre")
    df = load_data()
    producer = create_producer()
    i = 0

    while True:
        row = df.iloc[i % len(df)].to_dict()
        drift = is_drift_active()

        if drift:
            row = apply_drift(row)

        row["event_time"] = datetime.now().isoformat()
        row["drift_status"] = "DRIFT" if drift else "NORMAL"

        producer.send(config.kafka.topic_transactions, value=row)

        if i % 100 == 0:
            logger.info(f"[{'DRIFT' if drift else 'NORMAL'}] sent {i}")

        i += 1
        time.sleep(1 / config.producer.transactions_per_second)

if __name__ == "__main__":
    main()
