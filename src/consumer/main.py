import json
import joblib
import pandas as pd
from kafka import KafkaConsumer
from loguru import logger
from pymongo import MongoClient
from datetime import datetime

# ── Modèle ─────────────────────────────────────────────────
pipeline = joblib.load("models/pipeline_v1.pkl")
model = pipeline["model"]
feature_names = pipeline["feature_names"]

# ── MongoDB ────────────────────────────────────────────────
mongo = MongoClient("mongodb://admin:changeme@localhost:27017/")
db = mongo["fraude_db"]
collection = db["predictions"]

# ── Kafka ──────────────────────────────────────────────────
TOPIC = "transactions"
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8"))
)

def preprocess(row):
    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_names, fill_value=0)
    return df

def score_transaction(row):
    try:
        df = preprocess(row)
        score = model.predict_proba(df)[0][1]
        return float(score)
    except Exception as e:
        logger.error(f"Model fallback: {e}")
        return 0.5

def main():
    logger.info("Consumer démarré — en attente de transactions...")
    for message in consumer:
        row = message.value
        score = score_transaction(row)
        is_fraud = score > 0.5

        result = {
            "transaction_id": int(row.get("TransactionID", 0)),
            "amount": float(row.get("TransactionAmt", 0)),
            "score": score,
            "is_fraud": is_fraud,
            "drift_status": row.get("drift_status", "NORMAL"),
            "timestamp": datetime.utcnow()
        }

        # ── Sauvegarde MongoDB ─────────────────────────────
        collection.insert_one(result)

        logger.info(f"ALERT: {result}")

if __name__ == "__main__":
    main()
