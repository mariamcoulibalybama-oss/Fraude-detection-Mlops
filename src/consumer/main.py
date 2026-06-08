import json
import joblib
import pandas as pd
from kafka import KafkaConsumer
from loguru import logger
from pymongo import MongoClient
from datetime import datetime
from prometheus_client import Counter, Gauge, start_http_server

# ── Métriques Prometheus ───────────────────────────────────
start_http_server(8000)

transactions_total = Counter(
    'fraud_transactions_total',
    'Nombre total de transactions traitees',
    ['drift_status']
)
fraud_total = Counter(
    'fraud_detected_total',
    'Nombre total de fraudes detectees'
)
score_gauge = Gauge(
    'fraud_score_current',
    'Score de fraude de la derniere transaction'
)
score_moyen_gauge = Gauge(
    'fraud_score_moyen',
    'Score moyen des 100 dernieres transactions'
)

# ── Modele ─────────────────────────────────────────────────
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

score_history = []

def main():
    logger.info("Consumer demarre — metriques sur port 8000")
    for message in consumer:
        row = message.value
        score = score_transaction(row)
        is_fraud = score > 0.5
        drift_status = row.get("drift_status", "NORMAL")
        drift_phase = row.get("drift_phase", "NORMAL")
        drift_level = int(row.get("drift_level", 0))
        mois_simule = row.get("mois_simule", "2025-01")

        result = {
            "transaction_id": int(row.get("TransactionID", 0)),
            "amount": float(row.get("TransactionAmt", 0)),
            "score": score,
            "is_fraud": is_fraud,
            "drift_status": drift_status,
            "drift_phase": drift_phase,
            "drift_level": drift_level,
            "mois_simule": mois_simule,
            "timestamp": datetime.utcnow(),
        }

        # ── MongoDB ────────────────────────────────────────
        collection.insert_one(result)

        # ── Prometheus ─────────────────────────────────────
        transactions_total.labels(drift_status=drift_status).inc()
        score_gauge.set(score)

        if is_fraud:
            fraud_total.inc()

        score_history.append(score)
        if len(score_history) > 100:
            score_history.pop(0)
        score_moyen_gauge.set(sum(score_history) / len(score_history))

        logger.info(f"ALERT: {result}")

if __name__ == "__main__":
    main()
