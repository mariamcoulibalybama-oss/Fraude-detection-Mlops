import json
import time
import joblib
import pandas as pd
from kafka import KafkaConsumer
from loguru import logger
from pymongo import MongoClient
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from dotenv import load_dotenv
load_dotenv()

from src.features.behavioral import get_behavioral_features

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

scoring_latency = Histogram(
    'fraud_scoring_latency_seconds',
    'Latence de scoring en secondes',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

throughput_gauge = Gauge(
    'fraud_throughput_tps',
    'Transactions traitees par seconde'
)

# ── Chargement du modèle ───────────────────────────────────
pipeline = joblib.load("models/pipeline_v1.pkl")
model = pipeline["model"]
feature_names = pipeline["feature_names"]

# ── Connexion MongoDB ──────────────────────────────────────
mongo = MongoClient("mongodb://admin:changeme@localhost:27017/")
db = mongo["fraude_db"]
collection = db["predictions"]

# ── Connexion Kafka ────────────────────────────────────────
TOPIC = "transactions"
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="earliest",
    group_id="fraude-consumer-group",
    enable_auto_commit=True
)

def preprocess(row):
    df = pd.DataFrame([row])

    if "TransactionDT" in df.columns:
        df["heure_transaction"] = (df["TransactionDT"] // 3600) % 24
    else:
        df["heure_transaction"] = 0

    # ── Features comportementales Redis (module src.features) ──
    behav = get_behavioral_features(row)
    for k, v in behav.items():
        df[k] = v

    df = df.reindex(columns=feature_names, fill_value=0)
    return df

def score_transaction(row):
    """Calcule le score de fraude et mesure la latence. Retourne (score, latence_ms)."""
    try:
        start = time.time()

        df = preprocess(row)
        score = model.predict_proba(df)[0][1]

        latence_ms = (time.time() - start) * 1000
        scoring_latency.observe(latence_ms / 1000)

        return float(score), latence_ms

    except Exception as e:
        logger.error(f"Model fallback: {e}")
        return 0.5, 0.0

score_history = []
window_start = time.time()
window_count = 0

def main():
    logger.info("Consumer demarre — metriques sur port 8000")

    global window_start, window_count

    for message in consumer:
        row = message.value

        score, latence_ms = score_transaction(row)
        is_fraud = score > 0.5

        drift_status = row.get("drift_status", "NORMAL")
        drift_phase = row.get("drift_phase", "NORMAL")
        drift_level = int(row.get("drift_level", 0))
        mois_simule = row.get("mois_simule", "2026-01")

        result = {
            "transaction_id": int(row.get("TransactionID", 0)),
            "amount": float(row.get("TransactionAmt", 0)),
            "score": score,
            "is_fraud": is_fraud,
            "drift_status": drift_status,
            "drift_phase": drift_phase,
            "drift_level": drift_level,
            "mois_simule": mois_simule,
            "latence_ms": round(latence_ms, 2),
            "timestamp": datetime.utcnow(),
        }

        collection.insert_one(result)

        transactions_total.labels(drift_status=drift_status).inc()
        score_gauge.set(score)

        if is_fraud:
            fraud_total.inc()

        score_history.append(score)
        if len(score_history) > 100:
            score_history.pop(0)
        score_moyen_gauge.set(sum(score_history) / len(score_history))

        window_count += 1
        elapsed = time.time() - window_start
        if elapsed >= 10:
            tps = window_count / elapsed
            throughput_gauge.set(tps)
            logger.info(f"Throughput : {tps:.1f} tx/sec | Latence : {latence_ms:.2f}ms")
            window_start = time.time()
            window_count = 0

        logger.info(f"ALERT: {result}")

if __name__ == "__main__":
    main()
