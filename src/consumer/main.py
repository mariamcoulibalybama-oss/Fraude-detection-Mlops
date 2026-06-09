import json
import time
import joblib
import pandas as pd
from kafka import KafkaConsumer
from loguru import logger
from pymongo import MongoClient
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ── Métriques Prometheus ───────────────────────────────────
# Démarre le serveur de métriques sur le port 8000
start_http_server(8000)

# Compteur total de transactions traitées par statut (NORMAL/DRIFT)
transactions_total = Counter(
    'fraud_transactions_total',
    'Nombre total de transactions traitees',
    ['drift_status']
)

# Compteur total de fraudes détectées
fraud_total = Counter(
    'fraud_detected_total',
    'Nombre total de fraudes detectees'
)

# Jauge du score de la dernière transaction
score_gauge = Gauge(
    'fraud_score_current',
    'Score de fraude de la derniere transaction'
)

# Jauge du score moyen des 100 dernières transactions
score_moyen_gauge = Gauge(
    'fraud_score_moyen',
    'Score moyen des 100 dernieres transactions'
)

# Histogramme de la latence de scoring en secondes
scoring_latency = Histogram(
    'fraud_scoring_latency_seconds',
    'Latence de scoring en secondes',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Jauge du throughput (transactions par seconde)
throughput_gauge = Gauge(
    'fraud_throughput_tps',
    'Transactions traitees par seconde'
)

# ── Chargement du modèle ───────────────────────────────────
# Charge le pipeline complet (modèle + features)
pipeline = joblib.load("models/pipeline_v1.pkl")
model = pipeline["model"]
feature_names = pipeline["feature_names"]

# ── Connexion MongoDB ──────────────────────────────────────
# Se connecte à MongoDB pour stocker les prédictions
mongo = MongoClient("mongodb://admin:changeme@localhost:27017/")
db = mongo["fraude_db"]
collection = db["predictions"]

# ── Connexion Kafka ────────────────────────────────────────
# Écoute le topic "transactions" en continu
TOPIC = "transactions"
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8"))
)

def preprocess(row):
    """Aligne les colonnes de la transaction sur les features du modèle"""
    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_names, fill_value=0)
    return df

def score_transaction(row):
    """
    Calcule le score de fraude et mesure la latence
    Retourne (score, latence_ms)
    """
    try:
        # Démarrer le chronomètre
        start = time.time()

        df = preprocess(row)
        score = model.predict_proba(df)[0][1]

        # Calculer la latence en millisecondes
        latence_ms = (time.time() - start) * 1000

        # Enregistrer la latence dans Prometheus
        scoring_latency.observe(latence_ms / 1000)

        return float(score), latence_ms

    except Exception as e:
        logger.error(f"Model fallback: {e}")
        return 0.5, 0.0

# Historique des scores pour calculer la moyenne
score_history = []

# Variables pour calculer le throughput
window_start = time.time()
window_count = 0

def main():
    logger.info("Consumer demarre — metriques sur port 8000")

    global window_start, window_count

    for message in consumer:
        row = message.value

        # Scorer la transaction et mesurer la latence
        score, latence_ms = score_transaction(row)
        is_fraud = score > 0.5

        # Récupérer les infos de drift
        drift_status = row.get("drift_status", "NORMAL")
        drift_phase = row.get("drift_phase", "NORMAL")
        drift_level = int(row.get("drift_level", 0))
        mois_simule = row.get("mois_simule", "2025-01")

        # Construire le résultat à sauvegarder
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

        # Sauvegarder dans MongoDB
        collection.insert_one(result)

        # ── Mettre à jour les métriques Prometheus ─────────
        # Incrémenter le compteur de transactions
        transactions_total.labels(drift_status=drift_status).inc()

        # Mettre à jour le score actuel
        score_gauge.set(score)

        # Incrémenter le compteur de fraudes si détectée
        if is_fraud:
            fraud_total.inc()

        # Calculer le score moyen sur les 100 dernières transactions
        score_history.append(score)
        if len(score_history) > 100:
            score_history.pop(0)
        score_moyen_gauge.set(sum(score_history) / len(score_history))

        # Calculer le throughput toutes les 10 secondes
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
