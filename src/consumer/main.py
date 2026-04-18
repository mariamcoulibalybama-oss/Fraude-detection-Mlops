"""
Consumer Kafka - template de démarrage.

Ce script est un squelette. À développer dans la semaine 5 du planning.

Objectif final :
- Consommer les transactions depuis Kafka
- Enrichir avec des features en temps réel (via Redis)
- Appeler le modèle XGBoost pour scorer
- Générer l'explication SHAP
- Stocker le résultat dans MongoDB
- Publier les alertes sur le topic 'alerts'

Usage:
    python -m src.consumer.main
"""

import json
from typing import Dict, Any

from kafka import KafkaConsumer

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_consumer() -> KafkaConsumer:
    """Crée un consumer Kafka avec désérialisation JSON."""
    return KafkaConsumer(
        config.kafka.topic_transactions,
        bootstrap_servers=config.kafka.bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="fraude-consumer-group",
    )


def process_transaction(tx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traite une transaction : enrichissement + scoring + stockage.

    TODO semaine 5 :
    - Récupérer les agrégats glissants depuis Redis
    - Charger le modèle depuis MLflow
    - Calculer les features finales
    - Scorer la transaction
    - Générer l'explication SHAP
    - Stocker dans MongoDB
    """
    # Placeholder : scoring aléatoire pour l'instant
    import random
    fraud_score = random.random()
    is_fraud = fraud_score > config.model.fraud_threshold

    return {
        **tx,
        "fraud_score": fraud_score,
        "is_fraud": is_fraud,
        "model_version": "placeholder-0.1",
    }


def main():
    logger.info("Démarrage du consumer")
    consumer = create_consumer()
    count = 0

    try:
        for message in consumer:
            tx = message.value
            result = process_transaction(tx)

            count += 1
            if count % 100 == 0:
                logger.info(f"{count} transactions traitées")

            if result["is_fraud"]:
                logger.warning(f"FRAUDE détectée : {result['transaction_id']} (score: {result['fraud_score']:.3f})")

    except KeyboardInterrupt:
        logger.info("Arrêt du consumer demandé")
    finally:
        consumer.close()
        logger.info(f"Consumer arrêté. Total : {count} transactions traitées")


if __name__ == "__main__":
    main()
