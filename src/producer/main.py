"""
Producer Kafka - template de démarrage.

Ce script est un squelette. À développer dans la semaine 4 du planning.

Objectif final :
- Lire des transactions depuis IEEE-CIS (ou générer aléatoirement)
- Les publier sur le topic Kafka 'transactions'
- Injecter du drift après X minutes pour tester le monitoring

Usage:
    python -m src.producer.main
"""

import json
import time
import random
from datetime import datetime
from typing import Dict, Any

from kafka import KafkaProducer

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_producer() -> KafkaProducer:
    """Crée un producer Kafka avec sérialisation JSON."""
    return KafkaProducer(
        bootstrap_servers=config.kafka.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
    )


def generate_transaction(drift: bool = False) -> Dict[str, Any]:
    """
    Génère une transaction aléatoire (version basique).

    TODO semaine 4 :
    - S'inspirer de la distribution réelle d'IEEE-CIS
    - Ajouter plus de features (DeviceType, ProductCD, etc.)
    - Gérer le mode 'drift' : décaler les distributions
    """
    base_amount = 50.0 if not drift else 200.0  # Drift : montants plus élevés
    return {
        "transaction_id": f"tx_{int(time.time() * 1000)}_{random.randint(0, 9999)}",
        "timestamp": datetime.utcnow().isoformat(),
        "amount": round(random.expovariate(1 / base_amount), 2),
        "card_id": f"card_{random.randint(1, 10000)}",
        "merchant_id": f"merch_{random.randint(1, 500)}",
        "is_mobile": random.random() > 0.5,
    }


def main():
    logger.info(f"Démarrage du producer - cadence : {config.producer.transactions_per_second} tx/s")
    logger.info(f"Injection de drift après : {config.producer.inject_drift_after_minutes} minutes")

    producer = create_producer()
    start_time = time.time()
    count = 0

    try:
        while True:
            # Calculer si on est en mode drift
            elapsed_minutes = (time.time() - start_time) / 60
            drift_active = elapsed_minutes >= config.producer.inject_drift_after_minutes

            # Générer et envoyer une transaction
            tx = generate_transaction(drift=drift_active)
            producer.send(config.kafka.topic_transactions, tx)
            count += 1

            if count % 100 == 0:
                mode = "DRIFT" if drift_active else "NORMAL"
                logger.info(f"[{mode}] {count} transactions envoyées")

            # Respecter la cadence
            time.sleep(1 / config.producer.transactions_per_second)

    except KeyboardInterrupt:
        logger.info("Arrêt du producer demandé")
    finally:
        producer.flush()
        producer.close()
        logger.info(f"Producer arrêté. Total : {count} transactions envoyées")


if __name__ == "__main__":
    main()
