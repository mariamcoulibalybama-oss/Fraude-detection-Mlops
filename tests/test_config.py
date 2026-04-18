"""
Tests unitaires pour le module de configuration.

Lancer tous les tests : pytest tests/
"""

import pytest
from src.utils.config import Config, KafkaConfig, RedisConfig


def test_kafka_config_defaults():
    """Vérifie les valeurs par défaut de la config Kafka."""
    kafka = KafkaConfig()
    assert kafka.bootstrap_servers is not None
    assert kafka.topic_transactions == "transactions"


def test_redis_config_defaults():
    """Vérifie les valeurs par défaut de la config Redis."""
    redis = RedisConfig()
    assert redis.host is not None
    assert isinstance(redis.port, int)
    assert redis.port > 0


def test_global_config_instantiation():
    """Vérifie que la config globale s'instancie correctement."""
    config = Config.from_env()
    assert config.kafka is not None
    assert config.redis is not None
    assert config.mongo is not None
    assert config.mlflow is not None


def test_mongo_uri_format():
    """Vérifie le format de l'URI MongoDB."""
    config = Config.from_env()
    uri = config.mongo.uri
    assert uri.startswith("mongodb://")
    assert "@" in uri
    assert str(config.mongo.port) in uri
