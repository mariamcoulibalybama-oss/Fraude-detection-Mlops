"""
Module de configuration centralisé.

Charge les variables d'environnement depuis .env et les expose
via une classe Config typée.
"""

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic_transactions: str = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")
    topic_alerts: str = os.getenv("KAFKA_TOPIC_ALERTS", "alerts")


@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))


@dataclass
class MongoConfig:
    host: str = os.getenv("MONGO_HOST", "localhost")
    port: int = int(os.getenv("MONGO_PORT", "27017"))
    user: str = os.getenv("MONGO_USER", "admin")
    password: str = os.getenv("MONGO_PASSWORD", "changeme")
    database: str = os.getenv("MONGO_DB", "fraude")

    @property
    def uri(self) -> str:
        return f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}/"


@dataclass
class MLflowConfig:
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud_detection")


@dataclass
class ModelConfig:
    name: str = os.getenv("MODEL_NAME", "fraud_xgboost")
    stage: str = os.getenv("MODEL_STAGE", "Production")
    fraud_threshold: float = float(os.getenv("FRAUD_THRESHOLD", "0.5"))


@dataclass
class ProducerConfig:
    transactions_per_second: int = int(os.getenv("TRANSACTIONS_PER_SECOND", "10"))
    inject_drift_after_minutes: int = int(os.getenv("INJECT_DRIFT_AFTER_MINUTES", "30"))


@dataclass
class Config:
    """Configuration globale du projet."""
    kafka: KafkaConfig
    redis: RedisConfig
    mongo: MongoConfig
    mlflow: MLflowConfig
    model: ModelConfig
    producer: ProducerConfig
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            kafka=KafkaConfig(),
            redis=RedisConfig(),
            mongo=MongoConfig(),
            mlflow=MLflowConfig(),
            model=ModelConfig(),
            producer=ProducerConfig(),
        )


# Instance globale utilisable directement
config = Config.from_env()
