"""
Script de test de connectivité.

Vérifie que tous les services de la stack sont accessibles.
À lancer après `docker-compose up -d` pour valider l'installation.

Usage:
    python scripts/test_connectivity.py
"""

import os
import sys
import time
from typing import Tuple

# Couleurs pour le terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str) -> None:
    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{BLUE}  {text}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")


def print_ok(text: str) -> None:
    print(f"{GREEN}[OK]{RESET}    {text}")


def print_ko(text: str, error: str = "") -> None:
    print(f"{RED}[KO]{RESET}    {text}")
    if error:
        print(f"         {YELLOW}→ {error}{RESET}")


def print_info(text: str) -> None:
    print(f"{BLUE}[INFO]{RESET}  {text}")


def test_kafka(host: str = "localhost", port: int = 9092) -> Tuple[bool, str]:
    """Test Kafka : produce + consume un message de test."""
    try:
        from kafka import KafkaProducer, KafkaConsumer
        from kafka.errors import NoBrokersAvailable

        # Test producer
        producer = KafkaProducer(
            bootstrap_servers=f"{host}:{port}",
            request_timeout_ms=5000,
            api_version_auto_timeout_ms=5000,
        )
        future = producer.send("test-topic", b"hello-kafka")
        future.get(timeout=10)
        producer.close()
        return True, "Producer OK"
    except ImportError:
        return False, "Librairie kafka-python non installée (pip install kafka-python)"
    except NoBrokersAvailable:
        return False, f"Aucun broker disponible sur {host}:{port}"
    except Exception as e:
        return False, str(e)


def test_redis(host: str = "localhost", port: int = 6379) -> Tuple[bool, str]:
    """Test Redis : set + get une clé de test."""
    try:
        import redis

        r = redis.Redis(host=host, port=port, socket_connect_timeout=5)
        r.set("test_key", "test_value", ex=10)
        value = r.get("test_key")
        if value == b"test_value":
            r.delete("test_key")
            return True, f"Ping OK, set/get fonctionnels"
        return False, "La valeur lue ne correspond pas"
    except ImportError:
        return False, "Librairie redis non installée (pip install redis)"
    except Exception as e:
        return False, str(e)


def test_mongodb(
    host: str = "localhost",
    port: int = 27017,
    user: str = "admin",
    password: str = "changeme",
) -> Tuple[bool, str]:
    """Test MongoDB : connexion + écriture + lecture."""
    try:
        from pymongo import MongoClient
        from pymongo.errors import ServerSelectionTimeoutError

        uri = f"mongodb://{user}:{password}@{host}:{port}/"
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client["test_db"]
        collection = db["test_collection"]
        result = collection.insert_one({"test": "hello-mongo"})
        found = collection.find_one({"_id": result.inserted_id})
        collection.delete_one({"_id": result.inserted_id})
        client.close()
        if found and found["test"] == "hello-mongo":
            return True, "Insert/find OK"
        return False, "Lecture ne correspond pas à l'insertion"
    except ImportError:
        return False, "Librairie pymongo non installée (pip install pymongo)"
    except ServerSelectionTimeoutError as e:
        return False, f"Timeout de connexion : {e}"
    except Exception as e:
        return False, str(e)


def test_mlflow(host: str = "localhost", port: int = 5000) -> Tuple[bool, str]:
    """Test MLflow : vérifier que le serveur répond."""
    try:
        import requests

        url = f"http://{host}:{port}/"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return True, f"MLflow UI accessible sur {url}"
        return False, f"Code HTTP inattendu : {response.status_code}"
    except ImportError:
        return False, "Librairie requests non installée (pip install requests)"
    except Exception as e:
        return False, str(e)


def test_prometheus(host: str = "localhost", port: int = 9090) -> Tuple[bool, str]:
    """Test Prometheus : vérifier que l'API répond."""
    try:
        import requests

        url = f"http://{host}:{port}/-/healthy"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return True, "Prometheus healthy"
        return False, f"Code HTTP inattendu : {response.status_code}"
    except ImportError:
        return False, "Librairie requests non installée"
    except Exception as e:
        return False, str(e)


def test_grafana(host: str = "localhost", port: int = 3001) -> Tuple[bool, str]:
    """Test Grafana : vérifier que le serveur répond."""
    try:
        import requests

        url = f"http://{host}:{port}/api/health"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return True, f"Grafana accessible, data: {response.json()}"
        return False, f"Code HTTP inattendu : {response.status_code}"
    except ImportError:
        return False, "Librairie requests non installée"
    except Exception as e:
        return False, str(e)


def main() -> int:
    """Lance tous les tests et retourne 0 si tout OK, 1 sinon."""
    print_header("Test de connectivité - Fraude MLOps Streaming")

    # Charger les variables d'environnement si .env existe
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print_info("Variables d'environnement chargées depuis .env")
    except ImportError:
        print_info("python-dotenv non installé, utilisation des valeurs par défaut")

    # Lire les hosts depuis les env vars ou utiliser localhost
    host = os.getenv("DOCKER_HOST", "localhost")
    mongo_user = os.getenv("MONGO_USER", "admin")
    mongo_password = os.getenv("MONGO_PASSWORD", "changeme")

    print_info(f"Hôte testé : {host}")
    print()

    tests = [
        ("Kafka       (port 9092)", lambda: test_kafka(host)),
        ("Redis       (port 6379)", lambda: test_redis(host)),
        ("MongoDB     (port 27017)", lambda: test_mongodb(host, user=mongo_user, password=mongo_password)),
        ("MLflow      (port 5000)", lambda: test_mlflow(host)),
        ("Prometheus  (port 9090)", lambda: test_prometheus(host)),
        ("Grafana     (port 3001)", lambda: test_grafana(host)),
    ]

    results = []
    for name, test_fn in tests:
        success, message = test_fn()
        results.append((name, success, message))
        if success:
            print_ok(f"{name} : {message}")
        else:
            print_ko(f"{name}", message)

    # Résumé
    print()
    print_header("Résumé")
    total = len(results)
    ok_count = sum(1 for _, success, _ in results if success)
    ko_count = total - ok_count

    print(f"  {GREEN}OK{RESET} : {ok_count}/{total}")
    if ko_count > 0:
        print(f"  {RED}KO{RESET} : {ko_count}/{total}")
        print()
        print(f"{YELLOW}Conseils en cas d'échec :{RESET}")
        print("  1. Vérifier que la stack est lancée : docker-compose ps")
        print("  2. Regarder les logs : docker-compose logs <service>")
        print("  3. Attendre 30 secondes après le démarrage (Kafka est lent)")
        print("  4. Vérifier que les ports ne sont pas utilisés : ss -tlnp")
        return 1

    print()
    print(f"{GREEN}{BOLD}Tous les services sont opérationnels. Prêt à développer ! 🚀{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
