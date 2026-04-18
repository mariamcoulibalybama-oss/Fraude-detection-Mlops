# Fraude MLOps Streaming

Plateforme de détection de fraude bancaire en streaming avec monitoring de drift et retraining automatique.

## Vue d'ensemble

Ce projet implémente un pipeline complet de détection de fraude en temps réel :

- **Ingestion** : producer Python qui simule un flux de transactions via Kafka
- **Feature store** : Redis pour les agrégats glissants temps réel
- **Modèle** : XGBoost entraîné sur IEEE-CIS, explicabilité via SHAP
- **Stockage** : MongoDB pour l'historique complet
- **MLOps** : MLflow (tracking), Evidently (drift), Prometheus + Grafana (monitoring)
- **UI** : Streamlit pour les analystes

## Workflow de développement

```
Machine locale              GitHub              VPS (production)
─────────────────          ──────             ──────────────────
Écrire le code      ─push─►                        
                             ├─pull─►  docker-compose up -d
Commit + push       ─────►                         
                                                  Stack tourne en continu
```

Cycle type : code en local → commit → push → SSH au VPS → pull → redémarrer Docker.

## Prérequis

### Sur ta machine locale
- **Python 3.10+**
- **Docker Desktop** (pour tester en local)
- **Git**
- **VS Code** (recommandé) avec l'extension **Remote - SSH**

### Sur le VPS
- **Docker + Docker Compose** (déjà installé)
- **Accès SSH** avec ton utilisateur dédié

## Installation

### 1. Cloner le repo sur ta machine locale

```bash
git clone https://github.com/<ton-username>/fraude-mlops-streaming.git
cd fraude-mlops-streaming
```

### 2. Créer un environnement Python virtuel

```bash
python3 -m venv venv
source venv/bin/activate   # Linux / Mac
# ou
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 3. Configurer les variables d'environnement

```bash
cp .env.example .env
# Éditer .env avec tes valeurs (rien à changer pour un usage local standard)
```

### 4. Tester la connectivité Docker (en local)

```bash
docker-compose up -d kafka redis mongodb
python scripts/test_connectivity.py
```

Si tout est vert, la base fonctionne.

### 5. Télécharger le dataset

```bash
# Depuis Kaggle (nécessite un compte Kaggle et kaggle CLI configuré)
kaggle competitions download -c ieee-fraud-detection -p data/raw/
unzip data/raw/ieee-fraud-detection.zip -d data/raw/
```

## Déploiement sur le VPS

### Première fois

```bash
# Se connecter au VPS
ssh stagiaire@72.62.16.22

# Cloner le repo
cd /home/stagiaire/projet-fraude
git clone https://github.com/<ton-username>/fraude-mlops-streaming.git .

# Copier la config
cp .env.example .env
# Adapter .env si nécessaire

# Lancer la stack
docker-compose up -d

# Vérifier
docker ps
python scripts/test_connectivity.py
```

### Mises à jour ultérieures

```bash
ssh stagiaire@72.62.16.22
cd /home/stagiaire/projet-fraude
git pull
docker-compose up -d --build
```

## Accès aux interfaces

Une fois la stack lancée, les interfaces sont accessibles sur :

| Service     | URL locale                      | URL VPS                       |
|-------------|----------------------------------|-------------------------------|
| Streamlit   | http://localhost:8501            | http://72.62.16.22:8501       |
| MLflow UI   | http://localhost:5000            | http://72.62.16.22:5000       |
| Grafana     | http://localhost:3001            | http://72.62.16.22:3001       |
| Prometheus  | http://localhost:9090            | http://72.62.16.22:9090       |

**Identifiants Grafana par défaut** : `admin` / `admin` (à changer à la première connexion).

## Structure du projet

```
fraude-mlops-streaming/
├── src/
│   ├── producer/          # Génération de transactions vers Kafka
│   ├── consumer/          # Consommation du stream + scoring
│   ├── model/             # Entraînement et chargement du modèle
│   ├── features/          # Feature engineering + feature store Redis
│   ├── monitoring/        # Drift detection avec Evidently
│   └── utils/             # Utilitaires partagés (config, logging)
├── notebooks/             # Exploration et expérimentation
├── docker/                # Dockerfiles et configs Docker
├── tests/                 # Tests unitaires (pytest)
├── data/
│   ├── raw/              # Dataset brut IEEE-CIS
│   ├── processed/        # Dataset préparé
│   └── generated/        # Données générées pour le stream
├── config/                # Fichiers de configuration (YAML)
├── scripts/               # Scripts utilitaires
├── docs/                  # Documentation du projet
├── docker-compose.yml     # Orchestration de la stack
├── requirements.txt       # Dépendances Python
├── .env.example           # Variables d'environnement modèle
└── README.md
```

## Planning du projet

| Semaine | Phase                          | Livrable                              |
|---------|--------------------------------|---------------------------------------|
| 1       | Cadrage + environnement        | Cadrage + stack de base (Kafka/Redis/Mongo) |
| 2       | EDA IEEE-CIS                   | Notebook exploratoire + dataset préparé |
| 3       | Modélisation XGBoost + SHAP    | Modèle enregistré dans MLflow          |
| 4       | Producer Kafka                 | Générateur de transactions + drift injectable |
| 5       | Consumer + pipeline end-to-end | Pipeline complet fonctionnel           |
| 6       | Monitoring (Evidently/Prometheus) | Dashboards drift + techniques       |
| 7       | Retraining auto + UI Streamlit | Boucle MLOps complète + UI            |
| 8       | Rapport + soutenance           | Rapport final + démo                  |

## KPIs cibles

| KPI                          | Cible         |
|------------------------------|---------------|
| Recall                       | ≥ 75 %        |
| Precision                    | ≥ 60 %        |
| AUPRC                        | ≥ 0,75        |
| Latence scoring (P95)        | < 500 ms      |
| Throughput                   | ≥ 50 tx/sec   |
| Temps de détection de drift  | < 5 min       |
| Temps de retraining auto     | < 15 min      |

## Commandes utiles

```bash
# Voir les logs d'un service
docker-compose logs -f kafka

# Redémarrer un service
docker-compose restart consumer

# Tout arrêter (sans supprimer les données)
docker-compose stop

# Tout supprimer (conteneurs + réseaux, mais pas les volumes)
docker-compose down

# Tout supprimer incluant les données
docker-compose down -v

# Lancer les tests
pytest tests/

# Formater le code
black src/ tests/
```

## En cas de problème

Consulte `docs/TROUBLESHOOTING.md` pour les problèmes fréquents (Kafka qui ne démarre pas, erreurs de permissions, etc.).

## Licence

Projet pédagogique — usage éducatif uniquement.
