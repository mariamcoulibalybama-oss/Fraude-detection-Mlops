# Guide de démarrage - premier jour

Ce guide te guide pas à pas pour ton premier jour sur le projet.
Temps estimé : **2 heures**.

## Étape 1 — Installer les outils sur ta machine (30 min)

### Windows

1. Installer [Python 3.10+](https://www.python.org/downloads/) (cocher "Add Python to PATH" à l'installation)
2. Installer [Git for Windows](https://git-scm.com/download/win)
3. Installer [Docker Desktop](https://www.docker.com/products/docker-desktop/)
4. Installer [VS Code](https://code.visualstudio.com/)
5. Dans VS Code, installer les extensions : **Python**, **Remote - SSH**, **Docker**

### Mac

```bash
# Si tu n'as pas Homebrew : https://brew.sh
brew install python@3.11 git
brew install --cask docker visual-studio-code
```

Puis dans VS Code, installer les extensions **Python**, **Remote - SSH**, **Docker**.

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# VS Code : télécharger depuis code.visualstudio.com
```

### Vérifier l'installation

```bash
python --version    # Python 3.10+
git --version
docker --version
docker compose version
```

## Étape 2 — Configurer Git (5 min)

```bash
git config --global user.name "Ton Nom"
git config --global user.email "ton.email@example.com"
```

## Étape 3 — Créer le repo GitHub (10 min)

1. Aller sur [github.com](https://github.com) et se connecter (ou créer un compte).
2. Cliquer sur "New repository".
3. Nom : `fraude-mlops-streaming`
4. Description : "Plateforme de détection de fraude en streaming avec MLOps"
5. Cocher "Private" (au moins pour l'instant)
6. **Ne pas** cocher "Initialize with README" (on l'a déjà)
7. Créer le repo.

GitHub te donne une URL du type `https://github.com/ton-username/fraude-mlops-streaming.git` — la garder sous le coude.

## Étape 4 — Initialiser le projet en local (15 min)

```bash
# Naviguer où tu veux stocker tes projets
cd ~/projets       # Mac/Linux
# ou
cd C:\Users\TonNom\projets    # Windows

# Cloner le template (ou copier les fichiers fournis)
mkdir fraude-mlops-streaming
cd fraude-mlops-streaming

# Initialiser Git
git init
git branch -M main
git remote add origin https://github.com/ton-username/fraude-mlops-streaming.git
```

Copier tous les fichiers du template fourni dans ce dossier, puis :

```bash
# Créer et activer l'environnement virtuel Python
python -m venv venv

# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Copier la config d'environnement
cp .env.example .env
# (sur Windows : copy .env.example .env)
```

## Étape 5 — Premier test en local (20 min)

```bash
# Lancer seulement Kafka, Redis, MongoDB pour commencer
docker compose up -d zookeeper kafka redis mongodb

# Attendre 30 secondes (Kafka est lent à démarrer)

# Vérifier que tout tourne
docker ps

# Tester la connectivité
python scripts/test_connectivity.py
```

Tu dois voir 4 lignes vertes `[OK]` pour Kafka, Redis, MongoDB (les autres en KO, c'est normal, ils ne sont pas lancés).

## Étape 6 — Premier commit (5 min)

```bash
git add .
git commit -m "Initial setup : stack Docker de base (Kafka + Redis + MongoDB)"
git push -u origin main
```

Aller sur GitHub, vérifier que ton code apparaît bien.

## Étape 7 — Configurer la connexion SSH au VPS (15 min)

### Générer une clé SSH (si tu n'en as pas)

```bash
ssh-keygen -t ed25519 -C "ton.email@example.com"
# Appuyer sur Entrée pour accepter tous les défauts
```

### Ajouter la clé au VPS

Envoyer la clé publique à ton encadrant pour qu'il l'ajoute au VPS, ou utiliser `ssh-copy-id` :

```bash
ssh-copy-id stagiaire@72.62.16.22
# Entrer le mot de passe qu'on t'a donné
```

### Tester la connexion

```bash
ssh stagiaire@72.62.16.22
# Si ça se connecte sans demander de mot de passe, c'est gagné
```

### Changer le mot de passe initial (une fois connecté au VPS)

```bash
passwd
# Ancien mot de passe : celui qu'on t'a donné
# Nouveau : un mot de passe solide
```

## Étape 8 — Déployer sur le VPS pour la première fois (15 min)

```bash
# Se connecter au VPS
ssh stagiaire@72.62.16.22

# Aller dans ton dossier de travail
cd /home/stagiaire/projet-fraude

# Cloner ton repo
git clone https://github.com/ton-username/fraude-mlops-streaming.git .

# Copier la config
cp .env.example .env
# Adapter si besoin (vim .env ou nano .env)

# Lancer la stack
docker compose up -d zookeeper kafka redis mongodb

# Attendre 30 secondes puis tester
sleep 30
python3 scripts/test_connectivity.py
```

## Étape 9 — Configurer VS Code Remote SSH (10 min)

Dans VS Code :

1. Ouvrir la palette de commandes (Ctrl+Shift+P ou Cmd+Shift+P)
2. Taper "Remote-SSH: Connect to Host"
3. "Add New SSH Host"
4. Entrer : `ssh stagiaire@72.62.16.22`
5. Sauvegarder dans la config SSH par défaut
6. Se reconnecter à ce host

Tu peux maintenant ouvrir le dossier `/home/stagiaire/projet-fraude` dans VS Code comme s'il était en local — c'est beaucoup plus confortable que de travailler en SSH pur.

## Récapitulatif de ton setup

À la fin de cette journée, tu dois avoir :

- [x] Python, Git, Docker, VS Code installés
- [x] Un repo GitHub `fraude-mlops-streaming`
- [x] Une copie locale du projet avec `venv` activé
- [x] Kafka, Redis, MongoDB qui tournent en local via Docker
- [x] Test de connectivité qui passe en local
- [x] Connexion SSH au VPS fonctionnelle
- [x] Mot de passe VPS changé
- [x] Kafka, Redis, MongoDB qui tournent aussi sur le VPS
- [x] Test de connectivité qui passe sur le VPS
- [x] VS Code Remote SSH configuré

## Ton workflow à partir de maintenant

```bash
# 1. Code sur ta machine locale dans VS Code
# 2. Tester en local avec Docker
# 3. Commit + push
git add .
git commit -m "Description du changement"
git push

# 4. Déployer sur le VPS
ssh stagiaire@72.62.16.22
cd /home/stagiaire/projet-fraude
git pull
docker compose up -d --build   # Rebuild si les images ont changé
```

## Prochaine étape : semaine 2

Télécharger le dataset IEEE-CIS et commencer l'exploration. Voir `notebooks/01_eda_ieee_cis.ipynb` (à créer).

## En cas de problème

Consulter `docs/TROUBLESHOOTING.md`.

Si tu es bloqué plus de 30 minutes sur un problème, demander de l'aide à ton encadrant plutôt que de t'acharner seul.

Bon courage !
