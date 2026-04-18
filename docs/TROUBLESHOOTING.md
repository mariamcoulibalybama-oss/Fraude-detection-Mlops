# Troubleshooting

Guide de résolution des problèmes fréquents.

## Kafka ne démarre pas / erreur "NoBrokersAvailable"

**Symptôme** : le test de connectivité échoue sur Kafka, ou le producer retourne `kafka.errors.NoBrokersAvailable`.

**Solutions dans l'ordre** :

1. **Attendre 30 secondes** après `docker-compose up -d` — Kafka met du temps à démarrer.

2. Vérifier que Zookeeper est bien démarré avant Kafka :
   ```bash
   docker-compose logs zookeeper | tail -20
   docker-compose logs kafka | tail -20
   ```

3. Redémarrer Kafka seul :
   ```bash
   docker-compose restart kafka
   ```

4. Si le problème persiste, nettoyer les volumes :
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

## Erreur "port is already allocated"

**Symptôme** : `docker-compose up` échoue avec un message du genre `Bind for 0.0.0.0:5000 failed: port is already allocated`.

**Cause** : un autre service utilise déjà ce port sur ta machine.

**Solutions** :

1. Identifier le service qui utilise le port :
   ```bash
   sudo ss -tlnp | grep :5000
   # ou sur Mac
   lsof -i :5000
   ```

2. Arrêter ce service, ou changer le port dans `docker-compose.yml` :
   ```yaml
   mlflow:
     ports:
       - "5001:5000"   # Utiliser 5001 côté host
   ```

## MongoDB : "Authentication failed"

**Symptôme** : le consumer/Streamlit n'arrive pas à se connecter à MongoDB.

**Cause** : les identifiants dans `.env` ne correspondent pas à ceux utilisés au démarrage.

**Solution** : supprimer le volume MongoDB et le recréer avec les bons identifiants :
```bash
docker-compose stop mongodb
docker volume rm fraude-mlops_mongodb_data
docker-compose up -d mongodb
```

## Le conteneur est killé (OOM)

**Symptôme** : un conteneur redémarre en boucle, `docker-compose logs <service>` montre `Killed`.

**Cause** : manque de RAM sur la machine.

**Solutions** :

1. Vérifier l'utilisation mémoire :
   ```bash
   docker stats --no-stream
   free -h
   ```

2. Fermer les services non essentiels (Grafana, Prometheus) pendant le développement :
   ```bash
   docker-compose stop grafana prometheus mlflow
   ```

3. Ajuster les limites dans `docker-compose.yml` :
   ```yaml
   kafka:
     environment:
       - KAFKA_HEAP_OPTS=-Xmx256m -Xms128m  # Réduire si besoin
   ```

## Permissions Docker sur le VPS

**Symptôme** : `docker ps` retourne `permission denied`.

**Cause** : l'utilisateur n'est pas dans le groupe docker.

**Solution** :
```bash
# En root
sudo usermod -aG docker stagiaire
# Puis se déconnecter / reconnecter
exit
ssh stagiaire@72.62.16.22
docker ps  # devrait marcher
```

## Les modifications de code ne sont pas prises en compte

**Symptôme** : tu modifies un fichier Python mais le comportement ne change pas.

**Cause** : le conteneur utilise l'ancienne image.

**Solution** : rebuild l'image après modification :
```bash
docker-compose up -d --build <service>
# ou pour forcer un rebuild complet
docker-compose build --no-cache <service>
docker-compose up -d <service>
```

## Les données Kafka disparaissent au redémarrage

**Symptôme** : les messages publiés ne sont plus disponibles après un `docker-compose restart`.

**Cause** : le volume Kafka n'est pas persistant, ou le topic a été recréé.

**Solution** : ne pas faire `docker-compose down -v` (le `-v` supprime les volumes). Utiliser simplement `docker-compose stop` puis `docker-compose start`.

## Git refuse de push à cause de gros fichiers

**Symptôme** : `git push` échoue avec "file exceeds GitHub's 100 MB limit".

**Cause** : un fichier de données ou un modèle a été commité par erreur.

**Solution** :
1. Vérifier que `.gitignore` est correct (les dossiers `data/`, `models/`, `*.csv`, `*.pkl` doivent y être).
2. Retirer le fichier du dernier commit :
   ```bash
   git rm --cached data/raw/train_transaction.csv
   git commit --amend
   ```
3. Si le fichier est dans un commit ancien, utiliser `git filter-branch` ou `bfg-repo-cleaner`.

## Connexion SSH au VPS qui traîne ou échoue

**Symptôme** : `ssh stagiaire@72.62.16.22` met plusieurs secondes à répondre ou timeout.

**Solutions** :

1. Vérifier la connexion internet.
2. Tester avec verbose : `ssh -v stagiaire@72.62.16.22`
3. Vérifier côté VPS que le serveur SSH tourne : sur le panel Hostinger, vérifier que le VPS est bien "Running".

## VS Code Remote-SSH : "Could not establish connection"

**Symptôme** : l'extension VS Code Remote-SSH n'arrive pas à se connecter.

**Solutions** :

1. Tester d'abord la connexion SSH en ligne de commande pour valider que l'accès marche.
2. Vérifier `~/.ssh/config` :
   ```
   Host vps-fraude
       HostName 72.62.16.22
       User stagiaire
       IdentityFile ~/.ssh/id_rsa
   ```
3. Dans VS Code, utiliser le host configuré : `vps-fraude` au lieu de `stagiaire@72.62.16.22`.

## Le test de connectivité ne passe pas tous les services

Relance le test plusieurs fois à 30 secondes d'intervalle. Certains services (Kafka surtout) ont besoin de temps pour être prêts.

Si un service reste indisponible, regarde ses logs :
```bash
docker-compose logs <service> | tail -50
```
