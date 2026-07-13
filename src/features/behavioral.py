"""
Feature store comportemental — calcul des agregats temps reel via Redis.
Utilise le temps SIMULE (TransactionDT), pas l'heure systeme, car le
producer rejoue des donnees historiques en accelere.
"""
import os
import redis

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    password=os.getenv("REDIS_PASSWORD", ""),
    decode_responses=True
)


def get_behavioral_features(row):
    """
    Calcule 4 features comportementales pour une transaction :
    - nb_tx_1h : nombre de transactions de la carte dans la derniere heure
    - montant_cumule_1h : somme des montants sur cette fenetre
    - temps_depuis_derniere_tx : delai depuis la transaction precedente
    - ecart_montant_vs_moyenne_carte : ecart relatif vs moyenne historique
      (moyenne calculee AVANT mise a jour — pas de leakage)
    """
    card = str(row.get("card1", "unknown"))
    montant = float(row.get("TransactionAmt", 0))
    event_dt = float(row.get("TransactionDT", 0))

    window_key = f"card:{card}:window"
    last_ts_key = f"card:{card}:last_ts"
    hist_sum_key = f"card:{card}:hist_sum"
    hist_count_key = f"card:{card}:hist_count"

    # ── 1 & 2. Fenetre glissante 1h (sorted set, score = temps simule) ──
    member = f"{event_dt}_{montant}_{redis_client.incr('global_counter')}"
    redis_client.zadd(window_key, {member: event_dt})
    redis_client.zremrangebyscore(window_key, "-inf", event_dt - 3600)
    redis_client.expire(window_key, 86400)

    members = redis_client.zrange(window_key, 0, -1)
    nb_tx_1h = len(members)
    montant_cumule_1h = sum(float(m.split("_")[1]) for m in members)

    # ── 3. Delai depuis la derniere transaction ──
    last_ts = redis_client.get(last_ts_key)
    temps_depuis_derniere_tx = (event_dt - float(last_ts)) if last_ts else -1.0
    redis_client.set(last_ts_key, event_dt, ex=86400)

    # ── 4. Ecart vs moyenne historique (calcule AVANT mise a jour) ──
    hist_sum = float(redis_client.get(hist_sum_key) or 0)
    hist_count = int(redis_client.get(hist_count_key) or 0)
    moyenne = (hist_sum / hist_count) if hist_count > 0 else montant
    ecart_montant_vs_moyenne_carte = (montant - moyenne) / moyenne if moyenne > 0 else 0.0

    redis_client.incrbyfloat(hist_sum_key, montant)
    redis_client.incr(hist_count_key)
    redis_client.expire(hist_sum_key, 604800)
    redis_client.expire(hist_count_key, 604800)

    return {
        "nb_tx_1h": nb_tx_1h,
        "montant_cumule_1h": montant_cumule_1h,
        "temps_depuis_derniere_tx": temps_depuis_derniere_tx,
        "ecart_montant_vs_moyenne_carte": ecart_montant_vs_moyenne_carte,
    }
