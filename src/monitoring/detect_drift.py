"""
Détection de drift sur les scores de prédiction
Compare scores période NORMAL vs période DRIFT
"""
import pandas as pd
from pymongo import MongoClient
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# ── Chargement MongoDB ─────────────────────────────────────
print("📦 Chargement des données depuis MongoDB...")
client = MongoClient("mongodb://admin:changeme@localhost:27017/")
collection = client["fraude_db"]["predictions"]
docs = list(collection.find({}, {"_id": 0}))

if len(docs) < 10:
    print("❌ Pas assez de données. Lance le consumer d'abord.")
    exit()

df = pd.DataFrame(docs)
print(f"   {len(df)} transactions chargées")

# ── Séparer NORMAL vs DRIFT ────────────────────────────────
df_normal = df[df["drift_status"] == "NORMAL"]["score"]
df_drift  = df[df["drift_status"] == "DRIFT"]["score"]

print(f"   NORMAL : {len(df_normal)} transactions")
print(f"   DRIFT  : {len(df_drift)} transactions")

if len(df_normal) < 5 or len(df_drift) < 5:
    print("❌ Pas assez de données dans chaque groupe.")
    exit()

# ── Analyse statistique ────────────────────────────────────
print("\n📊 Comparaison des scores :")
print(f"   Score moyen NORMAL : {df_normal.mean():.3f}")
print(f"   Score moyen DRIFT  : {df_drift.mean():.3f}")
print(f"   Score max NORMAL   : {df_normal.max():.3f}")
print(f"   Score max DRIFT    : {df_drift.max():.3f}")
print(f"   Taux fraude NORMAL : {(df[df['drift_status']=='NORMAL']['is_fraud'].sum() / len(df_normal) * 100):.1f}%")
print(f"   Taux fraude DRIFT  : {(df[df['drift_status']=='DRIFT']['is_fraud'].sum() / len(df_drift) * 100):.1f}%")

# ── Test statistique KS ────────────────────────────────────
from scipy import stats
ks_stat, p_value = stats.ks_2samp(df_normal, df_drift)
print(f"\n🔬 Test de Kolmogorov-Smirnov :")
print(f"   KS statistic : {ks_stat:.3f}")
print(f"   p-value      : {p_value:.4f}")

if p_value < 0.05:
    print("   ⚠️  DRIFT DÉTECTÉ — les distributions sont significativement différentes")
else:
    print("   ✅ Pas de drift significatif détecté")

# ── Rapport HTML ───────────────────────────────────────────
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report

df_ref_ev = pd.DataFrame({"score": df_normal.values})
df_cur_ev = pd.DataFrame({"score": df_drift.values})

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_ref_ev, current_data=df_cur_ev)

# ── Rapport HTML ───────────────────────────────────────────
output_path = BASE_DIR / "reports/drift_report.html"
output_path.parent.mkdir(exist_ok=True)

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Drift Report</title>
</head>
<body style="font-family:Arial; padding:40px; background:#111; color:#eee;">
<h1>Rapport de Drift - Fraud Detection</h1>
<h2>Resultats</h2>
<table border=1 cellpadding=10 style="border-collapse:collapse;">
  <tr><th>Metrique</th><th>NORMAL</th><th>DRIFT</th></tr>
  <tr><td>Nombre transactions</td><td>{len(df_normal)}</td><td>{len(df_drift)}</td></tr>
  <tr><td>Score moyen</td><td>{df_normal.mean():.3f}</td><td>{df_drift.mean():.3f}</td></tr>
  <tr><td>Score max</td><td>{df_normal.max():.3f}</td><td>{df_drift.max():.3f}</td></tr>
  <tr><td>Taux fraude</td>
      <td>{df[df['drift_status']=='NORMAL']['is_fraud'].mean()*100:.1f}%</td>
      <td>{df[df['drift_status']=='DRIFT']['is_fraud'].mean()*100:.1f}%</td></tr>
</table>
<h2>Test statistique (Kolmogorov-Smirnov)</h2>
<p>KS statistic : <b>{ks_stat:.3f}</b></p>
<p>p-value : <b>{p_value:.4f}</b></p>
<p style="color:{'#ff4444' if p_value < 0.05 else '#44ff44'}; font-size:20px;">
  {'DRIFT DETECTE' if p_value < 0.05 else 'Pas de drift'}
</p>
</body>
</html>
"""
with open(output_path, "w") as f:
    f.write(html)

print(f"✅ Rapport HTML sauvegardé : reports/drift_report.html")
