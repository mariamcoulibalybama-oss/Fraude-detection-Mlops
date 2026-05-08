import joblib
import json
import shap
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # pas d'écran sur le VPS
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data/raw/ieee-fraud-detection/train_transaction.csv"

print("📦 Chargement du pipeline...")
pipeline = joblib.load(BASE_DIR / "models/pipeline_v1.pkl")
model = pipeline["model"]
feature_names = pipeline["feature_names"]

print("📦 Chargement d'un échantillon de données...")
df = pd.read_csv(DATA_PATH, nrows=500)
df = df.select_dtypes(include=["number"])
df = df.loc[:, df.isnull().mean() < 0.5]
df = df.fillna(0)

X = df.drop(columns=["isFraud"])
X = X.reindex(columns=feature_names, fill_value=0)

print("🔍 Calcul des valeurs SHAP...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ── Top 10 features les plus importantes ──────────────────
print("\n📊 Top 10 features les plus importantes :")
shap_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": abs(shap_values).mean(axis=0)
}).sort_values("importance", ascending=False)

for i, row in shap_importance.head(10).iterrows():
    print(f"   {row['feature']:<20} {row['importance']:.4f}")

# ── Graphique summary plot ─────────────────────────────────
print("\n📊 Génération du graphique SHAP...")
output_dir = BASE_DIR / "reports"
output_dir.mkdir(exist_ok=True)

plt.figure()
shap.summary_plot(
    shap_values, X,
    max_display=15,
    show=False
)
plt.tight_layout()
plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()

print("✅ Graphique sauvegardé : reports/shap_summary.png")
