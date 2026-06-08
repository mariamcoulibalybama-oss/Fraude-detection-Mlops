import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import time

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def get_collection():
    client = MongoClient("mongodb://admin:changeme@localhost:27017/")
    return client["fraude_db"]["predictions"]

collection = get_collection()

st.title("🔍 Fraud Detection — Dashboard Temps Reel")
st.caption(f"Derniere mise a jour : {datetime.now().strftime('%H:%M:%S')}")

def load_data():
    docs = list(collection.find({}, {"_id": 0}).sort("mois_simule", -1).limit(500))
    return pd.DataFrame(docs) if docs else pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("Aucune transaction recue. Lance le producer et le consumer.")
    st.stop()

# ── Statut actuel ──────────────────────────────────────────
phase_actuelle = df["drift_phase"].iloc[0] if "drift_phase" in df.columns else "NORMAL"
mois_actuel = df["mois_simule"].iloc[0] if "mois_simule" in df.columns else "2025-01"

phase_colors = {
    "NORMAL": "🟢",
    "DRIFT_AMT": "🟡",
    "DRIFT_CARD": "🟠",
    "DRIFT_BEHAVIOR": "🟠",
    "DRIFT_FORT": "🔴",
    "NOUVEAU_NORMAL": "🔵"
}
phase_labels = {
    "NORMAL": "Periode stable",
    "DRIFT_AMT": "Drift phase 1 — Montants en hausse",
    "DRIFT_CARD": "Drift phase 2 — Types de cartes changent",
    "DRIFT_BEHAVIOR": "Drift phase 3 — Comportements clients derivent",
    "DRIFT_FORT": "Drift phase 4 — Drift fort detecte",
    "NOUVEAU_NORMAL": "Nouveau normal — Modele readapte"
}

emoji = phase_colors.get(phase_actuelle, "🟢")
label = phase_labels.get(phase_actuelle, "Normal")
st.subheader(f"{emoji} Statut actuel ({mois_actuel}) : {label}")

st.divider()

# ── KPIs ───────────────────────────────────────────────────
total      = len(df)
nb_fraud   = df["is_fraud"].sum()
taux_fraud = nb_fraud / total * 100
nb_drift   = (df["drift_status"] == "DRIFT").sum()
score_moy  = df["score"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("📨 Transactions", total)
col2.metric("🚨 Fraudes detectees", int(nb_fraud), f"{taux_fraud:.1f}%")
col3.metric("⚠️ En drift", int(nb_drift))
col4.metric("📊 Score moyen", f"{score_moy:.3f}")

st.divider()

# ── Tableau recap par mois ─────────────────────────────────
st.subheader("📋 Tableau de suivi mensuel 2025-2026")

if "mois_simule" in df.columns:
    recap = df.groupby(["mois_simule", "drift_phase"]).agg(
        transactions=("score", "count"),
        score_moyen=("score", "mean"),
        fraudes=("is_fraud", "sum")
    ).reset_index()

    recap = recap.sort_values("mois_simule")

    # Ajouter emoji statut
    def get_statut(phase):
        emojis = {
            "NORMAL": "🟢 Stable",
            "DRIFT_AMT": "🟡 Attention",
            "DRIFT_CARD": "🟠 Alerte",
            "DRIFT_BEHAVIOR": "🟠 Alerte",
            "DRIFT_FORT": "🔴 Critique",
            "NOUVEAU_NORMAL": "🔵 Adapte"
        }
        return emojis.get(phase, "🟢 Stable")

    recap["statut"] = recap["drift_phase"].apply(get_statut)
    recap["score_moyen"] = recap["score_moyen"].round(3)
    recap["fraudes"] = recap["fraudes"].astype(int)

    recap.columns = ["Mois", "Phase", "Transactions", "Score moyen", "Fraudes", "Statut"]
    st.dataframe(recap[["Mois", "Statut", "Score moyen", "Transactions", "Fraudes"]], 
                 use_container_width=True)

st.divider()

# ── Graphique evolution score par mois ────────────────────
st.subheader("📈 Evolution du score moyen par mois (2025-2026)")
if "mois_simule" in df.columns:
    df_mois = df.groupby("mois_simule")["score"].mean().reset_index()
    df_mois = df_mois.sort_values("mois_simule")
    df_mois.columns = ["Mois", "Score moyen"]
    st.line_chart(df_mois.set_index("Mois"))

st.divider()

# ── Distribution des scores ────────────────────────────────
st.subheader("📊 Distribution des scores de fraude")
counts, edges = np.histogram(df["score"], bins=20)
labels = [f"{e:.2f}" for e in edges[:-1]]
chart_df = pd.DataFrame({"score": labels, "count": counts})
st.bar_chart(chart_df.set_index("score"))

st.divider()

# ── Alertes fraude ─────────────────────────────────────────
st.subheader("🚨 Dernieres alertes de fraude")
fraudes = df[df["is_fraud"] == True][["transaction_id", "amount", "score", "drift_phase", "mois_simule"]]
if fraudes.empty:
    st.info("Aucune fraude detectee pour le moment.")
else:
    st.dataframe(fraudes.head(20), use_container_width=True)

st.divider()

# ── Dernieres transactions ─────────────────────────────────
st.subheader("📋 Dernieres transactions")
st.dataframe(
    df[["transaction_id", "amount", "score", "is_fraud", "drift_phase", "mois_simule"]].head(50),
    use_container_width=True
)

time.sleep(5)
st.rerun()
