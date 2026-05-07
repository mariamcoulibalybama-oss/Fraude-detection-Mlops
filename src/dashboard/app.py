import streamlit as st
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import time

# ── Config page ────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# ── Connexion MongoDB ───────────────────────────────────────
@st.cache_resource
def get_collection():
    client = MongoClient("mongodb://admin:changeme@localhost:27017/")
    return client["fraude_db"]["predictions"]

collection = get_collection()

# ── Titre ──────────────────────────────────────────────────
st.title("🔍 Fraud Detection — Dashboard Temps Réel")
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")

# ── Chargement données ─────────────────────────────────────
def load_data():
    docs = list(collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(200))
    return pd.DataFrame(docs) if docs else pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("Aucune transaction reçue. Lance le producer et le consumer.")
    st.stop()

# ── KPIs ───────────────────────────────────────────────────
total      = len(df)
nb_fraud   = df["is_fraud"].sum()
taux_fraud = nb_fraud / total * 100
nb_drift   = (df["drift_status"] == "DRIFT").sum()
score_moy  = df["score"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("📨 Transactions", total)
col2.metric("🚨 Fraudes détectées", int(nb_fraud), f"{taux_fraud:.1f}%")
col3.metric("⚠️ Transactions en drift", int(nb_drift))
col4.metric("📊 Score moyen", f"{score_moy:.3f}")

st.divider()

# ── Graphique scores ───────────────────────────────────────
st.subheader("📈 Distribution des scores de fraude")
import numpy as np
counts, edges = np.histogram(df["score"], bins=20)
labels = [f"{e:.2f}" for e in edges[:-1]]
chart_df = pd.DataFrame({"score": labels, "count": counts})
st.bar_chart(chart_df.set_index("score"))
st.divider()

# ── Alertes fraude ─────────────────────────────────────────
st.subheader("🚨 Dernières alertes de fraude")
fraudes = df[df["is_fraud"] == True][["transaction_id", "amount", "score", "drift_status", "timestamp"]]
if fraudes.empty:
    st.info("Aucune fraude détectée pour le moment.")
else:
    st.dataframe(fraudes, use_container_width=True)

st.divider()

# ── Toutes les transactions ────────────────────────────────
st.subheader("📋 Dernières transactions")
st.dataframe(
    df[["transaction_id", "amount", "score", "is_fraud", "drift_status", "timestamp"]],
    use_container_width=True
)

# ── Auto-refresh toutes les 5 secondes ────────────────────
time.sleep(5)
st.rerun()
