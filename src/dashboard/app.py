import streamlit as st
import pandas as pd
import numpy as np
import json
from pymongo import MongoClient
from datetime import datetime
from pathlib import Path
import time

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[2]
DRIFT_FLAG = BASE_DIR / "config/drift_flag.json"

@st.cache_resource
def get_collection():
    client = MongoClient("mongodb://admin:changeme@localhost:27017/")
    return client["fraude_db"]["predictions"]

collection = get_collection()

def get_drift_status():
    try:
        with open(DRIFT_FLAG) as f:
            return json.load(f).get("drift", False)
    except:
        return False

def set_drift(value: bool):
    with open(DRIFT_FLAG, "w") as f:
        json.dump({"drift": value}, f)

# ── Titre ──────────────────────────────────────────────────
st.title("🔍 Fraud Detection — Dashboard Temps Reel")
st.caption(f"Derniere mise a jour : {datetime.now().strftime('%H:%M:%S')}")

# ── Controle Drift ─────────────────────────────────────────
st.subheader("🎛️ Controle du Drift")
drift_active = get_drift_status()

col_btn1, col_btn2, col_status = st.columns([2, 2, 4])

with col_btn1:
    if st.button("⚠️ Injecter le Drift", type="primary", disabled=drift_active):
        set_drift(True)
        st.success("Drift injecte !")
        st.rerun()

with col_btn2:
    if st.button("✅ Stopper le Drift", disabled=not drift_active):
        set_drift(False)
        st.success("Drift stoppe !")
        st.rerun()

with col_status:
    if drift_active:
        st.error("🔴 DRIFT ACTIF — Montants x3 en cours")
    else:
        st.success("🟢 NORMAL — Donnees normales")

st.divider()

# ── Chargement données ─────────────────────────────────────
def load_data():
    docs = list(collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(200))
    return pd.DataFrame(docs) if docs else pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("Aucune transaction recue. Lance le producer et le consumer.")
    st.stop()

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

# ── Graphique scores ───────────────────────────────────────
st.subheader("📈 Distribution des scores de fraude")
counts, edges = np.histogram(df["score"], bins=20)
labels = [f"{e:.2f}" for e in edges[:-1]]
chart_df = pd.DataFrame({"score": labels, "count": counts})
st.bar_chart(chart_df.set_index("score"))

st.divider()

# ── Score moyen NORMAL vs DRIFT ────────────────────────────
st.subheader("⚠️ Score moyen : NORMAL vs DRIFT")
drift_compare = df.groupby("drift_status")["score"].mean().reset_index()
drift_compare.columns = ["drift_status", "score_moyen"]
st.bar_chart(drift_compare.set_index("drift_status"))

st.divider()

# ── Alertes fraude ─────────────────────────────────────────
st.subheader("🚨 Dernieres alertes de fraude")
fraudes = df[df["is_fraud"] == True][["transaction_id", "amount", "score", "drift_status", "timestamp"]]
if fraudes.empty:
    st.info("Aucune fraude detectee pour le moment.")
else:
    st.dataframe(fraudes, use_container_width=True)

st.divider()

# ── Toutes les transactions ────────────────────────────────
st.subheader("📋 Dernieres transactions")
st.dataframe(
    df[["transaction_id", "amount", "score", "is_fraud", "drift_status", "timestamp"]],
    use_container_width=True
)

time.sleep(5)
st.rerun()
