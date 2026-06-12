"""
Support Integrity Auditor (SIA) — Streamlit Web Application
Simplified Version
"""
import sys, os, json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("🛡️ Support Integrity Auditor")

# --- DEBUGGING BLOCK ---
st.warning(f"Root path identified as: {C.ROOT}")
st.warning(f"Looking for predictions at: {C.PREDICTIONS_CSV}")
st.warning(f"Does predictions.csv exist? {os.path.exists(C.PREDICTIONS_CSV)}")
# -----------------------
# ── Project root on sys.path ──────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config as C
from src.model.predict import load_artifacts, SEVERITY_LABELS, METADATA_FEATURES
from src.features.feature_engg import (
    clean_text, encode_priority, encode_channel, encode_category,
    extract_email_domain, normalize_resolution_time, normalize_satisfaction,
)
from src.pseudo_labeling.template_extractor import compute_template_score
from src.dossier.dossier_generator import generate_dossier, validate_dossier

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & HELPERS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Support Integrity Auditor", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_model():
    return load_artifacts(str(C.MODELS_DIR))

@st.cache_data
def load_predictions():
    path = str(C.PREDICTIONS_CSV)
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data
def load_dossiers():
    path = str(C.DOSSIERS_JSON)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    return []

def prepare_single_ticket(ticket_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([ticket_data])
    df = clean_text(df)
    df = encode_priority(df)
    df = encode_channel(df)
    df = encode_category(df)
    df = extract_email_domain(df)
    df = normalize_resolution_time(df)
    df = normalize_satisfaction(df)
    return compute_template_score(df)

def predict_ticket(df: pd.DataFrame) -> pd.DataFrame:
    model, tfidf = load_model()
    from scipy.sparse import hstack, csr_matrix
    
    X_text = tfidf.transform(df["combined_text"])
    X_meta = csr_matrix(df[METADATA_FEATURES].values.astype(float))
    X = hstack([X_text, X_meta])

    df = df.copy()
    df["predicted_severity_encoded"] = model.predict(X)
    df["predicted_severity"] = df["predicted_severity_encoded"].map(SEVERITY_LABELS)
    df["confidence"] = model.predict_proba(X).max(axis=1).round(4)
    df["severity_delta"] = df["predicted_severity_encoded"] - df["priority_encoded"]
    df["is_mismatch"] = (df["severity_delta"] != 0).astype(int)
    df["mismatch_type"] = df["severity_delta"].apply(
        lambda d: "Hidden Crisis" if d > 0 else ("False Alarm" if d < 0 else "Consistent")
    )
    return df

# ══════════════════════════════════════════════════════════════════════════════
#  UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
st.title("🛡️ Support Integrity Auditor")
st.markdown("Semantics-driven detection of priority mismatches in support tickets")

page = st.sidebar.radio("Navigation", ["📊 Dashboard", "🔍 Single Ticket", "📁 Batch Upload", "📋 Dossier Explorer"])

if page == "📊 Dashboard":
    df = load_predictions()
    dossiers = load_dossiers()

    if df is None:
        st.error("No predictions found. Run the prediction pipeline first.")
        st.stop()

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Tickets", f"{len(df):,}")
    k2.metric("Mismatches", f"{df['is_mismatch'].sum():,}")
    k3.metric("Mismatch Rate", f"{df['is_mismatch'].mean():.1%}")
    k4.metric("Hidden Crises", f"{(df['mismatch_type'] == 'Hidden Crisis').sum():,}")
    k5.metric("False Alarms", f"{(df['mismatch_type'] == 'False Alarm').sum():,}")
    st.divider()

    # Charts Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Mismatch Type Distribution")
        mm_counts = df["mismatch_type"].value_counts()
        fig = go.Figure(go.Pie(labels=mm_counts.index, values=mm_counts.values, hole=0.5))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Predicted Severity Distribution")
        sev_order = ["Low", "Medium", "High", "Critical"]
        sev_counts = df["predicted_severity"].value_counts().reindex(sev_order, fill_value=0)
        fig = go.Figure(go.Bar(x=sev_order, y=sev_counts.values, text=sev_counts.values))
        st.plotly_chart(fig, use_container_width=True)

    # Charts Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Severity Delta: By Category")
        ct_cat = pd.crosstab(df[C.COL_CATEGORY], df["mismatch_type"])[["Consistent", "Hidden Crisis", "False Alarm"]]
        fig = go.Figure(go.Heatmap(z=ct_cat.values, x=ct_cat.columns, y=ct_cat.index, colorscale="RdYlGn_r"))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Severity Delta: By Channel")
        ct_ch = pd.crosstab(df[C.COL_CHANNEL], df["mismatch_type"])[["Consistent", "Hidden Crisis", "False Alarm"]]
        fig = go.Figure(go.Heatmap(z=ct_ch.values, x=ct_ch.columns, y=ct_ch.index, colorscale="RdYlGn_r"))
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Single Ticket":
    st.header("Analyze a Single Ticket")
    with st.form("single_ticket_form"):
        c1, c2 = st.columns(2)
        with c1:
            t_id = st.text_input("Ticket ID", "TKT-001")
            c_name = st.text_input("Customer Name", "John Doe")
            c_email = st.text_input("Email", "john@example.com")
            subject = st.text_input("Subject", "App crashing - Urgent")
            channel = st.selectbox("Channel", ["Chat", "Email", "Web Form"])
        with c2:
            category = st.selectbox("Category", ["Technical", "Billing", "Account", "Fraud", "General Inquiry"])
            priority = st.selectbox("Assigned Priority", C.PRIORITY_LEVELS)
            res_hrs = st.number_input("Resolution Time (hrs)", 0.0, 200.0, 12.0)
            sat = st.slider("Satisfaction Score", 1, 5, 3)
            desc = st.text_area("Description", "App crashes on launch. Data lost.")

        if st.form_submit_button("Analyze Ticket"):
            ticket_data = {
                C.COL_ID: t_id, "Customer_Name": c_name, C.COL_EMAIL: c_email, C.COL_SUBJECT: subject,
                C.COL_DESC: desc, C.COL_CATEGORY: category, C.COL_PRIORITY: priority, C.COL_CHANNEL: channel,
                "Submission_Date": "2026-01-01", C.COL_RES_HRS: res_hrs, "Assigned_Agent": "A-01", C.COL_SAT: sat,
            }
            try:
                df_res = predict_ticket(prepare_single_ticket(ticket_data))
                row = df_res.iloc[0]
                
                # Results Metrics
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Assigned Priority", priority)
                r2.metric("Inferred Severity", row["predicted_severity"])
                r3.metric("Verdict", row["mismatch_type"])
                r4.metric("Confidence", f"{row['confidence']:.1%}")

                if row["is_mismatch"]:
                    st.subheader("📋 Evidence Dossier")
                    dossier = generate_dossier(row)
                    st.write(f"**Constraint Analysis:** {dossier['constraint_analysis']}")
                    for ev in dossier["feature_evidence"]:
                        st.markdown(f"- **[{ev.get('signal')}]** {ev.get('value')} → *{ev.get('weight', ev.get('interpretation'))}* (Source: {ev.get('source_field')})")
                    with st.expander("Raw JSON"): st.json(dossier)
                else:
                    st.success("✅ Consistent — Assigned priority matches inferred severity.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif page == "📁 Batch Upload":
    st.header("Batch CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file and st.button("Run Predictions"):
        df_upload = pd.read_csv(uploaded_file)
        with st.spinner("Processing..."):
            df_proc = compute_template_score(normalize_satisfaction(normalize_resolution_time(
                extract_email_domain(encode_category(encode_channel(encode_priority(clean_text(df_upload.copy())))))
            )))
            df_result = predict_ticket(df_proc)
            
            # Save results setup here...
            st.success(f"Processed {len(df_result)} tickets!")
            st.dataframe(df_result[[C.COL_ID, C.COL_SUBJECT, C.COL_PRIORITY, "predicted_severity", "mismatch_type"]])

elif page == "📋 Dossier Explorer":
    st.header("Evidence Dossier Explorer")
    dossiers = load_dossiers()
    if not dossiers:
        st.warning("No dossiers found.")
        st.stop()

    search = st.text_input("🔎 Search by Ticket ID")
    filtered = [d for d in dossiers if search.upper() in str(d["ticket_id"]).upper()] if search else dossiers

    for d in filtered[:20]: # Showing first 20 for brevity
        with st.expander(f"{d['ticket_id']} - {d['mismatch_type']} (Δ: {d['severity_delta']})"):
            st.write(f"**Assigned:** {d['assigned_priority']} | **Inferred:** {d['inferred_severity']} | **Confidence:** {d['confidence']:.1%}")
            st.write(f"_{d['constraint_analysis']}_")
            for ev in d["feature_evidence"]:
                st.markdown(f"- **[{ev.get('signal')}]** {ev.get('value')} → *{ev.get('weight', ev.get('interpretation'))}*")