import streamlit as st
import numpy as np
import joblib
import json
import faiss
from dotenv import load_dotenv
load_dotenv()

from rag import explain_risk
from embed import embed_text

# ---------- load artifacts  ----------
lr_model = joblib.load("models/risk_lr_model.joblib")
THRESHOLD = joblib.load("models/risk_threshold.joblib")

index = faiss.read_index("data/tickets.faiss")

with open("data/tickets_metadata.json") as f:
    metadata = json.load(f)

# ---------- Page config ----------
st.set_page_config(
    page_title="InsightOps — Support Ticket Risk Analyzer",
    page_icon="🛡️",
    layout="wide",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Viewport-level constraints to prevent vertical scrolling */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        overflow: hidden !important;
        max-height: 100vh !important;
        min-height: 100vh !important;
        box-sizing: border-box !important;
    }
    
    /* Hide default empty Streamlit header bar completely */
    [data-testid="stHeader"] {
        display: none !important;
    }
    
    .main-header {
        text-align: center;
        padding: 0.1rem 0;
    }
    .main-header h1 {
        font-size: 2rem;
        margin-bottom: 0.1rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ff7675, #74b9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #a4b0be;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }
    
    /* Risk Status Banners: compact */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.5rem;
        border-radius: 6px;
        text-align: center;
        margin: 0.3rem 0;
        font-size: 0.95rem;
        font-weight: bold;
    }
    .risk-low {
        background: linear-gradient(135deg, #26de81, #20bf6b);
        color: white;
        padding: 0.5rem;
        border-radius: 6px;
        text-align: center;
        margin: 0.3rem 0;
        font-size: 0.95rem;
        font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fed330, #f7b731);
        color: #1a1a2e;
        padding: 0.5rem;
        border-radius: 6px;
        text-align: center;
        margin: 0.3rem 0;
        font-size: 0.95rem;
        font-weight: bold;
    }
    
    /* Expander boxes inside grid columns: compact design */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-radius: 6px !important;
        margin-bottom: 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15) !important;
    }
    div[data-testid="stExpander"] summary {
        padding: 4px 8px !important;
        font-size: 0.85rem !important;
    }
    
    /* Centered block-container wrapper */
    div.block-container {
        max-width: 1250px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 0.6rem !important;
        padding-bottom: 0.6rem !important;
        max-height: 100vh !important;
        box-sizing: border-box !important;
    }
    
    /* Metric container styling */
    [data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 6px !important;
        padding: 6px 12px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
        font-size: 1.25rem !important;
        white-space: nowrap !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] > div {
        font-size: 0.8rem !important;
        white-space: nowrap !important;
    }
    
    /* Reduce global streamlit element padding/spacing */
    .element-container, div.row-widget {
        margin-bottom: 0.2rem !important;
    }
    
    /* Streamlit alert margins */
    .stAlert {
        padding: 5px 10px !important;
        margin-bottom: 0.4rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
        <h2 style="margin: 0; font-weight: 700; background: linear-gradient(45deg, #ff7675, #74b9ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🛡️ InsightOps</h2>
        <span style="font-size: 0.85rem; background-color: rgba(38, 222, 129, 0.15); color: #26de81; border: 1px solid rgba(38, 222, 129, 0.3); padding: 4px 10px; border-radius: 12px; font-weight: bold;">● AI ONLINE</span>
    </div>
    <div style="color: #a4b0be; font-size: 0.95rem; margin-top: -8px; margin-bottom: 12px;">AI-powered operational risk detection for support tickets</div>
    """,
    unsafe_allow_html=True
)

# ---------- Input Grid ----------
col_input, col_button = st.columns([5, 1])
with col_input:
    ticket_text = st.text_area(
        "Describe the support issue",
        label_visibility="collapsed",
        height=68,
        placeholder="Enter ticket details... (e.g. Production server is completely down for all users.)"
    )
with col_button:
    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True) # small spacing to align button
    analyze_clicked = st.button("🔍 Analyze Risk", type="primary", use_container_width=True)
    st.caption(f"Model threshold: {THRESHOLD:.3f}")

# ---------- Analysis ----------
if analyze_clicked:
    if not ticket_text.strip():
        st.warning("⚠️ Please enter ticket content before analyzing.")
    else:
        with st.spinner("Analyzing ticket risk..."):
            # 1. Generate embedding
            embedding = np.array(embed_text(ticket_text), dtype="float32")

            # 2. Predict risk
            risk_score = lr_model.predict_proba(embedding.reshape(1, -1))[:, 1][0]
            risk_label = int(risk_score >= THRESHOLD)

            # Determine risk level and display based on THRESHOLD
            if risk_score >= THRESHOLD:
                risk_level = "HIGH"
                risk_class = "risk-high"
                risk_icon = "🔴"
            elif risk_score >= THRESHOLD * 0.5:
                risk_level = "MEDIUM"
                risk_class = "risk-medium"
                risk_icon = "🟡"
            else:
                risk_level = "LOW"
                risk_class = "risk-low"
                risk_icon = "🟢"

        # ---------- Risk Score Display ----------
        st.markdown("<h3 style='margin-top: 10px; margin-bottom: 5px;'>📊 Risk Assessment</h3>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Risk Score", f"{risk_score:.3f}")
        with col2:
            st.metric("Risk Level", risk_level)
        with col3:
            st.metric("Classification", "HIGH RISK" if risk_label else "LOW RISK")
        with col4:
            confidence = risk_score if risk_score >= THRESHOLD else (1.0 - risk_score)
            st.metric("Model Confidence", f"{confidence * 100:.1f}%")

        # Color-coded status banner
        if risk_level == "HIGH":
            st.error(f"{risk_icon} **HIGH RISK** — Critical operational issue requiring immediate attention.")
        elif risk_level == "MEDIUM":
            st.warning(f"{risk_icon} **MEDIUM RISK** — Elevated priority and monitoring required.")
        else:
            st.success(f"{risk_icon} **LOW RISK** — Routine support request.")

        # 1. Visual Risk Score Gauge
        pct = int(risk_score * 100)
        st.markdown(
            f"""
            <div style="background: linear-gradient(to right, #26de81 0%, #26de81 {THRESHOLD*50}%, #fed330 {THRESHOLD*50}%, #fed330 {THRESHOLD*100}%, #ff6b6b {THRESHOLD*100}%, #ff6b6b 100%); height: 10px; border-radius: 5px; position: relative; margin-top: 5px; margin-bottom: 5px;">
                <div style="position: absolute; left: calc({pct}% - 6px); top: -3px; width: 12px; height: 16px; background-color: #1a1a2e; border: 2px solid white; border-radius: 3px; box-shadow: 0 1px 3px rgba(0,0,0,0.3);"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #6c757d; font-weight: bold; margin-bottom: 10px;">
                <span style="color: #20bf6b;">LOW (0.00 - {THRESHOLD*0.5:.2f})</span>
                <span style="color: #f7b731;">MEDIUM ({THRESHOLD*0.5:.2f} - {THRESHOLD:.2f})</span>
                <span style="color: #ee5a24;">HIGH ({THRESHOLD:.2f} - 1.00)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 3. RAG explanation (for ALL risk levels now)
        rag_output = explain_risk(embedding, risk_score=risk_score, k=5)

        # ---------- Why This Risk? ----------
        st.markdown("<h3 style='margin-top: 5px; margin-bottom: 5px;'>🔍 Why This Risk?</h3>", unsafe_allow_html=True)
        
        evidence_list = []
        text_lower = ticket_text.lower()
        
        if risk_label == 1:
            if any(w in text_lower for w in ['down', 'outage', 'downtime', 'halt', 'crash', 'fatal', 'kernel panic', 'service unavailable', 'inaccessible', 'unaccessible', 'not responding']):
                evidence_list.append("⚠️ **Outage/Downtime Indicator**: System/service unavailability or crash mentioned.")
            if any(w in text_lower for w in ['production', 'prod server', 'production server']):
                evidence_list.append("⚙️ **Production Impact**: Direct impact on production environments or servers.")
            if any(w in text_lower for w in ['security', 'unauthorized', 'compromise', 'breach', 'hack', 'credential', 'stolen', 'leak', 'intrusion']):
                evidence_list.append("🔒 **Security Indicator**: Unauthorized access, compromise, or critical credential issues detected.")
            if any(w in text_lower for w in ['deducted', 'money', 'charge', 'billing', 'financial', 'invoice', 'payroll']):
                evidence_list.append("💳 **Financial Impact**: Billing discrepancies, payroll deductions, or monetary impact mentioned.")
            if any(w in text_lower for w in ['employees', 'users', 'customers', 'all users', 'multiple users', 'everyone', 'widespread']):
                evidence_list.append("👥 **Broad User Impact**: Issue explicitly references impact on multiple users or customers.")
            if not evidence_list:
                evidence_list.append("🚨 **High Operational Severity**: Semantic features strongly match high-risk historical patterns.")
        else:
            if any(w in text_lower for w in ['password reset', 'reset password', 'reset my password', 'forgot password', 'unlock account', 'account is locked', 'account locked']):
                evidence_list.append("🔑 **Password Reset Request**: Standard password/access reset query.")
            if any(w in text_lower for w in ['piv card', 'piv exemption', '1-day piv', 'rsa token', 'temporary rsa']):
                evidence_list.append("🎫 **Routine Request / PIV card**: Routine security token or card exemption.")
            if any(w in text_lower for w in ['install', 'installation', 'upgrade', 'license request', 'additional licenses']):
                evidence_list.append("📦 **Administrative Request**: Routine license allocation or standard software installation.")
            if any(w in text_lower for w in ['slightly slow', 'slow today', 'minor delay', 'performance is slightly']):
                evidence_list.append("🐢 **Minor Performance Issue**: Low-severity warning regarding performance slowness.")
            if not evidence_list:
                evidence_list.append("✅ **Routine Support Request**: Low operational footprint matching administrative or routine actions.")
                
        for ev in evidence_list:
            st.write(ev)
            
        confidence = risk_score if risk_score >= THRESHOLD else (1.0 - risk_score)
        st.write(f"**Model Confidence:** {confidence * 100:.1f}%")

        # ---------- Explanation ----------
        st.markdown("<h3 style='margin-top: 5px; margin-bottom: 2px;'>💡 AI Explanation & Historical Insights</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.9rem; color: #a4b0be; margin-bottom: 8px;'>Historical context: 5 similar incidents analyzed · {rag_output['high_risk_ratio']:.0%} classified as high risk.</div>", unsafe_allow_html=True)
        
        col_exp_left, col_exp_right = st.columns([1, 1])
        with col_exp_left:
            st.info(rag_output["explanation"])
        with col_exp_right:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    "Avg Resolution Time",
                    f"{rag_output['avg_resolution_time']:,.0f} min",
                )
            with m2:
                st.metric(
                    "High-Risk Ratio",
                    f"{rag_output['high_risk_ratio']:.0%}",
                )
            with m3:
                st.metric(
                    "Dominant Category",
                    f"{rag_output.get('dominant_category', 'N/A')}",
                )

# ---------- Footer ----------
st.caption("InsightOps v2.0 — Built with Streamlit, SentenceTransformers, FAISS, and Scikit-learn")
