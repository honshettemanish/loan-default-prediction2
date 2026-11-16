import json, math
import requests
import streamlit as st

st.set_page_config(page_title="Loan Default Predictor", page_icon="💳", layout="centered")

with open("model.json") as f:
    M = json.load(f)

def score(v, threshold):
    z = M["intercept"]; contrib = {}
    for i, name in enumerate(M["features"]):
        c = M["coef"][i] * (v[name] - M["mean"][i]) / M["std"][i]
        contrib[name] = c; z += c
    p = 1 / (1 + math.exp(-z))
    return p, int(p >= threshold), contrib

st.title("💳 Loan Default Prediction")
st.caption("Change any financial feature in the sidebar and watch the predicted class update live.")

st.sidebar.header("Borrower features")
v = {
    "loan_amount": st.sidebar.slider("Loan amount ($)", 1000, 40000, 15000, 500),
    "annual_income": st.sidebar.slider("Annual income ($)", 10000, 250000, 60000, 1000),
    "credit_score": st.sidebar.slider("Credit score", 300, 850, 690),
    "interest_rate": st.sidebar.slider("Interest rate (%)", 5.0, 24.0, 12.0, 0.5),
    "dti_ratio": st.sidebar.slider("Debt-to-income ratio", 0.01, 0.90, 0.25, 0.01),
    "employment_years": st.sidebar.slider("Employment years", 0.0, 30.0, 5.0, 0.5),
    "late_payments": st.sidebar.slider("Late payments (count)", 0, 10, 0),
    "credit_utilization": st.sidebar.slider("Credit utilization", 0.0, 1.0, 0.35, 0.01),
}
threshold = st.sidebar.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)

st.sidebar.divider()
use_n8n = st.sidebar.checkbox("Use n8n webhook for prediction")
n8n_url = st.sidebar.text_input("n8n webhook URL", value=st.secrets.get("N8N_WEBHOOK_URL", "")) if use_n8n else ""

p, cls, contrib = score(v, threshold)
source = "built-in model"
if use_n8n and n8n_url:
    try:
        r = requests.get(n8n_url, params={**v, "threshold": threshold, "format": "json"}, timeout=15)
        r.raise_for_status(); d = r.json()
        p, cls = d["default_probability"], d["predicted_class"]; source = "n8n workflow"
    except Exception as e:
        st.warning(f"n8n call failed ({e}); using built-in model instead.")

if cls == 1:
    st.error(f"### DEFAULT (High Risk) — class 1")
else:
    st.success(f"### NO DEFAULT (Low Risk) — class 0")
c1, c2 = st.columns(2)
c1.metric("Default probability", f"{p*100:.1f}%")
c2.metric("Threshold", f"{threshold:.2f}")
st.progress(min(max(p, 0.0), 1.0))
st.caption(f"Prediction served by: {source}")

st.subheader("Why? Feature impact (log-odds)")
st.bar_chart({k: v_ for k, v_ in sorted(contrib.items(), key=lambda x: x[1])})
st.caption("Positive bars push toward default, negative bars toward repayment.")
