import streamlit as st
import pickle
import pandas as pd
from collections import Counter

# =========================
# LOAD MODEL
# =========================

saved = pickle.load(open("model/ckd_model.pkl", "rb"))

model = saved["model"]
feature_names = saved["features"]

# =========================
# UI
# =========================

st.set_page_config(
    page_title="CKD Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#6C63FF"
SECONDARY = "#00C9A7"
BG = "#F7F9FC"
CARD_BG = "#FFFFFF"
ALERT_RED = "#FF6B6B"
SUCCESS_GREEN = "#00C853"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --primary: {PRIMARY};
  --secondary: {SECONDARY};
  --bg: {BG};
  --card: {CARD_BG};
  --danger: {ALERT_RED};
  --success: {SUCCESS_GREEN};
  --text: #0F172A;
  --muted: #64748B;
  --border: rgba(15, 23, 42, 0.10);
  --shadow: 0 16px 40px rgba(2, 6, 23, 0.10);
  --shadow-soft: 0 12px 28px rgba(2, 6, 23, 0.08);
}}

html, body, [class*="css"] {{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
}}

.stApp {{
  background: radial-gradient(1200px 600px at 12% 0%, rgba(108, 99, 255, 0.18), transparent 55%),
              radial-gradient(900px 540px at 95% 10%, rgba(0, 201, 167, 0.14), transparent 55%),
              linear-gradient(180deg, var(--bg), #ffffff);
}}

/* Centered max-width container */
.app-container {{
  max-width: 1200px;
  margin: 0 auto;
  padding: 18px 10px 28px 10px;
  position: relative;
}}

/* Floating blur shapes */
.blur-shape {{
  position: absolute;
  border-radius: 999px;
  filter: blur(40px);
  opacity: 0.55;
  pointer-events: none;
}}
.blur-1 {{
  width: 260px; height: 260px;
  top: -70px; left: -60px;
  background: rgba(108, 99, 255, 0.55);
}}
.blur-2 {{
  width: 220px; height: 220px;
  top: 30px; right: -40px;
  background: rgba(0, 201, 167, 0.45);
}}
.blur-3 {{
  width: 200px; height: 200px;
  bottom: -80px; left: 25%;
  background: rgba(108, 99, 255, 0.18);
}}

/* Navbar */
.topbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 18px;
  border-radius: 18px;
  background: rgba(255,255,255,0.72);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid rgba(255,255,255,0.7);
  box-shadow: var(--shadow-soft);
}}
.brand {{
  display: flex; align-items: center; gap: 12px;
}}
.logo {{
  width: 44px; height: 44px;
  border-radius: 14px;
  display: grid; place-items: center;
  background: linear-gradient(135deg, rgba(108,99,255,1), rgba(0,201,167,1));
  box-shadow: 0 12px 26px rgba(108, 99, 255, 0.25);
  color: white;
  font-size: 22px;
}}
.brand-title {{
  line-height: 1.15;
}}
.brand-title h1 {{
  margin: 0;
  font-size: 20px;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.3px;
}}
.brand-title p {{
  margin: 4px 0 0 0;
  color: var(--muted);
  font-size: 12.5px;
  font-weight: 500;
}}
.chip {{
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(108, 99, 255, 0.10);
  border: 1px solid rgba(108, 99, 255, 0.20);
  color: rgba(108, 99, 255, 0.95);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}}

/* Sidebar glass card */
section[data-testid="stSidebar"] > div {{
  background: transparent !important;
}}
section[data-testid="stSidebar"] {{
  border-right: 0 !important;
}}
section[data-testid="stSidebar"] .stSidebarContent {{
  padding: 18px 14px 24px 14px;
}}
.sidebar-card {{
  background: rgba(255,255,255,0.62);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,0.65);
  border-radius: 18px;
  box-shadow: var(--shadow-soft);
  padding: 14px 14px 6px 14px;
}}
.sidebar-title {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}}
.sidebar-title h3 {{
  margin: 0;
  font-size: 14px;
  font-weight: 800;
  color: var(--text);
}}
.sidebar-title span {{
  font-size: 12px;
  color: var(--muted);
  font-weight: 600;
}}

/* Cards */
.card {{
  background: rgba(255,255,255,0.85);
  border: 1px solid rgba(255,255,255,0.7);
  border-radius: 18px;
  box-shadow: var(--shadow-soft);
  padding: 20px 20px 16px 20px;
  transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}}
.card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 18px 44px rgba(2,6,23,0.12);
  border-color: rgba(108,99,255,0.18);
}}
.card h2 {{
  margin: 0 0 10px 0;
  font-size: 14px;
  letter-spacing: -0.2px;
  color: var(--text);
}}
.card-title {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}}
.card-title .left {{
  display: flex;
  align-items: center;
  gap: 10px;
}}
.card-title .icon {{
  width: 34px;
  height: 34px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, rgba(108,99,255,0.16), rgba(0,201,167,0.14));
  border: 1px solid rgba(108,99,255,0.18);
}}
.subtle {{
  color: var(--muted);
  font-size: 12.5px;
  font-weight: 500;
}}
.fade-in {{
  animation: fadeInUp 520ms ease-out both;
}}
@keyframes fadeInUp {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to {{ opacity: 1; transform: translateY(0px); }}
}}

/* Field labels + inputs */
.field-label {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(255,255,255,0.70);
  border: 1px solid rgba(15,23,42,0.08);
  box-shadow: 0 10px 20px rgba(2, 6, 23, 0.05);
  margin-bottom: 8px;
}}
.field-icon {{
  width: 34px; height: 34px;
  border-radius: 12px;
  display: grid; place-items: center;
  background: linear-gradient(135deg, rgba(108,99,255,0.16), rgba(0,201,167,0.14));
  border: 1px solid rgba(108,99,255,0.18);
}}
.field-text {{
  line-height: 1.05;
}}
.field-text b {{
  display: block;
  color: var(--text);
  font-size: 12.5px;
  letter-spacing: -0.1px;
}}
.field-text span {{
  display: block;
  color: var(--muted);
  font-size: 11.5px;
  margin-top: 3px;
}}

/* Per-field labels (bold, professional, with icons) */
.input-label {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 12px 0 6px 2px;
  font-size: 12.5px;
  font-weight: 800;
  color: var(--text);
}}
.input-label .i {{
  width: 18px;
  height: 18px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  background: rgba(108,99,255,0.10);
  border: 1px solid rgba(108,99,255,0.18);
  font-size: 12px;
}}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {{
  border-radius: 14px !important;
  border: 1px solid rgba(15,23,42,0.12) !important;
  box-shadow: 0 12px 26px rgba(2,6,23,0.06) !important;
  padding: 12px 12px !important;
  transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease !important;
}}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {{
  border-color: rgba(108,99,255,0.55) !important;
  box-shadow: 0 14px 34px rgba(108,99,255,0.18) !important;
  transform: translateY(-1px);
}}

div[data-testid="stSlider"] > div {{
  padding-top: 6px;
}}

/* Predict button */
div.stButton > button {{
  width: 100%;
  border: 0 !important;
  border-radius: 999px !important;
  padding: 12px 16px !important;
  font-weight: 800 !important;
  letter-spacing: 0.1px !important;
  color: #fff !important;
  background: linear-gradient(135deg, rgba(108,99,255,1), rgba(0,201,167,1)) !important;
  box-shadow: 0 16px 34px rgba(108,99,255,0.20) !important;
  transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease !important;
}}
div.stButton > button:hover {{
  transform: translateY(-2px);
  box-shadow: 0 22px 48px rgba(108,99,255,0.26) !important;
  filter: brightness(1.02);
}}

/* Result cards */
.result-card {{
  border-radius: 18px;
  padding: 18px 18px;
  border: 1px solid rgba(255,255,255,0.7);
  background: rgba(255,255,255,0.80);
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
  transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}}
.result-card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 22px 60px rgba(2,6,23,0.14), var(--shadow);
}}
.result-card::before {{
  content: "";
  position: absolute;
  inset: -120px;
  background: radial-gradient(circle at 20% 10%, rgba(108,99,255,0.30), transparent 35%),
              radial-gradient(circle at 80% 20%, rgba(0,201,167,0.22), transparent 40%);
  filter: blur(28px);
  opacity: 0.9;
}}
.result-card > div {{
  position: relative;
  z-index: 1;
}}
.result-title {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}}
.result-title h2 {{
  margin: 0;
  font-size: 16px;
  font-weight: 900;
  letter-spacing: -0.3px;
}}
.badge {{
  padding: 8px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0.2px;
}}
.badge-healthy {{
  color: rgba(0, 122, 71, 1);
  background: rgba(0, 200, 83, 0.12);
  border: 1px solid rgba(0, 200, 83, 0.22);
}}
.badge-ckd {{
  color: rgba(158, 30, 30, 1);
  background: rgba(255, 107, 107, 0.13);
  border: 1px solid rgba(255, 107, 107, 0.26);
}}
.glow-healthy {{
  box-shadow: 0 22px 52px rgba(0, 200, 83, 0.18), var(--shadow);
}}
.glow-ckd {{
  box-shadow: 0 22px 52px rgba(255, 107, 107, 0.18), var(--shadow);
}}
.result-main {{
  margin-top: 10px;
  display: flex;
  align-items: center;
  gap: 12px;
}}
.result-icon {{
  width: 44px; height: 44px;
  border-radius: 16px;
  display: grid; place-items: center;
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(15,23,42,0.08);
  box-shadow: 0 16px 34px rgba(2,6,23,0.08);
  font-size: 20px;
}}
.result-text {{
  line-height: 1.15;
}}
.result-text b {{
  display: block;
  font-size: 18px;
  letter-spacing: -0.2px;
  color: var(--text);
}}
.result-text span {{
  display: block;
  margin-top: 4px;
  font-size: 12.5px;
  color: var(--muted);
}}

/* Stat cards */
.stats {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
}}
.stat {{
  border-radius: 18px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.70);
  border: 1px solid rgba(15,23,42,0.08);
  box-shadow: 0 14px 30px rgba(2,6,23,0.06);
  transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}}
.stat:hover {{
  transform: translateY(-2px);
  box-shadow: 0 18px 44px rgba(2,6,23,0.10);
  border-color: rgba(108,99,255,0.16);
}}
.stat-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}}
.stat-badge {{
  width: 34px;
  height: 34px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  font-size: 16px;
  border: 1px solid rgba(15,23,42,0.06);
}}
.stat-red {{
  background: rgba(255, 107, 107, 0.12);
  border-color: rgba(255, 107, 107, 0.26);
}}
.stat-green {{
  background: rgba(0, 200, 83, 0.12);
  border-color: rgba(0, 200, 83, 0.22);
}}
.v-red {{ color: rgba(158, 30, 30, 1); }}
.v-green {{ color: rgba(0, 122, 71, 1); }}
}}
.stat .k {{
  color: var(--muted);
  font-size: 11.5px;
  font-weight: 700;
}}
.stat .v {{
  margin-top: 6px;
  font-size: 20px;
  font-weight: 900;
  color: var(--text);
  letter-spacing: -0.3px;
}}
.stat .hint {{
  margin-top: 4px;
  font-size: 11.5px;
  color: var(--muted);
}}

/* Percent rings */
.rings {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
}}
.ring-card {{
  border-radius: 18px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.70);
  border: 1px solid rgba(15,23,42,0.08);
  box-shadow: 0 14px 30px rgba(2,6,23,0.06);
  display: flex;
  align-items: center;
  gap: 12px;
  transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}}
.ring-card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 18px 44px rgba(2,6,23,0.10);
  border-color: rgba(108,99,255,0.16);
}}
.ring {{
  --p: 0;
  --c: rgba(108,99,255,1);
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(15,23,42,0.08) 0);
  display: grid;
  place-items: center;
  box-shadow: 0 16px 30px rgba(2,6,23,0.08);
}}
.ring::after {{
  content: "";
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(15,23,42,0.06);
}}
.ring-label {{
  position: absolute;
  font-size: 12px;
  font-weight: 900;
  color: var(--text);
}}
.ring-wrap {{
  position: relative;
  width: 64px;
  height: 64px;
}}
.ring-text b {{
  display: block;
  font-size: 12.5px;
  color: var(--text);
}}
.ring-text span {{
  display: block;
  margin-top: 4px;
  font-size: 12px;
  color: var(--muted);
}}

/* Footer */
.footer {{
  margin-top: 18px;
  text-align: center;
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
}}

/* Probability progress bars (custom, colored, rounded, text inside) */
.prob-wrap {{
  margin-top: 10px;
  display: grid;
  gap: 10px;
}}
.prob-row {{
  display: grid;
  gap: 6px;
}}
.prob-label {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}}
.prob-bar {{
  height: 16px;
  border-radius: 999px;
  background: rgba(15,23,42,0.08);
  border: 1px solid rgba(15,23,42,0.08);
  overflow: hidden;
  box-shadow: 0 10px 22px rgba(2,6,23,0.06);
}}
.prob-fill {{
  height: 100%;
  width: 0%;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 10px;
  color: white;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.1px;
  animation: fill 700ms ease-out forwards;
}}
.prob-fill.red {{
  background: linear-gradient(90deg, rgba(255,107,107,1), rgba(255,107,107,0.85));
}}
.prob-fill.green {{
  background: linear-gradient(90deg, rgba(0,200,83,1), rgba(0,200,83,0.80));
}}
@keyframes fill {{
  from {{ width: 0%; }}
  to {{ width: var(--w); }}
}}

/* Reduce default Streamlit padding, keep breathing room */
.block-container {{
  padding-top: 1.0rem;
  padding-bottom: 1.0rem;
}}
</style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="app-container">
  <div class="blur-shape blur-1"></div>
  <div class="blur-shape blur-2"></div>
  <div class="blur-shape blur-3"></div>

  <div class="topbar fade-in">
    <div class="brand">
      <div class="logo">⚕</div>
      <div class="brand-title">
        <h1>Chronic Kidney Disease Prediction</h1>
        <p>AI Powered Medical Diagnosis System</p>
      </div>
    </div>
    <div class="chip">Medical AI Dashboard</div>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
<div class="sidebar-card fade-in">
  <div class="sidebar-title">
    <h3>Patient Intake</h3>
    <span>Inputs</span>
  </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="field-label">
  <div class="field-icon">👤</div>
  <div class="field-text"><b>Basic Info</b><span>Demographics & vitals</span></div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="input-label"><span class="i">🎂</span>Age</div>', unsafe_allow_html=True)
    age = st.number_input("Age", 1, 100)
    st.markdown('<div class="input-label"><span class="i">💓</span>Blood Pressure</div>', unsafe_allow_html=True)
    bp = st.number_input("Blood Pressure", 40, 200)

    st.markdown(
        """
<div class="field-label">
  <div class="field-icon">🧪</div>
  <div class="field-text"><b>Urine Analysis</b><span>Kidney filtration markers</span></div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="input-label"><span class="i">⚗️</span>Specific Gravity</div>', unsafe_allow_html=True)
    sg = st.number_input("Specific Gravity", 1.005, 1.025)
    st.markdown('<div class="input-label"><span class="i">🧫</span>Albumin</div>', unsafe_allow_html=True)
    al = st.slider("Albumin", 0, 5)
    st.markdown('<div class="input-label"><span class="i">🍬</span>Sugar</div>', unsafe_allow_html=True)
    su = st.slider("Sugar", 0, 5)

    st.markdown(
        """
<div class="field-label">
  <div class="field-icon">🩸</div>
  <div class="field-text"><b>Blood Metrics</b><span>Glucose level</span></div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="input-label"><span class="i">🩸</span>Blood Glucose</div>', unsafe_allow_html=True)
    bgr = st.number_input("Blood Glucose", 50, 500)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="app-container">', unsafe_allow_html=True)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown(
        """
<div class="card fade-in">
  <div class="card-title">
    <div class="left">
      <div class="icon">🧾</div>
      <h2>Patient Information</h2>
    </div>
  </div>
  <div class="subtle">Enter values in the sidebar. Results and analytics render here after prediction.</div>
</div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
<div class="card fade-in">
  <div class="card-title">
    <div class="left">
      <div class="icon">🤖</div>
      <h2>Prediction Controls</h2>
    </div>
  </div>
  <div class="subtle">Run inference and view a clear voting vs probability breakdown.</div>
</div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# INPUT DATAFRAME
# =========================

input_df = pd.DataFrame([{
    "age": age,
    "bp": bp,
    "sg": sg,
    "al": al,
    "su": su,
    "bgr": bgr
}])

# =========================
# PREDICT BUTTON
# =========================

with right:
    predict_clicked = st.button("Predict", use_container_width=True)

if predict_clicked:

    # FINAL PREDICTION
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)

    # =========================
    # TREE VOTING LOGIC
    # =========================

    tree_preds = []

    for tree in model.estimators_:
        pred = tree.predict(input_df)[0]
        tree_preds.append(pred)

    votes = Counter(tree_preds)

    ckd_votes = votes.get(0, 0)      # class 0 = CKD
    healthy_votes = votes.get(1, 0)  # class 1 = Healthy

    # =========================
    # RESULT DISPLAY
    # =========================

    ckd_prob = float(prob[0][0]) * 100.0
    healthy_prob = float(prob[0][1]) * 100.0
    total_votes = max(1, (ckd_votes + healthy_votes))
    ckd_vote_pct = (ckd_votes / total_votes) * 100.0
    healthy_vote_pct = (healthy_votes / total_votes) * 100.0
    confidence_pct = max(ckd_prob, healthy_prob)

    is_ckd = (prediction == 0)
    badge_class = "badge-ckd" if is_ckd else "badge-healthy"
    glow_class = "glow-ckd" if is_ckd else "glow-healthy"
    badge_text = "CKD Detected" if is_ckd else "Healthy"
    result_icon = "🩺" if is_ckd else "✅"
    result_headline = "Patient has CKD (Unhealthy)" if is_ckd else "Patient is Healthy"
    result_sub = "Clinical decision support only. Confirm with physician review." if is_ckd else "Continue routine monitoring and healthy lifestyle."

    with left:
        st.markdown(
            f"""
<div class="result-card {glow_class} fade-in">
  <div>
    <div class="result-title">
      <h2>Prediction Result</h2>
      <div class="badge {badge_class}">{badge_text}</div>
    </div>
    <div class="result-main">
      <div class="result-icon">{result_icon}</div>
      <div class="result-text">
        <b>{result_headline}</b>
        <span>{result_sub}</span>
      </div>
    </div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="card fade-in" style="margin-top: 12px;">
  <div class="card-title">
    <div class="left">
      <div class="icon">🌳</div>
      <h2>Tree Voting Analysis</h2>
    </div>
  </div>
  <div class="subtle">Vote share is shown as stat cards (ensemble). Probabilities are shown separately.</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="stats fade-in">
  <div class="stat">
    <div class="stat-head">
      <div class="k">CKD Votes</div>
      <div class="stat-badge stat-red">🌳</div>
    </div>
    <div class="v v-red">{ckd_votes}</div>
    <div class="hint">{ckd_vote_pct:.1f}% of trees voted CKD</div>
  </div>
  <div class="stat">
    <div class="stat-head">
      <div class="k">Healthy Votes</div>
      <div class="stat-badge stat-green">🌳</div>
    </div>
    <div class="v v-green">{healthy_votes}</div>
    <div class="hint">{healthy_vote_pct:.1f}% of trees voted Healthy</div>
  </div>
  <div class="stat">
    <div class="stat-head">
      <div class="k">Confidence</div>
      <div class="stat-badge" style="background: rgba(108,99,255,0.12); border-color: rgba(108,99,255,0.20);">📈</div>
    </div>
    <div class="v">{confidence_pct:.1f}%</div>
    <div class="hint">Max class probability</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="card fade-in" style="margin-top: 12px;">
  <div class="card-title">
    <div class="left">
      <div class="icon">🧬</div>
      <h2>Probability Analysis</h2>
    </div>
  </div>
  <div class="subtle">Colored bars represent model probabilities (not votes).</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
<div class="card fade-in" style="margin-top: 10px;">
  <div class="prob-wrap">
    <div class="prob-row">
      <div class="prob-label"><span>CKD probability</span><span>{ckd_prob:.2f}%</span></div>
      <div class="prob-bar">
        <div class="prob-fill red" style="--w:{ckd_prob:.2f}%;">{ckd_prob:.0f}%</div>
      </div>
    </div>
    <div class="prob-row">
      <div class="prob-label"><span>Healthy probability</span><span>{healthy_prob:.2f}%</span></div>
      <div class="prob-bar">
        <div class="prob-fill green" style="--w:{healthy_prob:.2f}%;">{healthy_prob:.0f}%</div>
      </div>
    </div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        ring_ckd_color = ALERT_RED
        ring_healthy_color = SECONDARY
        st.markdown(
            f"""
<div class="card fade-in">
  <h2>Probability Rings</h2>
  <div class="subtle">Quick glance confidence indicators.</div>
  <div class="rings" style="margin-top: 12px;">
    <div class="ring-card">
      <div class="ring-wrap">
        <div class="ring" style="--p:{ckd_prob:.2f}; --c:{ring_ckd_color};"></div>
        <div class="ring-label">{ckd_prob:.0f}%</div>
      </div>
      <div class="ring-text">
        <b>CKD</b>
        <span>Model probability</span>
      </div>
    </div>
    <div class="ring-card">
      <div class="ring-wrap">
        <div class="ring" style="--p:{healthy_prob:.2f}; --c:{ring_healthy_color};"></div>
        <div class="ring-label">{healthy_prob:.0f}%</div>
      </div>
      <div class="ring-text">
        <b>Healthy</b>
        <span>Model probability</span>
      </div>
    </div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
<div class="footer fade-in">
  Powered by Machine Learning | Developed for Healthcare Innovation
</div>
</div>
    """,
    unsafe_allow_html=True,
)
