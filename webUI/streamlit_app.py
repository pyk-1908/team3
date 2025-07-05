import os
import sys
from PIL import Image

# Make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from Pipeline.models import get_model

# --- Page config ---
st.set_page_config(layout="wide", page_title="BCG Churn Analytics")

st.markdown("""
<style>
body, .block-container, [data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# --- Global CSS tweaks & Slider styling ---
st.markdown("""
<style>
  [data-testid="stSidebar"] { background-color: #DCF9E3 !important; }
  .stSidebar .stRadio > label,
  .stSidebar .stSlider > div,
  .stSidebar .stNumberInput > label {
    color: #004734 !important;
  }
  input[type="range"]::-webkit-slider-runnable-track { background: #004734; height:6px; border-radius:3px; }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance:none; margin-top:-6px;
    background:#fff; border:2px solid #004734;
    height:16px;width:16px;border-radius:50%;cursor:pointer;
  }
  input[type="range"]::-moz-range-track { background: #004734; height:6px; border-radius:3px; }
  input[type="range"]::-moz-range-thumb {
    background:#fff; border:2px solid #004734;
    height:16px;width:16px;border-radius:50%;cursor:pointer;
  }
  input[type="range"]::-ms-fill-lower  { background: #004734; }
  input[type="range"]::-ms-fill-upper  { background: #004734; }
  input[type="range"]::-ms-thumb {
    background:#fff; border:2px solid #004734;
    height:16px;width:16px;border-radius:50%;cursor:pointer;
  }
</style>
""", unsafe_allow_html=True)

# --- Landing Page state ---
if "started" not in st.session_state:
    st.session_state.started = False

# --- Landing Page ---
if not st.session_state.started:
    left, right = st.columns([1,1], gap="large")
    with left:
        st.image(os.path.join(os.path.dirname(__file__), "assets/bcg_logo.png"), width=150)
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.image(os.path.join(os.path.dirname(__file__), "assets/uni_saarland_logo.png"), width=150)
        st.markdown("<h1 style='color:#003e4f; font-size:3rem;'>BCG Churn Analytics</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#355c60; font-size:1.2rem;'>Predict churn before it happens with powerful visualizations and predictions.</p>", unsafe_allow_html=True)
        if st.button("Get Started"):
            st.session_state.started = True
    with right:
        try:
            st.image(Image.open(os.path.join(os.path.dirname(__file__), "assets/landing.png")), use_container_width=True)
        except FileNotFoundError:
            st.error("Landing image not found")
    st.stop()

# --- Data loading & models ---
@st.cache_data
def load_data():
    df = pd.read_csv("Pipeline/data/Cate_added_data.csv")
    df = df.rename(columns={"QuarterInt":"QuarterIdx","CATE_DR":"CATE_CausalML"})
    df = df[["Provider","Regionality","QuarterIdx","RiskFactor","Members_Lag","Rate_Lag","ChurnRate","CATE_CausalML"]]
    df["Provider"] = df["Provider"].str.strip()
    for c in ["QuarterIdx","RiskFactor","Members_Lag","Rate_Lag","ChurnRate","CATE_CausalML"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data
def load_time_series():
    return pd.read_csv("Pipeline/data/Final.csv")

@st.cache_resource
def fit_predictive(df):
    m = get_model("gradient_boosting")
    m.fit(df[["Rate_Lag","Members_Lag","RiskFactor","QuarterIdx"]], df["ChurnRate"])
    return m

@st.cache_resource
def fit_cate_model(df):
    #m = RandomForestRegressor(random_state=42)
    m =  get_model("gradient_boosting")
    m.fit(df[["RiskFactor","Rate_Lag","Members_Lag","QuarterIdx"]], df["CATE_CausalML"])
    return m

def get_provider_aggregated_data(df, providers):
    if not isinstance(providers, list):
        providers = [providers]
    out = {}
    for p in providers:
        sub = df[df.Provider.str.contains(p, case=False, na=False)]
        if sub.empty:
            out[p] = None
            continue
        agg = sub.groupby("Year").agg(
            ACR=("ACR","mean"),
            ChurnRate=("ChurnRate","mean"),
            Members=("Members","sum")
        ).reset_index()
        out[p] = agg if not agg.empty else None
    return out

def plot_aggregated_provider_data(aggregated_data):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
    plotted=False
    for name,data in aggregated_data.items():
        if data is None: continue
        ax1.plot(data.Year, data.ACR, marker="o", label=f"ACR – {name}")
        ax2.plot(data.Year, data.ChurnRate, marker="s", label=f"Churn – {name}")
        plotted=True
    if not plotted:
        st.warning("No valid data.")
        return
    ax1.set(title="ACR Trends", xlabel="Year", ylabel="Average ACR"); ax1.grid(alpha=0.3); ax1.legend()
    ax2.set(title="Churn Trends", xlabel="Year", ylabel="Average Churn Rate"); ax2.grid(alpha=0.3); ax2.legend()
    st.pyplot(fig)

def plot_provider_data(df, providers):
    agg = get_provider_aggregated_data(df, providers)
    plot_aggregated_provider_data(agg)



# --- Load everything ---
df     = load_data()
df2    = load_time_series()
pred   = fit_predictive(df)
cate   = fit_cate_model(df)
plist  = df["Provider"].unique().tolist()

# --- Header ---
c1, c2, c3 = st.columns([2,6,2], gap="small")
with c1: st.image(os.path.join(os.path.dirname(__file__), "assets/bcg_logo.png"), width=120)
with c2: st.markdown("<h1 style='text-align:center;color:#004734;'>Churn Rate Insights Dashboard</h1>", unsafe_allow_html=True)
with c3: st.image(os.path.join(os.path.dirname(__file__), "assets/uni_saarland_logo.png"), width=120)


# --- Sidebar ---
st.sidebar.header("🔧 Inputs")
mode = st.sidebar.radio("Mode", ["Single Provider","Compare Providers","Analysis"])

if mode=="Single Provider":
    prov  = st.sidebar.selectbox("Provider", plist)
    acr   = st.sidebar.slider(" Additional Contribution Rate (ACR)", df.Rate_Lag.min(), df.Rate_Lag.max(), df.Rate_Lag.mean(), step=0.1)
    risk  = st.sidebar.number_input("Risk Factor", df.RiskFactor.min(), df.RiskFactor.max(), df.RiskFactor.mean())
    selected=[prov]

elif mode=="Compare Providers":
    selected = st.sidebar.multiselect("Providers (exactly 2)", plist, default=plist[:2])
    if len(selected)!=2: st.sidebar.warning("Select exactly 2"); st.stop()
    acr1 = st.sidebar.slider(f" ACR for {selected[0]}", df.Rate_Lag.min(), df.Rate_Lag.max(), df.Rate_Lag.mean(), step=0.1)
    risk1= st.sidebar.number_input(f"Risk Factor for {selected[0]}", df.RiskFactor.min(), df.RiskFactor.max(), df.RiskFactor.mean())
    acr2 = st.sidebar.slider(f" ACR for {selected[1]}", df.Rate_Lag.min(), df.Rate_Lag.max(), df.Rate_Lag.mean(), step=0.1)
    risk2= st.sidebar.number_input(f"Risk Factor for {selected[1]}", df.Rate_Lag.min(), df.Rate_Lag.max(), df.Rate_Lag.mean())
else:
    selected = st.sidebar.multiselect("Select Providers to analyze", plist, default=plist[:3])
    if not selected: st.sidebar.warning("Pick at least one"); st.stop()

st.markdown("---")

def compute_metrics(provider, rf, ac):
    sub = df[df.Provider==provider].sort_values("QuarterIdx").iloc[-1]
    c = cate.predict([[rf, sub.Rate_Lag, sub.Members_Lag, sub.QuarterIdx]])[0] * ac
    p = pred.predict([[sub.Rate_Lag+ac, sub.Members_Lag, rf, sub.QuarterIdx+1]])[0]
    return float(c), float(p)

# --- Single Provider Output ---
if mode=="Single Provider":
    c_tot, p = compute_metrics(selected[0], risk, acr)
    st.subheader(f"Results for {selected[0]}")

    # Causal
    dfc = pd.DataFrame({"Metric":["Causal churn"], "Value":[c_tot]})
    m = abs(c_tot) * 1.2
    base = alt.Chart(dfc).encode(
    y=alt.Y("Metric:N", axis=None),
    x=alt.X("Value:Q", scale=alt.Scale(domain=[-m, m]))
    )

    bar=base.mark_bar(size=15, color="#2ca02c" if c_tot<0 else "#de425b")
    pt =base.mark_circle(size=250, color="#000000")
    txt_pos = base.mark_text(align='left', dx=5, color="#004734").transform_filter(alt.datum.Value>0).encode(
        text=alt.Text("Value:Q", format=".4f")
    )
    txt_neg = base.mark_text(align='right',dx=-5, color="#004734").transform_filter(alt.datum.Value<0).encode(
        text=alt.Text("Value:Q", format=".4f")
    )
    st.altair_chart((bar+pt+txt_pos+txt_neg).properties(title="CauseHealPred Churn rate", height=120), use_container_width=True)
    st.markdown("""
        <div style="background-color:#f9f9f9; border-left:5px solid #3498db; padding:1rem; border-radius:8px">
        <strong>💡 Info:</strong> <br> A negative churn rate indicates a gain of members for the selected provider.<br>
         A positive churn rate indicates a loss of members for the selected provider.
    """, unsafe_allow_html=True)

    st.markdown("---")
    # Predictive
    dfp = pd.DataFrame({"Metric":["Predicted churn"], "Value":[p]})
    m = abs(p) * 1.2
    base = alt.Chart(dfp).encode(
    y=alt.Y("Metric:N", axis=None),
    x=alt.X("Value:Q", scale=alt.Scale(domain=[-m, m]))
    )

    bar=base.mark_bar(size=15, color="#2ca02c" if p<0 else "#de425b")
    pt =base.mark_circle(size=250, color="#000000")
    txt_pos = base.mark_text(align='left', dx=5, color="#004734").transform_filter(alt.datum.Value>0).encode(
        text=alt.Text("Value:Q", format=".4f")
    )
    txt_neg = base.mark_text(align='right',dx=-5, color="#004734").transform_filter(alt.datum.Value<0).encode(
        text=alt.Text("Value:Q", format=".4f")
    )
    st.altair_chart((bar+pt+txt_pos+txt_neg).properties(title="Predictive Modeling Churn rate (Using basic predictive models)", height=120), use_container_width=True)
    st.markdown("""
    <div style="background-color:#f9f9f9; border-left:5px solid #3498db; padding:1rem; border-radius:8px">
        <strong>💡 Info:</strong> The model-predicted churn rate is based on the selected Additional Contribution Rate (ACR) and risk factor without causal insights.
    </div>
    """, unsafe_allow_html=True)

# --- Compare Providers Output ---
elif mode=="Compare Providers":
    c1,p1 = compute_metrics(selected[0], risk1, acr1)
    c2,p2 = compute_metrics(selected[1], risk2, acr2)
    r1 = df.loc[df.Provider==selected[0],"Regionality"].iloc[-1]
    r2 = df.loc[df.Provider==selected[1],"Regionality"].iloc[-1]
    dfc = pd.DataFrame([
        {"Provider":selected[0],"Regionality":r1,"Causal":c1,"Predicted":p1},
        {"Provider":selected[1],"Regionality":r2,"Causal":c2,"Predicted":p2}
    ])

    with st.expander("👁️ Comparison table"):
        st.table(dfc[["Provider","Regionality","Causal","Predicted"]]
          .style.format({"Causal":"{:.4f}","Predicted":"{:.4f}"}))

    st.markdown("---")

    # Causal compare
    m=max(abs(c1),abs(c2))*1.2
    base=alt.Chart(dfc).encode(
        y=alt.Y("Provider:N",axis=None),
        x=alt.X("Causal:Q",scale=alt.Scale(domain=[-m,m]),title=None)
    )
    bar=base.mark_bar(
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4
    ).encode(
        color=alt.Color("Provider:N",scale=alt.Scale(domain=selected,range=["#2ca02c","#1f77b4"])),
        size=alt.value(15)
    )
    pt = base.mark_circle(size=300, color="#004734")
    txt_p = base.mark_text(align='left', dx=5, color="#004734").transform_filter(alt.datum.Causal>0).encode(
        text=alt.Text("Causal:Q", format=".4f")
    )
    txt_n = base.mark_text(align='right', dx=-5, color="#004734").transform_filter(alt.datum.Causal<0).encode(
        text=alt.Text("Causal:Q", format=".4f")
    )
    st.altair_chart((bar + pt + txt_p + txt_n).properties(title="CauseHealPred Churn rate", height=200), use_container_width=True)
    st.markdown("""
    <div style="background-color:#f9f9f9; border-left:5px solid #3498db; padding:1rem; border-radius:8px">
        <strong>💡 Info:</strong>  How ACR’s impact on churn differs between providers.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Predicted compare
    m=max(abs(p1),abs(p2))*1.2
    base=alt.Chart(dfc).encode(
        y=alt.Y("Provider:N",axis=None),
        x=alt.X("Predicted:Q",scale=alt.Scale(domain=[-m,m]),title=None)
    )
    bar=base.mark_bar(
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4
    ).encode(
        color=alt.Color("Provider:N",scale=alt.Scale(domain=selected,range=["#2ca02c","#1f77b4"])),
        size=alt.value(15)
    )
    pt = base.mark_circle(size=300, color="#004734")
    txt_p = base.mark_text(align='left', dx=5, color="#004734").transform_filter(alt.datum.Predicted>0).encode(
        text=alt.Text("Predicted:Q", format=".4f")
    )
    txt_n = base.mark_text(align='right', dx=-5, color="#004734").transform_filter(alt.datum.Predicted<0).encode(
        text=alt.Text("Predicted:Q", format=".4f")
    )
    st.altair_chart((bar + pt + txt_p + txt_n).properties(title="Predictive Modeling Churn rate (Using basic predictive models)", height=200), use_container_width=True)
    st.markdown("""
    <div style="background-color:#f9f9f9; border-left:5px solid #3498db; padding:1rem; border-radius:8px">
        <strong>💡 Info:</strong>  Each provider’s model‐predicted churn under its own scenario.
    </div>
    """, unsafe_allow_html=True)

# --- Analysis ---
else:
    st.subheader("Aggregated Trends Analysis")
    st.write("Select providers to view their annual Additional Contribution Rate (ACR) and churn trends.")
    plot_provider_data(df2, selected)
