import streamlit as st
import pandas as pd
import numpy as np

from model_utils import (
    load_churn_xgb, load_churn_rf, load_rfm_kmeans,
    calculate_rfm_scores, segment_maker, load_rfm_thresholds_from_mlflow,
    load_clv_models, calculate_clv_prediction, get_latest_ab_overview,
)

st.set_page_config(page_title="E-Commerce Customer Analytics", page_icon="📊")

# ---- Model Load and Cache ----
@st.cache_resource(show_spinner=False)
def cached_churn_xgb(): return load_churn_xgb()
@st.cache_resource(show_spinner=False)
def cached_churn_rf(): return load_churn_rf()
@st.cache_resource(show_spinner=False)
def cached_rfm_model(): return load_rfm_kmeans()
@st.cache_resource(show_spinner=False)
def cached_clv_models(): return load_clv_models()
@st.cache_data(show_spinner=False)
def cached_rfm_thresholds(): return load_rfm_thresholds_from_mlflow()

# ---- Sidebar: cache panel ----
with st.sidebar:
    st.markdown("### ⚙️ Cache")
    st.caption("Cache is used for speed; first operation may take longer after clearing.")
    if st.button("🧹 Clear Cache", type="primary", use_container_width=True):
        st.cache_data.clear(); st.cache_resource.clear()
        st.success("Cache cleared.")
    st.divider()


# ---- Helper Functions ----
def predict_churn(_model, frequency, monetary, tenure, avg_order_value) -> float:
    if _model is None: raise ValueError("Model could not be loaded.")
    X = np.array([[np.float64(frequency), np.float64(monetary), np.float64(tenure), np.float64(avg_order_value)]])
    try: proba = _model.predict_proba(X)
    except Exception:
        df = pd.DataFrame(X, columns=["frequency","monetary","tenure","avg_order_value"])
        proba = _model.predict_proba(df)
    return float(proba[0][1])

def predict_rfm_cluster(_model, recency, frequency, monetary) -> int:
    if _model is None: raise ValueError("Model could not be loaded.")
    X = np.array([[np.float64(recency), np.float64(frequency), np.float64(monetary)]])
    try: cluster = _model.predict(X)
    except Exception:
        df = pd.DataFrame(X, columns=["recency","frequency","monetary"])
        cluster = _model.predict(df)
    return int(cluster[0])

def predict_clv(_bgf, _ggf, frequency, recency, T, monetary_value, time_months) -> float:
    if _bgf is None or _ggf is None: raise ValueError("CLV models could not be loaded.")
    return calculate_clv_prediction(_bgf, _ggf, frequency, recency, T, monetary_value, time_months)

# ---------- Churn Page ----------
def churn_analysis():
    st.header("🔮 Churn Prediction")
    model_choice = st.selectbox("Model", ["XGBoost", "Random Forest"], index=0)
    col1, col2 = st.columns([1,1])
    with col1:
        frequency = st.number_input("Transaction Count (F)", min_value=0, value=10, step=1)
        monetary  = st.number_input("Total Amount (£)", min_value=0.0, value=1000.0, step=10.0)
    with col2:
        tenure = st.number_input("Customer Age (days)", min_value=1, value=365, step=1)
        avg_order_value = st.number_input("Average Basket (£)", min_value=0.0, value=100.0, step=1.0)

    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
            model = cached_churn_xgb() if model_choice == "XGBoost" else cached_churn_rf()
            churn_p = predict_churn(model, frequency, monetary, tenure, avg_order_value)
        m1, m2 = st.columns([1,1])
        m1.metric("Churn Probability", f"{churn_p:.2%}")
        (m2.success if churn_p <= 0.5 else m2.error)(
            "Low Risk" if churn_p <= 0.5 else "High Risk"
        )

# ---------- RFM Page ----------
def rfm_analysis():
    st.header("📊 RFM Analysis and Segmentation")
    thr = cached_rfm_thresholds()
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        recency = st.number_input("Recency (days)", 0, 5000, max(1, int(float(thr["recency"][1]))))
    with c2:
        frequency = st.number_input("Frequency (count)", 0, 5000, max(1, int(float(thr["frequency"][1]))))
    with c3:
        monetary = st.number_input("Monetary (£)", 0.0, 1_000_000.0, max(0.0, float(thr["monetary"][1])), step=1.0)

    r, f, m = calculate_rfm_scores(recency, frequency, monetary)
    seg = segment_maker(f"{r}{f}{m}")

    k1, k2, k3, k4 = st.columns([1,1,1,1.3])
    k1.metric("R", r); k2.metric("F", f); k3.metric("M", m); k4.info(f"Segment: **{seg}**")

    if st.button("Persona Prediction (KMeans)", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
            kmeans = cached_rfm_model()
            cluster = predict_rfm_cluster(kmeans, recency, frequency, monetary)
        persona_map = {0: "At-Risk & Lost", 1: "Loyal & Valuable", 2: "Champions", 3: "Whales / B2B"}
        st.success(f"Cluster {cluster} • {persona_map.get(cluster, 'Unknown')}")

# ---------- CLV Page ----------
def clv_analysis():
    st.header("💰 CLV Prediction")
    with st.form("clv_form"):
        st.caption("Values are in days/£; frequency is total purchase count.")
        col1, col2 = st.columns([1,1])
        with col1:
            frequency = st.number_input("Frequency", 0, 1000, 5, 1)
            recency   = st.number_input("Recency (days)", 0, 5000, 30, 1)
        with col2:
            T         = st.number_input("Customer Age T (days)", 1, 10000, 365, 1)
            monetary  = st.number_input("Average Order (£)", 0.0, 100000.0, 100.0, 1.0)
        calculate = st.form_submit_button("Calculate CLV", type="primary", use_container_width=True)

    if calculate:
        with st.spinner("Calculating..."):
            bgf, ggf = cached_clv_models()
            periods = [3, 6, 12]
            clv_vals = [predict_clv(bgf, ggf, frequency, recency, T, monetary, p) for p in periods]

        cols = st.columns(len(periods))
        for i, (p, v) in enumerate(zip(periods, clv_vals)):
            cols[i].metric(f"{p}-Month CLV", f"£{v:.2f}")

        st.subheader("CLV Comparison")
        df_chart = pd.DataFrame({"Period": [f"{p} Months" for p in periods], "CLV": clv_vals}).set_index("Period")
        st.bar_chart(df_chart)

        st.subheader("Detailed Table")
        df_table = pd.DataFrame({
            "Period (Months)": periods,
            "Estimated CLV (£)": [f"£{v:,.2f}" for v in clv_vals],
            "Annual Equivalent (£)": [
                f"£{v*4:,.2f}" if p == 3 else f"£{v*2:,.2f}" if p == 6 else f"£{v:,.2f}"
                for p, v in zip(periods, clv_vals)
            ],
        })
        st.dataframe(df_table, use_container_width=True)

# ---------- A/B Test Page ----------
def ab_test_info():
    st.header("🧪 A/B Test Results")
    st.caption("Measures the impact of a discount campaign on spending; in simulation, group B's spending was increased by 20%.")
    st.caption("Normality tested with Shapiro-Wilk, statistical significance evaluated with Mann-Whitney U.")
    info = get_latest_ab_overview("AB_Testing")
    if not info:
        st.warning("A/B run not found in MLflow. Run `python ab_test.py`.")
        return

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    c1.metric("p-value", f"{info.get('p_value', 0):.4f}")
    c2.metric("Group A (N)", f"{int(info.get('a_n', 0)):,}")
    c3.metric("Group B (N)", f"{int(info.get('b_n', 0)):,}")
    c4.metric("Uplift (%)", f"{(info.get('uplift_mean', 0)*100):.1f}")

    st.subheader("Interpretation")
    p_value = info.get('p_value') or 1
    uplift_mean = (info.get('uplift_mean', 0) or 0) * 100
    if p_value < 0.05:
        st.success(f"Significant effect: campaign increased spending by %{uplift_mean:.1f}.")
    else:
        st.warning("Not statistically significant.")
    st.caption("**Note:** p-value < 0.05 → result considered *significant*. **0.0000** value indicates p-value is very close to zero, meaning the result is statistically significant.")

def main():
    st.title("🛒 E-Commerce Customer Analytics")
    st.caption("Compact and fast: **Churn**, **RFM**, **CLV** and **A/B**")
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Churn", "📊 RFM", "💰 CLV", "🧪 A/B Test"])
    with tab1: churn_analysis()
    with tab2: rfm_analysis()
    with tab3: clv_analysis()
    with tab4: ab_test_info()

if __name__ == "__main__":
    main()