import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

RFM_EXPERIMENT_NAME = os.getenv("RFM_EXPERIMENT", "RFM_Segmentation")
CLV_EXPERIMENT_NAME = os.getenv("CLV_EXPERIMENT", "CLV_Analysis")
RFM_REGISTERED_MODEL = os.getenv("RFM_REGISTERED_MODEL", "rfm_kmeans_clustering")
CHURN_XGB_MODEL = os.getenv("CHURN_XGB_MODEL", "churn_xgboost")
CHURN_RF_MODEL = os.getenv("CHURN_RF_MODEL", "churn_randomforest")

LOCAL_RFM_PARQUET = os.getenv("RFM_OUTPUT_FILE", "data/rfm_analysis.parquet")
LOCAL_CLEAN_PARQUET = os.getenv("CLEAN_PARQUET_FILE", "data/cleaned_online_retail_II.parquet")

_rfm_thresholds_cache: Optional[Dict[str, list]] = None

def _params_to_thresholds(params: Dict[str, str]) -> Optional[Dict[str, list]]:
    """
    Converts RFM thresholds from MLflow param dictionary to suitable dictionaries.
    """
    req = [
        "recency_q1","recency_q2","recency_q3","recency_q4",
        "frequency_q1","frequency_q2","frequency_q3","frequency_q4",
        "monetary_q1","monetary_q2","monetary_q3","monetary_q4",
    ]
    if not params or not all(k in params for k in req):
        return None
    return {
        "recency": [float(params[f"recency_q{i}"]) for i in range(1,5)],
        "frequency": [float(params[f"frequency_q{i}"]) for i in range(1,5)],
        "monetary": [float(params[f"monetary_q{i}"]) for i in range(1,5)],
    }

def _compute_thresholds_from_rfm_df(rfm: pd.DataFrame) -> Dict[str, list]:
    """
    Calculates threshold values from given RFM DataFrame based on percentiles.
    """
    return {
        "recency": [float(x) for x in np.percentile(rfm["Recency"], [20, 40, 60, 80])],
        "frequency": [float(x) for x in np.percentile(rfm["Frequency"], [20, 40, 60, 80])],
        "monetary": [float(x) for x in np.percentile(rfm["Monetary"], [20, 40, 60, 80])],
    }

def _compute_thresholds_from_rfm_parquet(path: str) -> Dict[str, list]:
    """
    Reads RFM parquet file and calculates threshold values.
    """
    return _compute_thresholds_from_rfm_df(pd.read_parquet(path))

def _compute_thresholds_from_clean_parquet(path: str) -> Dict[str, list]:
    """
    Generates RFM table from clean data parquet and calculates threshold values.
    """
    df = pd.read_parquet(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"]).copy()
    analyse_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("Customer ID").agg(
            Recency=("InvoiceDate", lambda x: (analyse_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalPrice", "sum"),
        ).reset_index()
    )
    return _compute_thresholds_from_rfm_df(rfm)

def load_rfm_thresholds_from_mlflow() -> Dict[str, list]:
    """
    Loads RFM thresholds with MLflow and local fallbacks, caches in memory.
    """
    global _rfm_thresholds_cache
    if _rfm_thresholds_cache is not None:
        return _rfm_thresholds_cache

    client = MlflowClient()
    try:
        exp = client.get_experiment_by_name(RFM_EXPERIMENT_NAME)
        if exp:
            try:
                runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
            except Exception:
                runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
            if runs:
                r = runs[0]
                thr = _params_to_thresholds(r.data.params)
                if thr:
                    _rfm_thresholds_cache = thr
                    return thr
                try:
                    local = download_artifacts(run_id=r.info.run_id, artifact_path="rfm_outputs/rfm_analysis.parquet")
                    thr = _compute_thresholds_from_rfm_parquet(local)
                    _rfm_thresholds_cache = thr
                    return thr
                except Exception:
                    pass
    except Exception:
        pass

    p1 = Path(LOCAL_RFM_PARQUET)
    if p1.exists():
        thr = _compute_thresholds_from_rfm_parquet(str(p1))
        _rfm_thresholds_cache = thr
        return thr
    p2 = Path(LOCAL_CLEAN_PARQUET)
    if p2.exists():
        thr = _compute_thresholds_from_clean_parquet(str(p2))
        _rfm_thresholds_cache = thr
        return thr

    raise RuntimeError("RFM thresholds not found. Log RFM step to MLflow first.")

def calculate_rfm_scores(recency: float, frequency: float, monetary: float) -> Tuple[int, int, int]:
    """
    Converts continuous values to 1-5 R,F,M scores based on thresholds.
    """
    thr = load_rfm_thresholds_from_mlflow()

    r_q = thr["recency"]
    if recency <= r_q[0]: r = 5
    elif recency <= r_q[1]: r = 4
    elif recency <= r_q[2]: r = 3
    elif recency <= r_q[3]: r = 2
    else: r = 1

    f_q = thr["frequency"]
    if frequency >= f_q[3]: f = 5
    elif frequency >= f_q[2]: f = 4
    elif frequency >= f_q[1]: f = 3
    elif frequency >= f_q[0]: f = 2
    else: f = 1

    m_q = thr["monetary"]
    if monetary >= m_q[3]: m = 5
    elif monetary >= m_q[2]: m = 4
    elif monetary >= m_q[1]: m = 3
    elif monetary >= m_q[0]: m = 2
    else: m = 1

    return int(r), int(f), int(m)

def segment_maker(rfm_score: str) -> str:
    """
    Converts RFM score string (e.g. "543") to segment label.
    """
    if rfm_score == "555": return "Best Customers"
    if len(rfm_score) >= 3 and rfm_score[1] == "5" and rfm_score[2] == "5": return "Loyal Big Spenders"
    if len(rfm_score) >= 3 and rfm_score[0] == "5" and rfm_score[1] == "5": return "Loyal & Recent"
    if len(rfm_score) >= 1 and rfm_score[0] == "5": return "Recent Customer"
    if len(rfm_score) >= 2 and rfm_score[1] == "5": return "Loyal Customer"
    if len(rfm_score) >= 3 and rfm_score[2] == "5": return "Big Spender"
    if rfm_score == "111": return "At Risk"
    if len(rfm_score) >= 1 and rfm_score[0] == "1": return "Lost Customer"
    if len(rfm_score) >= 3 and rfm_score[1] in ["3","4"] and rfm_score[2] in ["3","4"]: return "Promising Customers"
    return "Regular Customer"

def _load_registered_model(model_name: str, version: str = "latest"):
    """
    Loads model from MLflow Registered Model repository by name/version.
    """
    try:
        from mlflow import sklearn as ml_sklearn
        try:
            return ml_sklearn.load_model(f"models:/{model_name}/{version}")
        except Exception:
            from mlflow import pyfunc
            return pyfunc.load_model(f"models:/{model_name}/{version}")
    except Exception:
        return None

def load_churn_xgb():
    """
    Helper function to load XGBoost churn model.
    """
    return _load_registered_model(CHURN_XGB_MODEL)

def load_churn_rf():
    """
    Helper function to load RandomForest churn model.
    """
    return _load_registered_model(CHURN_RF_MODEL)

def load_rfm_kmeans():
    """
    Helper function to load RFM KMeans model.
    """
    return _load_registered_model(RFM_REGISTERED_MODEL)

def load_clv_models():
    """
    Loads BGF and GGF models from the last run in CLV experiment.
    """
    try:
        client = MlflowClient()
        exp = client.get_experiment_by_name(CLV_EXPERIMENT_NAME)
        if not exp: return None, None
        runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
        if not runs: return None, None
        run_id = runs[0].info.run_id
        bgf_local = client.download_artifacts(run_id, "lifetimes_models/bgf_model.pkl")
        ggf_local = client.download_artifacts(run_id, "lifetimes_models/ggf_model.pkl")
        import cloudpickle as pickle
        with open(bgf_local, "rb") as f: bgf = pickle.load(f)
        with open(ggf_local, "rb") as f: ggf = pickle.load(f)
        return bgf, ggf
    except Exception:
        return None, None

def calculate_clv_prediction(bgf, ggf, frequency: float, recency: float, T: float, monetary_value: float,
                             time_months: int = 6, discount_rate: float = 0.01) -> float:
    """
    Makes CLV prediction with lifetimes models.
    """
    try:
        if frequency <= 0 or monetary_value <= 0 or recency < 0 or T < recency:
            return 0.0
        s = ggf.customer_lifetime_value(
            bgf, [float(frequency)], [float(recency)], [float(T)], [float(monetary_value)],
            time=int(time_months), discount_rate=float(discount_rate), freq="D"
        )
        return max(0.0, float(s.iloc[0]))
    except Exception:
        try:
            ep = bgf.conditional_expected_number_of_purchases_up_to_time(
                int(time_months), float(frequency), float(recency), float(T)
            )
            ap = ggf.conditional_expected_average_profit(float(frequency), float(monetary_value))
            return max(0.0, float(ep * ap))
        except Exception:
            return 0.0

def get_latest_ab_overview(experiment_name: str = "AB_Testing") -> Optional[Dict]:
    """
    Returns summary metrics for the latest A/B test run for the given experiment.
    """
    try:
        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if not exp: return None
        runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
        if not runs: return None
        r = runs[0]; m = r.data.metrics; p = r.data.params
        return {
            "run_id": r.info.run_id,
            "p_value": float(m.get("p_value")) if "p_value" in m else None,
            "a_n": int(m.get("a_n")) if "a_n" in m else None,
            "b_n": int(m.get("b_n")) if "b_n" in m else None,
            "a_mean": float(m.get("a_mean")) if "a_mean" in m else None,
            "b_mean": float(m.get("b_mean")) if "b_mean" in m else None,
            "uplift_mean": float(m.get("uplift_mean")) if "uplift_mean" in m else None,
            "seed": p.get("seed"),
            "sample_frac": p.get("sample_frac"),
        }
    except Exception:
        return None