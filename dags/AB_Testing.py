import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import mlflow

# Environment variables
EXPERIMENT_NAME = os.getenv("AB_EXPERIMENT_NAME", "AB_Testing")
RFM_PARQUET = os.getenv("RFM_PARQUET_FILE", "data/rfm_analysis.parquet")
SEED = int(os.getenv("AB_SEED", "42"))
SAMPLE_FRAC = float(os.getenv("AB_SAMPLE_FRAC", "1.0"))

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000"))
mlflow.set_experiment(EXPERIMENT_NAME)

def run():
    # Data
    rfm = pd.read_parquet(RFM_PARQUET)
    df = rfm[rfm["Segment"] == "Promising Customers"].copy()
    # Sampling and A/B assignment
    rng = np.random.default_rng(SEED)
    if SAMPLE_FRAC < 1.0:
        df = df.sample(frac=SAMPLE_FRAC, random_state=SEED).copy()
    df["Group"] = rng.choice(["A", "B"], size=len(df))

    # Simple metrics: AOV and synthetic purchase value
    freq = np.maximum(df["Frequency"].astype(float), 1.0)
    df["AvgOrderValue"] = (df["Monetary"].astype(float) / freq).clip(lower=0.0)

    a_mask = df["Group"] == "A"
    b_mask = ~a_mask
    df.loc[a_mask, "SyntheticPurchaseValue"] = (
        df.loc[a_mask, "AvgOrderValue"] * rng.normal(1.0, 0.10, a_mask.sum())
    ).abs()
    df.loc[b_mask, "SyntheticPurchaseValue"] = (
        df.loc[b_mask, "AvgOrderValue"] * rng.normal(1.20, 0.15, b_mask.sum())
    ).abs()

    a_vals = df.loc[a_mask, "SyntheticPurchaseValue"].dropna()
    b_vals = df.loc[b_mask, "SyntheticPurchaseValue"].dropna()
    a_n, b_n = int(a_vals.size), int(b_vals.size)

    # 4) Test and summary
    stat, p_value = mannwhitneyu(b_vals, a_vals, alternative="two-sided")
    a_mean = float(a_vals.mean())
    b_mean = float(b_vals.mean())
    uplift = (b_mean / max(1e-9, a_mean)) - 1.0

    # 5) Required MLflow logs (minimum set)
    with mlflow.start_run(run_name="ab_run"):
        mlflow.log_param("seed", SEED)
        mlflow.log_param("sample_frac", SAMPLE_FRAC)
        mlflow.log_metric("p_value", float(p_value))
        mlflow.log_metric("a_n", a_n)
        mlflow.log_metric("b_n", b_n)
        mlflow.log_metric("a_mean", a_mean)
        mlflow.log_metric("b_mean", b_mean)
        mlflow.log_metric("uplift_mean", float(uplift))

    print(f"[A/B] N(A)={a_n}, N(B)={b_n}, p-value={p_value:.4f}, uplift={uplift*100:.1f}%")

if __name__ == "__main__":
    run()
