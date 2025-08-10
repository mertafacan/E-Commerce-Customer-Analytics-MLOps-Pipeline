import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import mlflow
from omegaconf import DictConfig

def run_ab_test(cfg: DictConfig, **kwargs):
    # Environment variables from Hydra configuration
    EXPERIMENT_NAME = cfg.mlflow.experiments.ab_testing
    RFM_PARQUET = cfg.data.paths.rfm_output_file
    SEED = cfg.data.ab_testing.seed
    SAMPLE_FRAC = cfg.data.ab_testing.sample_frac

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

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
        mlflow.log_param("seed", cfg.data.ab_testing.seed)
        mlflow.log_param("sample_frac", cfg.data.ab_testing.sample_frac)
        mlflow.log_metric("p_value", float(p_value))
        mlflow.log_metric("a_n", a_n)
        mlflow.log_metric("b_n", b_n)
        mlflow.log_metric("a_mean", a_mean)
        mlflow.log_metric("b_mean", b_mean)
        mlflow.log_metric("uplift_mean", float(uplift))

    print(f"[A/B] N(A)={a_n}, N(B)={b_n}, p-value={p_value:.4f}, uplift={uplift*100:.1f}%")

if __name__ == '__main__':
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    # Clear any existing Hydra instance to avoid conflicts
    GlobalHydra.instance().clear()

    # Initialize Hydra with configuration path
    initialize(version_base=None, config_path="conf")
    cfg = compose(config_name="config")

    run_ab_test(cfg)
