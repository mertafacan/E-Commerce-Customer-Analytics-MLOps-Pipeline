import os
import tempfile
from pathlib import Path
import datetime as dt

import pandas as pd
import mlflow
import cloudpickle as pickle
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from omegaconf import DictConfig

def load_data(path: str) -> pd.DataFrame:
    """Loads data from parquet file and converts InvoiceDate to datetime."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_parquet(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df

def fit_models_and_compute_clv(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """
    Fits lifetimes models for CLV calculation and computes 3, 6, 12-month CLV.
    """
    # End of observation period
    observation_end = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Create customer summary data
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col="Customer ID",
        datetime_col="InvoiceDate",
        monetary_value_col="TotalPrice",
        observation_period_end=observation_end,
    )

    # BetaGeoFitter: fit frequency/recency model
    bgf = BetaGeoFitter(penalizer_coef=cfg.data.clv.bgf_penalizer)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    # GammaGammaFitter: fit monetary model only for frequency > 0
    positive = summary["frequency"] > 0
    ggf = GammaGammaFitter(penalizer_coef=cfg.data.clv.ggf_penalizer)
    ggf.fit(summary.loc[positive, "frequency"], summary.loc[positive, "monetary_value"])

    # Calculate 3, 6 and 12-month CLV
    summary["clv_3_months"] = ggf.customer_lifetime_value(
        bgf,
        summary["frequency"],
        summary["recency"],
        summary["T"],
        summary["monetary_value"],
        time=int(3 * cfg.data.clv.days_per_month),
        freq="D",
        discount_rate=cfg.data.clv.discount_rate,
    )

    summary["clv_6_months"] = ggf.customer_lifetime_value(
        bgf,
        summary["frequency"],
        summary["recency"],
        summary["T"],
        summary["monetary_value"],
        time=int(6 * cfg.data.clv.days_per_month),
        freq="D",
        discount_rate=cfg.data.clv.discount_rate,
    )

    summary["clv_12_months"] = ggf.customer_lifetime_value(
        bgf,
        summary["frequency"],
        summary["recency"],
        summary["T"],
        summary["monetary_value"],
        time=int(12 * cfg.data.clv.days_per_month),
        freq="D",
        discount_rate=cfg.data.clv.discount_rate,
    )

    return summary, bgf, ggf

def clv_analysis(cfg: DictConfig, **kwargs):
    """
    Runs the entire CLV pipeline, logs with MLflow, and saves outputs.
    """
    # Environment variables from Hydra configuration
    mlflow_tracking_uri = cfg.mlflow.tracking_uri
    experiment_name = cfg.mlflow.experiments.clv_analysis
    input_parquet = cfg.data.paths.clean_parquet_file
    output_parquet = cfg.data.paths.clv_output_file
    
    bgf_penalizer = cfg.data.clv.bgf_penalizer
    ggf_penalizer = cfg.data.clv.ggf_penalizer
    discount_rate = cfg.data.clv.discount_rate
    days_per_month = cfg.data.clv.days_per_month

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    print("Starting — MLflow tracking:", mlflow_tracking_uri)
    with mlflow.start_run(run_name=cfg.mlflow_run_names.clv_analysis):
        mlflow.log_params(
            {
                "input_parquet": input_parquet,
                "output_parquet": output_parquet,
                "bgf_penalizer": bgf_penalizer,
                "ggf_penalizer": ggf_penalizer,
                "discount_rate": discount_rate,
                "days_per_month": days_per_month,
            }
        )

        df = load_data(input_parquet)
        clv_df, bgf, ggf = fit_models_and_compute_clv(df, cfg)

        # Log CLV and customer metrics
        mlflow.log_metric("avg_clv_3m", float(clv_df["clv_3_months"].mean()))
        mlflow.log_metric("avg_clv_6m", float(clv_df["clv_6_months"].mean()))
        mlflow.log_metric("avg_clv_12m", float(clv_df["clv_12_months"].mean()))
        mlflow.log_metric("customers_count", int(len(clv_df)))
        mlflow.log_metric("avg_frequency", float(clv_df["frequency"].mean()))
        mlflow.log_metric("avg_monetary", float(clv_df["monetary_value"].mean()))

        # Save models with cloudpickle and log as artifacts
        with tempfile.TemporaryDirectory() as tmp:
            bgf_path = Path(tmp) / "bgf_model.pkl"
            ggf_path = Path(tmp) / "ggf_model.pkl"
            with open(bgf_path, "wb") as f:
                pickle.dump(bgf, f)
            with open(ggf_path, "wb") as f:
                pickle.dump(ggf, f)

            mlflow.log_artifact(str(bgf_path), artifact_path="lifetimes_models")
            mlflow.log_artifact(str(ggf_path), artifact_path="lifetimes_models")

        # Save CLV output and log as artifact
        out_p = Path(output_parquet)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        clv_df.to_parquet(out_p, index=False)
        mlflow.log_artifact(str(out_p), artifact_path="clv_outputs")

        # Log artifact directories as parameters
        mlflow.log_param("artifact_models_dir", "lifetimes_models")
        mlflow.log_param("artifact_outputs_dir", "clv_outputs")

        print(f"✅ CLV completed — saved: {out_p}")
        print(f"📊 Average CLV (6 months): £{clv_df['clv_6_months'].mean():.2f}")

    return str(out_p)

if __name__ == '__main__':
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    # Clear any existing Hydra instance to avoid conflicts
    GlobalHydra.instance().clear()

    # Initialize Hydra with configuration path
    initialize(version_base=None, config_path="conf")
    cfg = compose(config_name="config")

    clv_analysis(cfg)
