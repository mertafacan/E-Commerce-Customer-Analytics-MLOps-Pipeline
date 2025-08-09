import os
import tempfile
from pathlib import Path
import datetime as dt

import pandas as pd
import mlflow
import cloudpickle as pickle
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

# Environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "CLV_Analysis")
INPUT_PARQUET = os.getenv("CLEAN_PARQUET_FILE", "data/cleaned_online_retail_II.parquet")
OUTPUT_PARQUET = os.getenv("CLV_OUTPUT_FILE", "data/clv_analysis.parquet")

BGF_PENALIZER = float(os.getenv("BGF_PENALIZER", "0.001"))
GGF_PENALIZER = float(os.getenv("GGF_PENALIZER", "0.01"))
DISCOUNT_RATE = float(os.getenv("CLV_DISCOUNT_RATE", "0.01"))

DAYS_PER_MONTH = float(os.getenv("DAYS_PER_MONTH", "30"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def load_data(path: str) -> pd.DataFrame:
    """Loads data from parquet file and converts InvoiceDate to datetime."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_parquet(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df

def fit_models_and_compute_clv(df: pd.DataFrame) -> pd.DataFrame:
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
    bgf = BetaGeoFitter(penalizer_coef=BGF_PENALIZER)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    # GammaGammaFitter: fit monetary model only for frequency > 0
    positive = summary["frequency"] > 0
    ggf = GammaGammaFitter(penalizer_coef=GGF_PENALIZER)
    ggf.fit(summary.loc[positive, "frequency"], summary.loc[positive, "monetary_value"])

    # Calculate 3, 6 and 12-month CLV
    summary["clv_3_months"] = ggf.customer_lifetime_value(
        bgf,
        summary["frequency"],
        summary["recency"],
        summary["T"],
        summary["monetary_value"],
        time=int(3 * DAYS_PER_MONTH),
        freq="D",
        discount_rate=DISCOUNT_RATE,
    )

    summary["clv_6_months"] = ggf.customer_lifetime_value(
        bgf,
        summary["frequency"],
        summary["recency"],
        summary["T"],
        summary["monetary_value"],
        time=int(6 * DAYS_PER_MONTH),
        freq="D",
        discount_rate=DISCOUNT_RATE,
    )

    summary["clv_12_months"] = ggf.customer_lifetime_value(
        bgf,
        summary["frequency"],
        summary["recency"],
        summary["T"],
        summary["monetary_value"],
        time=int(12 * DAYS_PER_MONTH),
        freq="D",
        discount_rate=DISCOUNT_RATE,
    )

    return summary, bgf, ggf

def run():
    """
    Runs the entire CLV pipeline, logs with MLflow, and saves outputs.
    """
    print("Starting — MLflow tracking:", MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME", "CLV_Lifetimes")):
        mlflow.log_params(
            {
                "input_parquet": INPUT_PARQUET,
                "output_parquet": OUTPUT_PARQUET,
                "bgf_penalizer": BGF_PENALIZER,
                "ggf_penalizer": GGF_PENALIZER,
                "discount_rate": DISCOUNT_RATE,
                "days_per_month": DAYS_PER_MONTH,
            }
        )

        df = load_data(INPUT_PARQUET)
        clv_df, bgf, ggf = fit_models_and_compute_clv(df)

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
        out_p = Path(OUTPUT_PARQUET)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        clv_df.to_parquet(out_p, index=False)
        mlflow.log_artifact(str(out_p), artifact_path="clv_outputs")

        # Log artifact directories as parameters
        mlflow.log_param("artifact_models_dir", "lifetimes_models")
        mlflow.log_param("artifact_outputs_dir", "clv_outputs")

        print(f"✅ CLV completed — saved: {out_p}")
        print(f"📊 Average CLV (6 months): £{clv_df['clv_6_months'].mean():.2f}")

    return str(out_p)

if __name__ == "__main__":
    run()
