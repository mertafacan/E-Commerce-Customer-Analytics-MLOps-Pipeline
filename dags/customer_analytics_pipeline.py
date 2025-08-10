# dags/customer_analytics_pipeline.py
import os
from pathlib import Path
import sys
import subprocess
import pendulum

from airflow.sdk import dag, task, Asset

INCLUDE_DIR = Path("/usr/local/airflow/include")
DATA_DIR = Path("/usr/local/airflow/data")


@dag(
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    default_args={"owner": "Mert", "retries": 1},
    tags=["customer-analytics", "ecommerce"],
)
def customer_analytics_pipeline():
    raw_to_clean_asset = Asset("cleaned_online_retail_II")
    rfm_asset = Asset("rfm_analysis")

    def _run_script(script: Path, cmd: list[str]) -> None:
        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")
        
        subprocess.run(
            cmd,
            check=True,
            text=True,
            stderr=subprocess.STDOUT,
            cwd="/usr/local/airflow/include",
        )

    @task(outlets=[raw_to_clean_asset])
    def step_01_prepare_with_spark():
        script = INCLUDE_DIR / "Data_Preparation.py"
        _run_script(script, ["spark-submit", str(script)])
        out_file = DATA_DIR / "cleaned_online_retail_II.parquet"
        if not out_file.exists():
            raise FileNotFoundError(f"Expected output not found: {out_file}")

    @task(inlets=[raw_to_clean_asset], outlets=[rfm_asset])
    def step_02_rfm_segmentation():
        script = INCLUDE_DIR / "RFM_and_Segmentation.py"
        _run_script(script, [sys.executable, str(script)])
        out_file = DATA_DIR / "rfm_analysis.parquet"
        if not out_file.exists():
            raise FileNotFoundError(f"Expected output not found: {out_file}")

    @task(inlets=[rfm_asset, raw_to_clean_asset])
    def step_03_churn_prediction():
        script = INCLUDE_DIR / "Churn_Prediction.py"
        _run_script(script, [sys.executable, str(script)])

    @task(inlets=[raw_to_clean_asset])
    def step_04_clv_analysis():
        script = INCLUDE_DIR / "CLV_Analysis.py"
        _run_script(script, [sys.executable, str(script)])

    @task(inlets=[raw_to_clean_asset])
    def step_05_ab_testing():
        script = INCLUDE_DIR / "AB_Testing.py"
        _run_script(script, [sys.executable, str(script)])

    s1 = step_01_prepare_with_spark()
    s2 = step_02_rfm_segmentation()
    s3 = step_03_churn_prediction()
    s4 = step_04_clv_analysis()
    s5 = step_05_ab_testing()
    
    s1 >> s2 >> s3 >> [s4, s5]


customer_analytics_pipeline()
