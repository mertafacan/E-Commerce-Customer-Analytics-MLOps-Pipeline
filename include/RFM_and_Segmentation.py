import os
import datetime as dt
import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from omegaconf import DictConfig


def load_data(path: str) -> pd.DataFrame:
    """Loads the data"""
    p = os.path.join(os.getcwd(), path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_parquet(p)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df.dropna(subset=["InvoiceDate"])


def calculate_rfm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates RFM metrics"""
    analyse_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (analyse_date - x.max()).days,
        "Invoice": "nunique",
        "TotalPrice": "sum"
    }).reset_index()
    rfm.columns = ["Customer ID", "Recency", "Frequency", "Monetary"]
    
    return rfm, analyse_date


def calculate_rfm_scores(rfm: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """Calculates RFM scores"""
    rfm["R_Score"] = pd.qcut(rfm["Recency"], n_quantiles, labels=[5,4,3,2,1])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), n_quantiles, labels=[1,2,3,4,5])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], n_quantiles, labels=[1,2,3,4,5])
    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    
    return rfm


def segment_maker(score: str) -> str:
    """Assigns segments based on RFM scores"""
    if score == '555':
        return 'Best Customers'
    if score[1] == '5' and score[2] == '5':
        return 'Loyal Big Spenders'
    if score[0] == '5' and score[1] == '5':
        return 'Loyal & Recent'
    if score[0] == '5':
        return 'Recent Customer'
    if score[1] == '5':
        return 'Loyal Customer'
    if score[2] == '5':
        return 'Big Spender'
    if score == '111':
        return 'At Risk'
    if score[0] == '1':
        return 'Lost Customer'
    if score[1] in ['3', '4'] and score[2] in ['3', '4']:
        return 'Promising Customers'
    return 'Regular Customer'


def perform_clustering(rfm: pd.DataFrame, n_clusters: int = 4) -> tuple:
    """Performs K-means clustering"""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    
    return rfm, kmeans, rfm_scaled


def add_additional_features(rfm: pd.DataFrame, df: pd.DataFrame, analyse_date) -> pd.DataFrame:
    """Adds additional features"""
    first_purchase = df.groupby("Customer ID")["InvoiceDate"].min().reset_index()
    first_purchase.columns = ["Customer ID", "FirstPurchaseDate"]
    rfm = pd.merge(rfm, first_purchase, on="Customer ID", how="left")
    rfm["Tenure"] = (analyse_date - rfm["FirstPurchaseDate"]).dt.days
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    rfm.drop("FirstPurchaseDate", axis=1, inplace=True)
    
    return rfm


def rfm_analysis_and_segmentation(cfg: DictConfig, **kwargs):
    """Main pipeline function"""
    # --- MLflow setup ---
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiments.rfm_segmentation)

    print("Starting — MLflow tracking:", cfg.mlflow.tracking_uri)
    with mlflow.start_run(run_name=cfg.mlflow_run_names.rfm_segmentation):
        mlflow.log_params({
            "clean_parquet_file": cfg.data.paths.clean_parquet_file,
            "rfm_output_file": cfg.data.paths.rfm_output_file,
            "n_clusters": cfg.data.rfm.n_clusters,
            "n_quantiles": cfg.data.rfm.n_quantiles,
        })

        # Data loading and RFM calculation
        df = load_data(cfg.data.paths.clean_parquet_file)
        rfm, analyse_date = calculate_rfm_metrics(df)
        rfm = calculate_rfm_scores(rfm, cfg.data.rfm.n_quantiles)
        
        # Segmentation
        rfm["Segment"] = rfm["RFM_Score"].apply(segment_maker)
        
        # Clustering
        rfm, kmeans, rfm_scaled = perform_clustering(rfm, cfg.data.rfm.n_clusters)
        
        # Additional features
        rfm = add_additional_features(rfm, df, analyse_date)
        
        # Persona mapping
        persona_map = {0: 'At-Risk & Lost', 1: 'Loyal & Valuable', 2: 'Champions', 3: 'Whales / B2B'}
        rfm["Persona"] = rfm["Cluster"].map(persona_map)
        
        # Metrics
        mlflow.log_param("n_clusters", cfg.data.rfm.n_clusters)
        mlflow.log_metric("inertia", float(kmeans.inertia_))
        mlflow.log_metric("total_customers", int(len(rfm)))
        mlflow.log_metric("avg_recency", float(rfm["Recency"].mean()))
        mlflow.log_metric("avg_frequency", float(rfm["Frequency"].mean()))
        mlflow.log_metric("avg_monetary", float(rfm["Monetary"].mean()))
        
        # RFM quintile thresholds
        recency_thresholds = [float(x) for x in np.percentile(rfm["Recency"], [20, 40, 60, 80])]
        frequency_thresholds = [float(x) for x in np.percentile(rfm["Frequency"], [20, 40, 60, 80])]
        monetary_thresholds = [float(x) for x in np.percentile(rfm["Monetary"], [20, 40, 60, 80])]
        
        mlflow.log_param("recency_q1", recency_thresholds[0])
        mlflow.log_param("recency_q2", recency_thresholds[1])
        mlflow.log_param("recency_q3", recency_thresholds[2])
        mlflow.log_param("recency_q4", recency_thresholds[3])
        
        mlflow.log_param("frequency_q1", frequency_thresholds[0])
        mlflow.log_param("frequency_q2", frequency_thresholds[1])
        mlflow.log_param("frequency_q3", frequency_thresholds[2])
        mlflow.log_param("frequency_q4", frequency_thresholds[3])
        
        mlflow.log_param("monetary_q1", monetary_thresholds[0])
        mlflow.log_param("monetary_q2", monetary_thresholds[1])
        mlflow.log_param("monetary_q3", monetary_thresholds[2])
        mlflow.log_param("monetary_q4", monetary_thresholds[3])
        
        # Model saving
        sample_input = pd.DataFrame(rfm_scaled[:1], columns=["Recency", "Frequency", "Monetary"])
        mlflow.sklearn.log_model(
            sk_model=kmeans,
            artifact_path="model",
            registered_model_name="rfm_kmeans_clustering",
            input_example=sample_input
        )
        
        # Save output
        out_p = os.path.join(os.getcwd(), cfg.data.paths.rfm_output_file)
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        rfm.to_parquet(out_p, index=False)
        mlflow.log_artifact(out_p, artifact_path="rfm_outputs")
        
        print(f"✅ RFM analysis completed — saved: {out_p}")
        print(f"📊 Total customers: {len(rfm)}")
        print(f"🎯 KMeans inertia: {kmeans.inertia_:.2f}")
        print(f"📈 Average Recency: {rfm['Recency'].mean():.1f} days")
        print(f"📈 Average Frequency: {rfm['Frequency'].mean():.1f} orders")
        print(f"💰 Average Monetary: £{rfm['Monetary'].mean():.2f}")

    return str(out_p)

if __name__ == '__main__':
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    # Clear any existing Hydra instance to avoid conflicts
    GlobalHydra.instance().clear()

    # Initialize Hydra with configuration path
    initialize(version_base=None, config_path="conf")
    cfg = compose(config_name="config")

    rfm_analysis_and_segmentation(cfg)
