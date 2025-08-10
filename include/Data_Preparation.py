import os
from pathlib import Path
from shutil import copyfile
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import urllib.request
from omegaconf import DictConfig

def get_spark(app_name: str = "RetailDataCleaning") -> SparkSession:
    """Creates a Spark session."""
    return SparkSession.builder.appName(app_name).getOrCreate()

def download_file(url: str, output_path: str) -> None:
    """
    Downloads a file from the given URL and saves it to output_path.
    """
    print(f"Downloading from {url} ...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")

def load_raw_data(spark: SparkSession, path: str):
    """Loads the raw data."""
    # In Astro Airflow, use absolute paths
    if not os.path.isabs(path):
        p = os.path.join("/usr/local/airflow", path)
    else:
        p = path
    
    # Always download the file fresh from URL
    url = "https://dagshub.com/mertafacan/E-Commerce-Customer-Analytics-MLOps-Pipeline/raw/main/data/online_retail_II.csv"
    print(f"Downloading fresh data from {url}")
    
    # Create data directory if it doesn't exist
    data_dir = "/usr/local/airflow/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Download to data directory
    target_path = os.path.join(data_dir, "online_retail_II.csv")
    try:
        download_file(url, target_path)
        # Update the path to point to the downloaded file
        p = target_path
    except Exception as e:
        raise FileNotFoundError(f"Could not download file from {url}. Error: {e}")
    
    # Reads the CSV file with header and schema inference
    return spark.read.option("header", True).option("inferSchema", True).csv(p)

def clean_data(df, min_qty: int, min_price: float):
    """Cleans the data: filters by minimum quantity and price, drops missing customer IDs and duplicates."""
    df = df.filter((col("Quantity") > min_qty) & (col("Price") > min_price))
    df = df.na.drop(subset=["Customer ID"])
    df = df.dropDuplicates()
    df = df.withColumn("Customer ID", col("Customer ID").cast("int"))
    df = df.withColumn("InvoiceDate", col("InvoiceDate").cast("timestamp"))
    return df

def add_total_price(df):
    """Adds the total price column."""
    return df.withColumn("TotalPrice", col("Quantity") * col("Price"))

def write_single_parquet(df, output_dir: str, single_output: str) -> str:
    """Writes data to a single Parquet file."""
    df.coalesce(1).write.mode("overwrite").parquet(output_dir)
    out_dir = os.path.join(os.getcwd(), output_dir)
    part_files = list(Path(out_dir).glob("part-*.parquet"))
    if not part_files:
        raise FileNotFoundError("No Parquet part found")
    target = os.path.join(os.getcwd(), single_output)
    copyfile(part_files[0], target)
    return target

def data_preparation(cfg: DictConfig, **kwargs):
    """Main pipeline function: executes data reading, cleaning, transformation, and saving operations."""
    
    # Set MLflow tracking URI and experiment name
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiments.data_preparation)

    with mlflow.start_run(run_name=cfg.mlflow_run_names.data_preparation):
        mlflow.log_params({
            "raw_data_path": cfg.data.paths.raw_data,
            "clean_parquet_dir": cfg.data.paths.clean_parquet_dir,
            "clean_parquet_file": cfg.data.paths.clean_parquet_file,
            "min_quantity": cfg.data.cleaning.min_quantity,
            "min_price": cfg.data.cleaning.min_price,
        })
        
        spark = get_spark()
        spark.sparkContext.setLogLevel("WARN")

        df = load_raw_data(spark, cfg.data.paths.raw_data)
        df = clean_data(df, cfg.data.cleaning.min_quantity, cfg.data.cleaning.min_price)
        df = add_total_price(df)
        single_path = write_single_parquet(df, cfg.data.paths.clean_parquet_dir, cfg.data.paths.clean_parquet_file)

        mlflow.log_metric("total_records", int(df.count()))
        mlflow.log_artifact(single_path, artifact_path="data_outputs")

        print(f"✅ Data preparation completed — saved: {single_path}")
        print(f"📊 Total records: {df.count()}")

    return str(single_path)

if __name__ == '__main__':
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    # Clear any existing Hydra instance to avoid conflicts
    GlobalHydra.instance().clear()

    # Initialize Hydra with configuration path
    initialize(version_base=None, config_path="conf")
    cfg = compose(config_name="config")

    data_preparation(cfg)
