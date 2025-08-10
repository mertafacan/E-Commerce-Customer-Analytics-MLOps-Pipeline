import os
import datetime as dt
import time
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from mlflow.models import infer_signature
from omegaconf import DictConfig

# Optional: Suppress GitPython warning
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")


def load_data(path_df: str, path_rfm: str):
    """Load up the main and RFM parquet files."""
    df = pd.read_parquet(path_df)
    rfm_df = pd.read_parquet(path_rfm)
    return df, rfm_df


def prepare_data(df: pd.DataFrame, rfm_df: pd.DataFrame):
    """Convert dates, compute churn flag, merge and engineer features."""
    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    analyse_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm_df['Churn'] = (rfm_df['Recency'] > 90).astype(int)

    first_purchase = (
        df.groupby('Customer ID')['InvoiceDate']
        .min().reset_index()
        .rename(columns={'InvoiceDate': 'FirstPurchaseDate'})
    )

    rfm = (
        rfm_df
        .merge(first_purchase, on='Customer ID', how='left')
        .assign(
            Tenure=lambda x: (analyse_date - x['FirstPurchaseDate']).dt.days,
            AvgOrderValue=lambda x: x['Monetary'] / x['Frequency']
        )
    )

    drop_cols = [
        'Customer ID', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score',
        'Segment', 'Cluster', 'Churn', 'Recency', 'Persona', 'FirstPurchaseDate'
    ]
    X = rfm.drop(columns=[c for c in drop_cols if c in rfm.columns])
    y = rfm['Churn']
    return X, y


def split_data(X, y, test_size: float, random_state: int):
    """Train/test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def train_and_log_model(model_name: str, model_params: dict, model_class,
                        X_train, X_test, y_train, y_test, test_size: float, random_state: int):
    """Train a model pipeline, log metrics and model with MLflow."""
    with mlflow.start_run(run_name=f'{model_name}_Churn',
                          nested=(mlflow.active_run() is not None)):
        start_time = time.time()
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model_class(**model_params))
        ])
        pipe.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        print(f'{model_name} AUC: {auc:.4f} | ACC: {acc:.4f}')

        # log params & metrics
        mlflow.log_params(model_params)
        mlflow.log_param('model_type', model_name)
        mlflow.log_metrics({'roc_auc': auc, 'accuracy': acc, 'train_time_sec': train_time})

        # log model
        example_df = X_test.iloc[:1].copy()
        signature = infer_signature(example_df, pipe.predict_proba(example_df))

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path='model',
            registered_model_name=f'churn_{model_name.lower()}',
            signature=signature,
            input_example=example_df
        )


def train_churn_model(cfg: DictConfig, **kwargs):
    """Main pipeline for churn prediction model training."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiments.churn_prediction)

    with mlflow.start_run(run_name=cfg.mlflow_run_names.churn_prediction):
        mlflow.log_params({
            'data_df_file': cfg.data.paths.clean_parquet_file,
            'data_rfm_file': cfg.data.paths.rfm_output_file,
            'test_size': cfg.model.test_size,
            'random_state': cfg.model.random_state
        })

        # load & prepare
        df, rfm_df = load_data(cfg.data.paths.clean_parquet_file, cfg.data.paths.rfm_output_file)
        X, y = prepare_data(df, rfm_df)
        X_train, X_test, y_train, y_test = split_data(X, y, cfg.model.test_size, cfg.model.random_state)

        # random forest
        rf_params = {
            'n_estimators': 200, 'max_depth': None,
            'random_state': cfg.model.random_state, 'n_jobs': -1
        }
        train_and_log_model('randomforest', rf_params, RandomForestClassifier,
                            X_train, X_test, y_train, y_test, cfg.model.test_size, cfg.model.random_state)

        # xgboost
        xgb_params = {
            'n_estimators': 300, 'learning_rate': 0.08, 'max_depth': 4,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'random_state': cfg.model.random_state, 'n_jobs': -1, 'eval_metric': 'logloss'
        }
        train_and_log_model('xgboost', xgb_params, XGBClassifier,
                            X_train, X_test, y_train, y_test, cfg.model.test_size, cfg.model.random_state)


if __name__ == '__main__':
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    # Clear any existing Hydra instance to avoid conflicts
    GlobalHydra.instance().clear()

    # Initialize Hydra with configuration path
    initialize(version_base=None, config_path="conf")
    cfg = compose(config_name="config")

    train_churn_model(cfg)
