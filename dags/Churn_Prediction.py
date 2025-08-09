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

# Optional: Suppress GitPython warning
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Environment variables
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow_server:5000')
EXPERIMENT_NAME     = os.getenv('MLFLOW_EXPERIMENT', 'Churn_Prediction')
DATA_DF_FILE        = os.getenv('DATA_DF_FILE', 'data/cleaned_online_retail_II.parquet')
DATA_RFM_FILE       = os.getenv('DATA_RFM_FILE', 'data/rfm_analysis.parquet')
TEST_SIZE           = float(os.getenv('TEST_SIZE', 0.2))
RANDOM_STATE        = int(os.getenv('RANDOM_STATE', 42))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


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


def split_data(X, y):
    """Train/test split."""
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def train_and_log_model(model_name: str, model_params: dict, model_class,
                        X_train, X_test, y_train, y_test):
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


def run():
    """Main pipeline."""
    with mlflow.start_run(run_name=os.getenv('MLFLOW_RUN_NAME', 'Churn_Prediction')):
        mlflow.log_params({
            'data_df_file': DATA_DF_FILE,
            'data_rfm_file': DATA_RFM_FILE,
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE
        })

        # load & prepare
        df, rfm_df = load_data(DATA_DF_FILE, DATA_RFM_FILE)
        X, y = prepare_data(df, rfm_df)
        X_train, X_test, y_train, y_test = split_data(X, y)

        # random forest
        rf_params = {
            'n_estimators': 200, 'max_depth': None,
            'random_state': RANDOM_STATE, 'n_jobs': -1
        }
        train_and_log_model('randomforest', rf_params, RandomForestClassifier,
                            X_train, X_test, y_train, y_test)

        # xgboost
        xgb_params = {
            'n_estimators': 300, 'learning_rate': 0.08, 'max_depth': 4,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE, 'n_jobs': -1, 'eval_metric': 'logloss'
        }
        train_and_log_model('xgboost', xgb_params, XGBClassifier,
                            X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    run()
