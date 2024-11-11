from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from my_project import data_loader, data_preprocessor, customer_segmentation, model_trainer, pricing_optimizer, price_forecaster, model_monitor

def ingest_data():
    data_loader.load_data_from_postgres()

def preprocess_data():
    data_preprocessor.clean_and_transform_data()

def segment_customers():
    customer_segmentation.perform_kmeans_clustering()

def train_model():
    model_trainer.train_regression_model()

def optimize_pricing():
    pricing_optimizer.apply_hjb_gils_optimization()

def forecast_prices():
    price_forecaster.perform_arima_forecasting()

def monitor_model():
    model_monitor.log_and_monitor_model()

with DAG('dynamic_price_optimization_pipeline', start_date=datetime(2023, 1, 1), schedule_interval='@daily', catchup=False) as dag:
    ingest_task = PythonOperator(task_id='ingest_data', python_callable=ingest_data)
    preprocess_task = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data)
    segment_task = PythonOperator(task_id='segment_customers', python_callable=segment_customers)
    train_task = PythonOperator(task_id='train_model', python_callable=train_model)
    optimize_task = PythonOperator(task_id='optimize_pricing', python_callable=optimize_pricing)
    forecast_task = PythonOperator(task_id='forecast_prices', python_callable=forecast_prices)
    monitor_task = PythonOperator(task_id='monitor_model', python_callable=monitor_model)

    # Set task dependencies
    ingest_task >> preprocess_task >> segment_task >> train_task >> optimize_task >> forecast_task >> monitor_task