from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.scripts.batch_tasks import (
    evaluate_and_log_task,
    ingest_batch_task,
    model_decay_branch_task,
    run_inference_task,
    trigger_retraining_task,
)

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email": ["alert-mlops@example.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(seconds=15),
}

with DAG(
    dag_id="sentiment_stream_batch_monitoring",
    default_args=default_args,
    description="Pipeline di monitoraggio continuo batch-per-batch con controllo del drift",
    schedule_interval="@daily",
    start_date=datetime(2026, 9, 1),
    end_date=datetime(2026, 9, 15),
    catchup=False,
    max_active_runs=1,
) as dag:

    # 1. Ingestione del batch associato alla data logica (ds)
    task_ingest = PythonOperator(
        task_id="ingest_batch",
        python_callable=ingest_batch_task,
        provide_context=True,
    )

    # 2. Chiamata all'API /predict con policy di retry strette e timeout
    task_inference = PythonOperator(
        task_id="run_inference",
        python_callable=run_inference_task,
        provide_context=True,
        retries=3,
        retry_delay=timedelta(seconds=15),
        execution_timeout=timedelta(minutes=3),
    )

    # 3. Calcolo metriche F1 pesato e logging su Hugging Face
    task_evaluate = PythonOperator(
        task_id="evaluate_metrics",
        python_callable=evaluate_and_log_task,
        provide_context=True,
    )

    # 4. Operatore condizionale: Bivio sul Model Decay (F1 < soglia)
    task_branch = BranchPythonOperator(
        task_id="branch_decay_check",
        python_callable=model_decay_branch_task,
        provide_context=True,
    )

    # 5a. Ramo A: Decay rilevato -> Upload snapshot su HF e trigger GitHub Actions
    task_trigger = PythonOperator(
        task_id="trigger_retraining",
        python_callable=trigger_retraining_task,
        provide_context=True,
    )

    # 5b. Ramo B: Performance nominali -> No-op
    task_pass = EmptyOperator(
        task_id="decay_not_detected",
    )

    # Definizione delle dipendenze del grafo
    task_ingest >> task_inference >> task_evaluate >> task_branch >> [task_trigger, task_pass]
