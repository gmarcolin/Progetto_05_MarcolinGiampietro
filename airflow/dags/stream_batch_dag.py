from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.scripts.batch_tasks import (
    evaluate_and_log_task,
    gatekeep_and_decide_task,
    hot_reload_and_verify_task,
    ingest_batch_task,
    model_decay_branch_task,
    run_inference_task,
    trigger_retraining_task,
)
from airflow.scripts.sensors import GitHubWorkflowSensor

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
    description="Orchestrazione streaming di dati e ciclo di vita del modello",
    schedule_interval="@daily",
    start_date=datetime(2026, 9, 1),
    end_date=datetime(2026, 9, 15),
    catchup=True,
    max_active_runs=1,
) as dag:

    # 1. Ingestione del batch associato alla data logica (ds)
    task_ingest = PythonOperator(
        task_id="ingest_batch",
        python_callable=ingest_batch_task,
        provide_context=True,
    )

    # 2. Inferenza batch con retry e timeout
    task_inference = PythonOperator(
        task_id="run_inference",
        python_callable=run_inference_task,
        provide_context=True,
        retries=3,
        retry_delay=timedelta(seconds=15),
        execution_timeout=timedelta(minutes=3),
    )

    # 3. Valutazione F1 e log metriche su Hugging Face
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

    # 5. Ramo Nominale (nessun drift)
    task_pass = EmptyOperator(
        task_id="decay_not_detected",
    )

    # 6. Ramo Retraining: Upload snapshot su HF e trigger GitHub Actions
    task_trigger_retrain = PythonOperator(
        task_id="trigger_retraining",
        python_callable=trigger_retraining_task,
        provide_context=True,
    )

    # 7. SENSOR 1: Attesa retraining su GitHub
    sensor_wait_retrain = GitHubWorkflowSensor(
        task_id="wait_retrain_sensor",
        workflow_name="Model Retraining (Challenger)",
        xcom_push_key="retrain_run_id",
        mode="reschedule",
        poke_interval=30,
        timeout=1800,  # 30 min max
    )

    # 8. TASK DECISIONALE: Gatekeeper check su Golden Benchmark
    task_gatekeep = PythonOperator(
        task_id="gatekeep_and_decide",
        python_callable=gatekeep_and_decide_task,
        provide_context=True,
    )

    # 9. SENSOR 2: Attesa deploy su GitHub
    sensor_wait_deploy = GitHubWorkflowSensor(
        task_id="wait_deploy_sensor",
        workflow_name="Model Deployment (Champion Promotion)",
        xcom_push_key="deploy_run_id",
        check_deploy_gate=True,
        mode="reschedule",
        poke_interval=20,
        timeout=600,  # 10 min max
    )

    # 10. HOT-RELOAD & VERIFY: Chiamata POST /reload, check /health e rollback
    task_hot_reload = PythonOperator(
        task_id="hot_reload_and_verify",
        python_callable=hot_reload_and_verify_task,
        provide_context=True,
    )

    # Definizione del grafo delle dipendenze
    task_ingest >> task_inference >> task_evaluate >> task_branch
    task_branch >> task_pass
    task_branch >> task_trigger_retrain >> sensor_wait_retrain >> task_gatekeep >> sensor_wait_deploy >> task_hot_reload
