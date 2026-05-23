"""
IntelliPipe Airflow DAGs
=========================
Three production DAGs:
1. intellipipe_dq_pipeline    — hourly DQ checks + anomaly scoring
2. intellipipe_model_retrain  — daily model retraining + shadow eval
3. intellipipe_rag_ingest     — daily dbt doc ingestion into pgvector
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner": "data-platform",
    "depends_on_past": False,
    "email": ["data-alerts@intellipipe.io"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# ─────────────────────────────────────────
# DAG 1: Hourly DQ Pipeline
# ─────────────────────────────────────────

def run_ge_validation(**ctx):
    """Run Great Expectations suites against latest data."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "src.quality.ge_runner", "--table", "raw_orders"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"GE validation failed:\n{result.stderr}")
    print(result.stdout)


def run_anomaly_scoring(**ctx):
    """Score latest batch with ensemble anomaly detectors."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "src.anomaly.batch_scorer", "--table", "raw_orders"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Anomaly scoring failed:\n{result.stderr}")
    print(result.stdout)


def process_alert_queue(**ctx):
    """Drain Redis alert queue and trigger LLM agent for each alert."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "src.agents.alert_processor"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Alert processing failed:\n{result.stderr}")
    print(result.stdout)


def update_sla_trends(**ctx):
    """Aggregate SLA trend data for the current hour."""
    import subprocess
    subprocess.run(
        ["python", "-m", "src.quality.sla_tracker", "--granularity", "hourly"],
        check=True
    )


with DAG(
    "intellipipe_dq_pipeline",
    default_args=DEFAULT_ARGS,
    description="Hourly data quality validation and anomaly detection",
    schedule_interval="0 * * * *",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["intellipipe", "data-quality", "production"],
) as dq_dag:

    ge_validation = PythonOperator(
        task_id="run_ge_validation",
        python_callable=run_ge_validation,
        doc_md="Run Great Expectations suites against raw_orders",
    )

    anomaly_scoring = PythonOperator(
        task_id="run_anomaly_scoring",
        python_callable=run_anomaly_scoring,
        doc_md="Score batch with IF+AE+ZScore ensemble",
    )

    dbt_run = BashOperator(
        task_id="dbt_run_staging",
        bash_command="cd /opt/intellipipe/dbt && dbt run --select staging --profiles-dir .",
        doc_md="Run dbt staging models",
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command="cd /opt/intellipipe/dbt && dbt test --select staging --profiles-dir .",
        doc_md="Run dbt tests",
    )

    alert_processing = PythonOperator(
        task_id="process_alert_queue",
        python_callable=process_alert_queue,
        doc_md="Drain Redis alerts → LLM agent → GitHub PR + Jira + Slack",
    )

    sla_update = PythonOperator(
        task_id="update_sla_trends",
        python_callable=update_sla_trends,
        doc_md="Aggregate hourly SLA trend metrics",
    )

    # Task dependencies
    ge_validation >> anomaly_scoring >> alert_processing
    ge_validation >> dbt_run >> dbt_test
    [alert_processing, dbt_test] >> sla_update


# ─────────────────────────────────────────
# DAG 2: Daily Model Retraining
# ─────────────────────────────────────────

def retrain_isolation_forest(**ctx):
    import subprocess
    subprocess.run(
        ["python", "-m", "src.anomaly.trainer", "--model", "isolation_forest"],
        check=True
    )


def retrain_autoencoder(**ctx):
    import subprocess
    subprocess.run(
        ["python", "-m", "src.anomaly.trainer", "--model", "autoencoder"],
        check=True
    )


def run_shadow_evaluation(**ctx):
    """Compare shadow model vs production model on held-out data."""
    import subprocess
    subprocess.run(
        ["python", "-m", "src.anomaly.shadow_eval", "--promote-if-better"],
        check=True
    )


def update_model_registry(**ctx):
    """Promote best model to production in MLflow Model Registry."""
    import subprocess
    subprocess.run(
        ["python", "-m", "src.anomaly.model_registry", "--action", "promote"],
        check=True
    )


with DAG(
    "intellipipe_model_retrain",
    default_args=DEFAULT_ARGS,
    description="Daily anomaly model retraining with shadow evaluation",
    schedule_interval="0 2 * * *",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["intellipipe", "ml", "production"],
) as retrain_dag:

    t_if = PythonOperator(task_id="retrain_isolation_forest", python_callable=retrain_isolation_forest)
    t_ae = PythonOperator(task_id="retrain_autoencoder", python_callable=retrain_autoencoder)
    t_shadow = PythonOperator(task_id="shadow_evaluation", python_callable=run_shadow_evaluation)
    t_registry = PythonOperator(task_id="update_model_registry", python_callable=update_model_registry)

    [t_if, t_ae] >> t_shadow >> t_registry


# ─────────────────────────────────────────
# DAG 3: Daily RAG Ingestion
# ─────────────────────────────────────────

def ingest_dbt_docs(**ctx):
    import subprocess
    subprocess.run(
        ["python", "-m", "src.rag.ingestor", "--source", "dbt", "--project-path", "/opt/intellipipe/dbt"],
        check=True
    )


def rebuild_vector_index(**ctx):
    """Vacuum and reindex the pgvector HNSW index after bulk ingestion."""
    import subprocess
    subprocess.run(
        ["python", "-m", "src.rag.index_manager", "--action", "reindex"],
        check=True
    )


with DAG(
    "intellipipe_rag_ingest",
    default_args=DEFAULT_ARGS,
    description="Daily RAG document ingestion from dbt project",
    schedule_interval="0 3 * * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["intellipipe", "rag", "production"],
) as rag_dag:

    dbt_docs_compile = BashOperator(
        task_id="dbt_docs_generate",
        bash_command="cd /opt/intellipipe/dbt && dbt docs generate --profiles-dir .",
    )

    ingest = PythonOperator(task_id="ingest_dbt_docs", python_callable=ingest_dbt_docs)
    reindex = PythonOperator(task_id="rebuild_vector_index", python_callable=rebuild_vector_index)

    dbt_docs_compile >> ingest >> reindex
