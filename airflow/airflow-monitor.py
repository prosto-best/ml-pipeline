from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from datetime import datetime
from airflow import DAG

with DAG(
    dag_id="airflow-with-kubernetes",
    schedule=None,
    start_date=datetime.now(),
    catchup=False,
    tags=["example"],
) as dag:
  airflow_with_kubernetes = KubernetesPodOperator(
    name="kubernetes_operator", 
    image="ghcr.io/prosto-best/cnyrub-monitor:latest",
    task_id="run-pod-with-kubernetes",
)

airflow_with_kubernetes.dry_run()