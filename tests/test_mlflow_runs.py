import mlflow.experiments
import pytest
import mlflow

from mlflow import (
    active_run,
    end_run,
    get_run,
    start_run,
)

mlflow.set_tracking_uri("http://localhost:8080")


def test_create_mlflow_run():
    """Test creating a new MLflow run."""
    with start_run() as run:
        assert active_run() is not None
        assert run.info.status == "RUNNING"
        
def test_end_mlflow_run():
    """Test ending an MLflow run and verifying its status."""
    run = start_run()
    run_id = run.info.run_id
    end_run()
    ended_run = get_run(run_id)
    assert ended_run.info.status == "FINISHED"

def test_nested_runs():
    """Test nested MLflow runs functionality."""
    with start_run() as parent_run:
        with start_run(nested=True) as child_run:
            assert active_run() is not None
            assert child_run.info.run_id != parent_run.info.run_id

def test_log_params_and_metrics():
    with start_run() as run:
        mlflow.log_param("param1", "value1")
        mlflow.log_metric("metric1", 1.0)
        
        run_id = run.info.run_id
        fetched_run = get_run(run_id)
        
        assert fetched_run.data.params["param1"] == "value1"
        assert fetched_run.data.metrics["metric1"] == 1.0

def test_tags():
    with start_run() as run:
        mlflow.set_tag("tag1", "value1")
        run_id = run.info.run_id
        
    fetched_run = get_run(run_id)
    assert fetched_run.data.tags["tag1"] == "value1"