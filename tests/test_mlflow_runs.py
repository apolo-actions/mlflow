import mlflow.experiments
import pytest
import mlflow
from sklearn.linear_model import LinearRegression
import numpy as np

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

def test_log_artifact():
    """Test logging and retrieving an artifact."""
    with start_run() as run:
        content = "artifact content"
        with open("test_artifact.txt", "w") as f:
            f.write(content)
        mlflow.log_artifact("test_artifact.txt")
        
        run_id = run.info.run_id
        artifact_uri = mlflow.get_artifact_uri("test_artifact.txt")
        assert "test_artifact.txt" in artifact_uri

def test_register_model():
    """Test registering a model."""
    
    with start_run() as run:
        X = np.array([[1], [2]])
        y = np.array([1, 2])
        model = LinearRegression()
        model.fit(X, y)
        
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, "test_model")
        
        assert registered_model.name == "test_model"
        assert registered_model.version == '1'

        loaded_model = mlflow.sklearn.load_model(f"models:/{registered_model.name}/1")
        prediction = loaded_model.predict([[3]])
        assert isinstance(loaded_model, LinearRegression)
        assert prediction.shape == (1,)
