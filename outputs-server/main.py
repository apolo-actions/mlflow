from contextlib import asynccontextmanager
import logging
import os

import requests
import uvicorn
from fastapi import Depends, FastAPI

from apolo_app_types.dynamic_outputs import (
    DynamicAppBasicResponse,
    DynamicAppFilterParams,
    DynamicAppListResponse,
    DynamicAppIdResponse,
    FilterOperator,
)

from filters import ModelFilter

server_variables = {}
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        MAIN_APP_DEPLOYMENT_NAME = os.environ["MAIN_APP_DEPLOYMENT_NAME"]
        MAIN_APP_DEPLOYMENT_NAMESPACE = os.environ["MAIN_APP_DEPLOYMENT_NAMESPACE"]
        MLFLOW_PORT = os.environ["MLFLOW_PORT"]
        MLFLOW_URL = f"http://{MAIN_APP_DEPLOYMENT_NAME}.{MAIN_APP_DEPLOYMENT_NAMESPACE}.svc.cluster.local:{MLFLOW_PORT}"
        server_variables['MLFLOW_URL'] = MLFLOW_URL
        logger.info(f"MLFLOW_URL set to {MLFLOW_URL}")
    except KeyError as e:
        raise RuntimeError(f"Environment variable {e} is not set") from e
    yield


app = FastAPI(lifespan=lifespan)


@app.get('/')
@app.get('/health')
@app.get('/healthz')
async def root() -> DynamicAppBasicResponse:
    return DynamicAppBasicResponse(status="healthy")


def build_mlflow_filter(model_filter: ModelFilter) -> str | None:
    """Build MLflow API filter string from ModelFilter conditions.

    MLflow supports SQL-like filter syntax:
    - name LIKE '%value%'
    - name = 'value'

    Args:
        model_filter: ModelFilter with parsed conditions

    Returns:
        MLflow filter string or None if no API-compatible filters
    """
    mlflow_conditions = []

    for condition in model_filter.conditions:
        if condition.field == "name":
            if condition.operator == FilterOperator.LIKE:
                mlflow_conditions.append(f"name LIKE '%{condition.value}%'")
            elif condition.operator == FilterOperator.EQ:
                mlflow_conditions.append(f"name = '{condition.value}'")

    if mlflow_conditions:
        return " AND ".join(mlflow_conditions)
    return None


def get_local_conditions(model_filter: ModelFilter) -> list:
    """Get conditions that must be applied locally (not supported by MLflow API).

    Returns conditions for fields other than name with LIKE/EQ operators.
    """
    local_conditions = []

    for condition in model_filter.conditions:
        # Name with LIKE/EQ is handled by MLflow API
        if condition.field == "name" and condition.operator in (
            FilterOperator.LIKE,
            FilterOperator.EQ,
        ):
            continue
        local_conditions.append(condition)

    return local_conditions


@app.get('/output')
async def get_outputs(
    params: DynamicAppFilterParams = Depends(),
) -> DynamicAppListResponse:
    try:
        # Build request params
        request_params = {}

        # Parse and apply filters
        model_filter = ModelFilter(params.filter) if params.filter else None

        if model_filter:
            # Build MLflow API filter for name conditions
            mlflow_filter = build_mlflow_filter(model_filter)
            if mlflow_filter:
                request_params['filter'] = mlflow_filter

        # Fetch from MLflow
        res = requests.get(
            server_variables['MLFLOW_URL'] + '/api/2.0/mlflow/registered-models/search',
            params=request_params,
        )
        res.raise_for_status()
        json_response = res.json()
        models = json_response.get("registered_models", [])

        # Apply local filtering for conditions not supported by MLflow API
        if model_filter:
            local_conditions = get_local_conditions(model_filter)
            if local_conditions:
                # Create a temporary filter with only local conditions
                temp_filter = ModelFilter(None)
                temp_filter.conditions = local_conditions
                models = temp_filter.apply(models)

        # Apply pagination
        models = models[params.offset : params.offset + params.limit]

        # Transform to response format
        data = [
            DynamicAppIdResponse(id=model.get("name", ""), value=model)
            for model in models
        ]

        return DynamicAppListResponse(status="success", data=data)
    except requests.RequestException as e:
        logger.error(f"MLflow request failed: {e}")
        return DynamicAppListResponse(status="error", data=None)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
