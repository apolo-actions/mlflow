from contextlib import asynccontextmanager
import logging
import os
from typing import Any, Dict, List
import requests
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from filters import ModelFilter

server_variables = {}
logger = logging.getLogger(__name__)

class BasicResponse(BaseModel):
    status: str
    data: Dict[str, Any] | List[Dict[str, Any]] | None = None

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
async def root() -> BasicResponse:
    return BasicResponse(status="healthy")

@app.get('/output')
async def get_outputs(filter: str | None = None):
    try:
        res = requests.get(server_variables['MLFLOW_URL'] + '/api/2.0/mlflow/registered-models/search')
        res.raise_for_status()
        json_response = res.json()
        models = json_response.get("registered_models", [])

        # Apply filtering if filter parameter is provided
        if filter:
            model_filter = ModelFilter(filter)
            models = model_filter.apply(models)

        return BasicResponse(status="success", data=models)
    except requests.RequestException as e:
        return {'error': str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)