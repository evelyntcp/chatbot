### Import libraries ###
import gc
import logging
import os
import subprocess
import threading
from contextlib import asynccontextmanager
from typing import Optional

import fastapi

# Consider next time if need to add __init__.py into src
import src.utils as utils
import streamlit as st
import torch
import uvicorn
import yaml
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src import model, rag

# Libraries needed for colab
try:
    import nest_asyncio
    from google.colab import userdata
    from pyngrok import conf, ngrok
except:
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for application lifespan. Initializes the model
    upon application startup and cleans up resources upon shutdown.

    Upon startup, it reads the model configuration from 'conf/base/model.yaml', initializes
    the specified machine learning model, and assigns this model to `app.model` for global access.

    Args:
    app (FastAPI): The FastAPI application instance.

    Raises:
    RuntimeError: Raised if the `model_path` is not set in the configuration.
    """
    with open("conf/base/model.yaml") as f:
        app.conf = yaml.safe_load(f)
        logger.info("Loaded config.")
    model_path = app.conf["save_model_path"]
    model_id = app.conf["model_id"]

    if model_path is None:
        raise RuntimeError("Model path not set.")
    app.model = model.Model(model_id, model_path)
    yield
    # del app.model
    # # os.rmdir(model_path)
    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()


app = FastAPI(title="Chatbot", lifespan=lifespan)
# app = FastAPI(title="Chatbot")

ORIGINS = ["*"]
# ORIGINS = ["http://localhost:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model.router)
app.include_router(rag.router)

logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
utils.setup_logging(logging_config_path="conf/base/logging.yaml")


### APIs ###
class HealthCheck(BaseModel):
    """Model for the health check response."""

    status: str = "OK"


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """Returns a health check status."""
    return HealthCheck(status="OK")


def run_fastapi():
    """Runs FastAPI app on host 0.0.0.0 and port 8000."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_fastapi()
