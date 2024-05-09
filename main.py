### Import libraries ###
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
import uvicorn
import yaml
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.model import Model

# Libraries needed for colab
try:
    import nest_asyncio
    from google.colab import userdata
    from pyngrok import conf, ngrok
except:
    pass

### Initialization ###
# with open("conf/base/model.yaml") as f:
#     global conf
#     conf = yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for application lifespan. Initializes the model
    upon application startup and cleans up resources upon shutdown.

    Args:
    app (FastAPI): The FastAPI application instance.

    Raises:
    RuntimeError: Raised if the `model_path` is not set in the configuration.
    """
    global model
    global conf
    with open("conf/base/model.yaml") as f:
        conf = yaml.safe_load(f)
        logger.info("Loaded config.")
    model_path = conf["save_model_path"]
    if model_path is None:
        raise RuntimeError("model_path environment variable not set")
    model = Model(model_path)
    yield


app = FastAPI(title="Chatbot", lifespan=lifespan)

ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
utils.setup_logging(logging_config_path="conf/base/logging.yaml")


### APIs ###
class GenerationRequest(BaseModel):
    """Request body for the /generate API endpoint"""

    prompt: str
    temperature: float = 0.1
    max_length: int = 250
    top_k: int = 50
    top_p: float = 0.9
    num_return_sequences: int = 1


@app.post("/response/")
async def response(request: GenerationRequest) -> dict:
    """
    Process the generation request and return the response.

    Args:
    request (GenerationRequest): The request object containing the prompt,
    max length, and number of return sequences.

    Returns:
    dict: A dictionary containing the generated response.

    Raises:
    HTTPException: Raised if the prompt is empty.
    """
    if request.prompt:
        response = model.generate_response(
            request.prompt,
            request.temperature,
            request.top_k,
            request.top_p,
        )
        return {"response": response}
    else:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")


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
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    # to run on colab
    # os.environ["NGROK"] = userdata.get("NGROK")
    # conf.get_default().auth_token = os.environ["NGROK"]
    # ngrok.kill()
    # ngrok_tunnel = ngrok.connect(8000)
    # print("Public URL:", ngrok_tunnel.public_url)

    # to run on colab multithreading
    # threading.Thread(target=run_fastapi, daemon=True).start()
    # os.system("streamlit run src/streamlit_chat.py")

    # nest_asyncio.apply()
    # uvicorn.run(app, port=8000)

    # run from local PC with GPU
    run_fastapi()
