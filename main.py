import fastapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chat
import config

app = fastapi.FastAPI(title=config.SETTINGS.API_NAME)

ORIGINS =  ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,  
    allow_methods=["*"],
    allow_headers=[
        "*"
    ],
)

app.include_router(chat.router)

# can move to utils
class HealthCheck(BaseModel):
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
    return HealthCheck(status="OK")

if __name__ == "__main__":
    import subprocess
    import uvicorn
    from main import app

    command = [
        "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]

    subprocess.run(command)