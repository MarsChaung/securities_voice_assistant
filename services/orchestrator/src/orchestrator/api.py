from fastapi import FastAPI

from answer_contract import TurnRequest, TurnResponse
from observability import configure_logging

from .config import get_settings
from .service import TurnService

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="Securities Voice Assistant Orchestrator",
    version="0.1.0",
    description="證券知識型語音客服的安全決策基礎 API",
)
service = TurnService()


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.post("/v1/turns/evaluate", response_model=TurnResponse)
def evaluate_turn(request: TurnRequest) -> TurnResponse:
    return service.evaluate(request)
