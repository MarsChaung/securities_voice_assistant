FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md alembic.ini ./
COPY services ./services
COPY packages ./packages
COPY knowledge ./knowledge

RUN uv sync --frozen --no-dev

EXPOSE 8080 8081

CMD ["uvicorn", "orchestrator.api:app", "--host", "0.0.0.0", "--port", "8080"]
