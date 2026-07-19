import math
from collections.abc import Sequence
from typing import Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

EmbeddingVector = tuple[float, ...]


class EmbeddingServiceError(RuntimeError):
    """Embedding 服務無法安全產生可比較的向量。"""


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]: ...


class _EmbeddingData(BaseModel):
    index: int = Field(ge=0)
    embedding: list[float] = Field(min_length=1)


class _EmbeddingResponse(BaseModel):
    data: list[_EmbeddingData]


class OpenAICompatibleEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 5.0,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("hybrid retrieval 必須指定 embedding model")
        self._endpoint = f"{base_url.rstrip('/')}/embeddings"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def embed(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        if not texts:
            return ()

        try:
            response = self._client.post(
                self._endpoint,
                json={"model": self._model, "input": list(texts)},
                headers=self._headers,
            )
            response.raise_for_status()
            payload = _EmbeddingResponse.model_validate(response.json())
        except (httpx.HTTPError, ValidationError, ValueError, TypeError):
            raise EmbeddingServiceError("embedding service unavailable or invalid") from None

        ordered = sorted(payload.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise EmbeddingServiceError("embedding service returned incomplete vectors")

        vectors = tuple(tuple(item.embedding) for item in ordered)
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or any(
            not math.isfinite(value) for vector in vectors for value in vector
        ):
            raise EmbeddingServiceError("embedding service returned invalid vectors")
        return vectors
