import httpx
import pytest

from retrieval import EmbeddingServiceError, OpenAICompatibleEmbeddingClient


def test_openai_compatible_embedding_client_orders_vectors_by_index() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://embedding.test/v1/embeddings"
        assert request.headers["authorization"] == "Bearer synthetic-secret"
        assert request.read() == b'{"model":"demo-embedding","input":["first","second"]}'
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    client = OpenAICompatibleEmbeddingClient(
        base_url="http://embedding.test/v1/",
        model="demo-embedding",
        api_key="synthetic-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.embed(("first", "second")) == ((1.0, 0.0), (0.0, 1.0))


def test_openai_compatible_embedding_client_rejects_invalid_response() -> None:
    client = OpenAICompatibleEmbeddingClient(
        base_url="http://embedding.test/v1",
        model="demo-embedding",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"data": [{"index": 0, "embedding": [float("nan")]}]},
                )
            )
        ),
    )

    with pytest.raises(EmbeddingServiceError, match="unavailable or invalid"):
        client.embed(("synthetic input",))


def test_openai_compatible_embedding_client_hides_remote_error_details() -> None:
    client = OpenAICompatibleEmbeddingClient(
        base_url="http://embedding.test/v1",
        model="demo-embedding",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    503,
                    text="remote body must not be copied into application errors",
                )
            )
        ),
    )

    with pytest.raises(EmbeddingServiceError) as error:
        client.embed(("synthetic input",))

    assert "remote body" not in str(error.value)
