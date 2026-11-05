import json
from pathlib import Path

import httpx2
import pytest
from typesafe_sdk import (
    RetryPolicy,
    SystemOneResponse,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
)

from jev_compiler_eval.jev import (
    JevApiClient,
    JevCacheError,
    JevClientConfig,
    JevClientError,
)


class FakeSdkClient:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def system_one(self, state: object, questions: object) -> SystemOneResponse:
        self.calls += 1
        return SystemOneResponse.model_validate(
            {
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "answers": {
                    "decision": {
                        "type": "choice",
                        "choice": "candidate-b",
                        "confidence": 0.8,
                        "probabilities": {
                            "candidate-a": 0.2,
                            "candidate-b": 0.8,
                        },
                    }
                },
            }
        )

    def close(self) -> None:
        self.closed = True


class TimeoutSdkClient:
    def system_one(self, state: object, questions: object) -> SystemOneResponse:
        raise TypeSafeAPITimeoutError(5.0)


class AuthenticationErrorSdkClient:
    def system_one(self, state: object, questions: object) -> SystemOneResponse:
        raise TypeSafeAuthenticationError(
            401,
            {"detail": "secret server response"},
            httpx2.Headers({"x-typesafe-request-id": "request-1"}),
        )


def make_config(tmp_path: Path) -> JevClientConfig:
    return JevClientConfig(
        model="jev-1.13.0",
        cache_dir=tmp_path / "cache",
        max_retries=3,
        request_timeout_seconds=5.0,
    )


def request() -> dict[str, object]:
    return {
        "state": {"operation": "matmul"},
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "Choose the fastest candidate.",
                "criteria": {
                    "candidate-a": {"tile": 8},
                    "candidate-b": {"tile": 16},
                },
            }
        },
    }


def test_calls_sdk_and_caches_successful_response(tmp_path: Path) -> None:
    sdk_client = FakeSdkClient()
    client = JevApiClient(
        make_config(tmp_path),
        sdk_client=sdk_client,  # type: ignore[arg-type]
    )
    values = request()

    first = client.decide(**values)  # type: ignore[arg-type]
    second = client.decide(**values)  # type: ignore[arg-type]

    assert sdk_client.calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.model == "jev-1.13.0"
    assert first.answers["decision"]["choice"] == "candidate-b"
    assert first.input_tokens == 100
    assert first.output_tokens == 10
    assert second.answers == first.answers
    assert len(list(make_config(tmp_path).cache_dir.glob("*.json"))) == 1


def test_configures_official_sdk_retry_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sdk_client = FakeSdkClient()

    def make_sdk_client(**kwargs: object) -> FakeSdkClient:
        captured.update(kwargs)
        return sdk_client

    monkeypatch.setattr(
        "jev_compiler_eval.jev.client.TypeSafeClient",
        make_sdk_client,
    )

    with JevApiClient(make_config(tmp_path)):
        pass

    retry = captured["retry"]
    assert isinstance(retry, RetryPolicy)
    assert retry.max_retries == 3
    assert 429 in retry.http_statuses
    assert 529 in retry.http_statuses
    assert captured["model"] == "jev-1.13.0"
    assert captured["timeout"] == 5.0
    assert captured["base_url"] == "https://api.typesafe.ai"
    assert sdk_client.closed is True


def test_cache_key_is_independent_of_mapping_order(tmp_path: Path) -> None:
    sdk_client = FakeSdkClient()
    client = JevApiClient(
        make_config(tmp_path),
        sdk_client=sdk_client,  # type: ignore[arg-type]
    )
    client.decide(
        state={"operation": "matmul", "dtype": "float32"},
        questions={"decision": {"type": "noul", "instructions": "Relevant?"}},
    )

    result = client.decide(
        state={"dtype": "float32", "operation": "matmul"},
        questions={"decision": {"instructions": "Relevant?", "type": "noul"}},
    )

    assert result.cache_hit is True
    assert sdk_client.calls == 1


def test_cache_stores_request_body_with_response(tmp_path: Path) -> None:
    client = JevApiClient(
        make_config(tmp_path),
        sdk_client=FakeSdkClient(),  # type: ignore[arg-type]
    )
    values = request()

    client.decide(**values)  # type: ignore[arg-type]

    cache_path = next(make_config(tmp_path).cache_dir.glob("*.json"))
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cached["schema_version"] == 2
    assert cached["request_sha256"] == cache_path.stem
    assert cached["request"] == {
        "base_url": "https://api.typesafe.ai",
        "model": "jev-1.13.0",
        **values,
    }
    assert cached["response"]["request_id"] is None


def test_rejects_cache_whose_request_does_not_match_its_hash(tmp_path: Path) -> None:
    client = JevApiClient(
        make_config(tmp_path),
        sdk_client=FakeSdkClient(),  # type: ignore[arg-type]
    )
    values = request()
    client.decide(**values)  # type: ignore[arg-type]
    cache_path = next(make_config(tmp_path).cache_dir.glob("*.json"))
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    cached["request"]["state"] = {"operation": "conv2d"}
    cache_path.write_text(json.dumps(cached), encoding="utf-8")

    with pytest.raises(JevCacheError, match="failed to read Jev cache file"):
        client.decide(**values)  # type: ignore[arg-type]


def test_cache_does_not_contain_api_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-test-key")
    client = JevApiClient(
        make_config(tmp_path),
        sdk_client=FakeSdkClient(),  # type: ignore[arg-type]
    )

    client.decide(**request())  # type: ignore[arg-type]

    cache_text = next(make_config(tmp_path).cache_dir.glob("*.json")).read_text()
    assert "secret-test-key" not in cache_text


def test_rejects_corrupted_cache(tmp_path: Path) -> None:
    client = JevApiClient(
        make_config(tmp_path),
        sdk_client=FakeSdkClient(),  # type: ignore[arg-type]
    )
    values = request()
    client.decide(**values)  # type: ignore[arg-type]
    cache_path = next(make_config(tmp_path).cache_dir.glob("*.json"))
    cache_path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

    with pytest.raises(JevCacheError, match="failed to read Jev cache file"):
        client.decide(**values)  # type: ignore[arg-type]


def test_converts_sdk_timeout_to_project_error(tmp_path: Path) -> None:
    client = JevApiClient(
        make_config(tmp_path),
        sdk_client=TimeoutSdkClient(),  # type: ignore[arg-type]
    )

    with pytest.raises(JevClientError, match="Jev API request timed out"):
        client.decide(**request())  # type: ignore[arg-type]


def test_converts_sdk_api_error_without_exposing_response_body(
    tmp_path: Path,
) -> None:
    client = JevApiClient(
        make_config(tmp_path),
        sdk_client=AuthenticationErrorSdkClient(),  # type: ignore[arg-type]
    )

    with pytest.raises(JevClientError) as captured:
        client.decide(**request())  # type: ignore[arg-type]

    assert str(captured.value) == "Jev API request failed with HTTP status 401"
    assert captured.value.status == 401
    assert captured.value.request_id == "request-1"
    assert "secret server response" not in str(captured.value)


def test_missing_api_key_fails_during_client_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    with pytest.raises(
        JevClientError, match="failed to initialize the TypeSafe client"
    ):
        JevApiClient(make_config(tmp_path))


@pytest.mark.parametrize(
    ("override", "value", "message"),
    [
        ("model", "", "Jev model must not be empty"),
        ("max_retries", -1, "Jev max_retries must be a non-negative integer"),
        (
            "request_timeout_seconds",
            0.0,
            "Jev request timeout must be a finite positive number",
        ),
    ],
)
def test_rejects_invalid_client_config(
    tmp_path: Path,
    override: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "model": "jev-1.13.0",
        "cache_dir": tmp_path / "cache",
        "max_retries": 3,
        "request_timeout_seconds": 5.0,
    }
    values[override] = value

    with pytest.raises(ValueError, match=message):
        JevClientConfig(**values)  # type: ignore[arg-type]
