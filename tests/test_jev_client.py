import hashlib
import json
from dataclasses import replace
from pathlib import Path

import httpx2
import pytest
from typesafe_sdk import (
    RetryPolicy,
    SystemOneResponse,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeClient,
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
        self.request_options: list[dict[str, object]] = []

    def system_one(
        self, state: object, questions: object, **kwargs: object
    ) -> SystemOneResponse:
        self.calls += 1
        self.request_options.append(kwargs)
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


def cache_request(
    config: JevClientConfig, values: dict[str, object]
) -> dict[str, object]:
    payload = {"base_url": config.base_url, "model": config.model, **values}
    if config.long_context:
        payload["long_context"] = True
    return payload


def request_hash(payload: dict[str, object], *, sorted_keys: bool = False) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=sorted_keys,
        ).encode()
    ).hexdigest()


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
    expected_key = request_hash(cache_request(make_config(tmp_path), values))
    assert first.request_sha256 == second.request_sha256 == expected_key
    assert (
        first.cache_path
        == second.cache_path
        == str(make_config(tmp_path).cache_dir / f"{expected_key}.json")
    )


@pytest.mark.parametrize("long_context", [False, True])
def test_long_context_header_requires_explicit_opt_in(
    tmp_path: Path, long_context: bool
) -> None:
    sdk_client = FakeSdkClient()
    client = JevApiClient(
        replace(make_config(tmp_path), long_context=long_context),
        sdk_client=sdk_client,  # type: ignore[arg-type]
    )

    client.decide(**request())  # type: ignore[arg-type]

    expected = (
        {"extra_headers": {"TypeSafe-Danger-Zone-Long-Context": "true"}}
        if long_context
        else {}
    )
    assert sdk_client.request_options == [expected]


def test_default_and_explicit_false_share_version_three_cache(
    tmp_path: Path,
) -> None:
    sdk_client = FakeSdkClient()
    config = make_config(tmp_path)
    client = JevApiClient(config, sdk_client=sdk_client)  # type: ignore[arg-type]
    values = request()
    expected_request = cache_request(config, values)
    expected_key = request_hash(expected_request)

    client.decide(**values)  # type: ignore[arg-type]
    explicit_default = JevApiClient(
        replace(config, long_context=False),
        sdk_client=sdk_client,  # type: ignore[arg-type]
    )
    cached = explicit_default.decide(**values)  # type: ignore[arg-type]

    assert config.long_context is False
    assert sdk_client.request_options == [{}]
    assert cached.cache_hit is True
    assert sdk_client.calls == 1
    cache_path = config.cache_dir / f"{expected_key}.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["request"] == expected_request
    assert "long_context" not in payload["request"]
    assert payload["request_sha256"] == expected_key


def test_long_context_and_default_responses_use_separate_cache_entries(
    tmp_path: Path,
) -> None:
    sdk_client = FakeSdkClient()
    config = make_config(tmp_path)
    default_client = JevApiClient(
        config,
        sdk_client=sdk_client,  # type: ignore[arg-type]
    )
    long_client = JevApiClient(
        replace(config, long_context=True),
        sdk_client=sdk_client,  # type: ignore[arg-type]
    )

    default_first = default_client.decide(**request())  # type: ignore[arg-type]
    long_first = long_client.decide(**request())  # type: ignore[arg-type]
    default_cached = default_client.decide(**request())  # type: ignore[arg-type]
    long_cached = long_client.decide(**request())  # type: ignore[arg-type]

    assert default_first.cache_hit is False
    assert long_first.cache_hit is False
    assert default_cached.cache_hit is True
    assert long_cached.cache_hit is True
    assert sdk_client.calls == 2
    assert len(list(config.cache_dir.glob("*.json"))) == 2
    assert sdk_client.request_options == [
        {},
        {"extra_headers": {"TypeSafe-Danger-Zone-Long-Context": "true"}},
    ]
    default_path = config.cache_dir / f"{default_first.request_sha256}.json"
    long_path = config.cache_dir / f"{long_first.request_sha256}.json"
    assert "long_context" not in json.loads(default_path.read_text())["request"]
    assert json.loads(long_path.read_text())["request"]["long_context"] is True


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_rejects_non_boolean_long_context_config(tmp_path: Path, value: object) -> None:
    with pytest.raises(ValueError, match="long_context.*boolean"):
        replace(make_config(tmp_path), long_context=value)


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


def test_cache_key_preserves_mapping_insertion_order(tmp_path: Path) -> None:
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

    assert result.cache_hit is False
    assert sdk_client.calls == 2
    assert len(list(make_config(tmp_path).cache_dir.glob("*.json"))) == 2


def test_candidate_presentation_order_uses_separate_cache_entries(
    tmp_path: Path,
) -> None:
    sdk_client = FakeSdkClient()
    config = make_config(tmp_path)
    client = JevApiClient(config, sdk_client=sdk_client)  # type: ignore[arg-type]
    values = request()
    first = client.decide(**values)  # type: ignore[arg-type]
    reversed_values = request()
    criteria = reversed_values["questions"]["decision"]["criteria"]
    reversed_values["questions"]["decision"]["criteria"] = dict(
        reversed(list(criteria.items()))
    )

    reversed_result = client.decide(**reversed_values)  # type: ignore[arg-type]
    first_cached = client.decide(**values)  # type: ignore[arg-type]
    reversed_cached = client.decide(**reversed_values)  # type: ignore[arg-type]

    assert not first.cache_hit and not reversed_result.cache_hit
    assert first_cached.cache_hit and reversed_cached.cache_hit
    assert sdk_client.calls == 2
    assert first.request_sha256 != reversed_result.request_sha256
    for result, expected_ids in (
        (first, ["candidate-a", "candidate-b"]),
        (reversed_result, ["candidate-b", "candidate-a"]),
    ):
        stored = json.loads(Path(result.cache_path).read_text(encoding="utf-8"))
        assert (
            list(stored["request"]["questions"]["decision"]["criteria"]) == expected_ids
        )


@pytest.mark.parametrize(
    "tampering",
    ["state", "candidate_order", "long_context", "missing_request", "declared_hash"],
)
def test_rejects_tampered_stored_request(tmp_path: Path, tampering: str) -> None:
    sdk_client = FakeSdkClient()
    config = make_config(tmp_path)
    client = JevApiClient(config, sdk_client=sdk_client)  # type: ignore[arg-type]
    values = request()
    result = client.decide(**values)  # type: ignore[arg-type]
    path = Path(result.cache_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tampering == "state":
        payload["request"]["state"]["operation"] = "secret tampered operation"
    elif tampering == "candidate_order":
        criteria = payload["request"]["questions"]["decision"]["criteria"]
        payload["request"]["questions"]["decision"]["criteria"] = dict(
            reversed(list(criteria.items()))
        )
    elif tampering == "long_context":
        payload["request"]["long_context"] = True
    elif tampering == "missing_request":
        del payload["request"]
    else:
        payload["request_sha256"] = "0" * 64
    modified = json.dumps(payload)
    path.write_text(modified, encoding="utf-8")

    with pytest.raises(JevCacheError, match="failed to read Jev cache file") as error:
        client.decide(**values)  # type: ignore[arg-type]

    assert "secret tampered operation" not in str(error.value)
    assert sdk_client.calls == 1
    assert path.read_text(encoding="utf-8") == modified


def test_old_schema_at_current_cache_path_is_rejected_without_overwriting(
    tmp_path: Path,
) -> None:
    sdk_client = FakeSdkClient()
    config = make_config(tmp_path)
    client = JevApiClient(config, sdk_client=sdk_client)  # type: ignore[arg-type]
    result = client.decide(**request())  # type: ignore[arg-type]
    path = Path(result.cache_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    payload.pop("request")
    legacy_text = json.dumps(payload)
    path.write_text(legacy_text, encoding="utf-8")

    with pytest.raises(JevCacheError, match="failed to read Jev cache file"):
        client.decide(**request())  # type: ignore[arg-type]

    assert sdk_client.calls == 1
    assert path.read_text(encoding="utf-8") == legacy_text


def test_legacy_sorted_cache_remains_unchanged_and_is_not_rekeyed(
    tmp_path: Path,
) -> None:
    sdk_client = FakeSdkClient()
    config = make_config(tmp_path)
    values = request()
    expected_request = cache_request(config, values)
    old_key = request_hash(expected_request, sorted_keys=True)
    new_key = request_hash(expected_request)
    assert old_key != new_key
    legacy_path = config.cache_dir / f"{old_key}.json"
    legacy_text = json.dumps(
        {
            "schema_version": 1,
            "request_sha256": old_key,
            "response": {"sentinel": "legacy response must not be reused"},
        }
    )
    config.cache_dir.mkdir()
    legacy_path.write_text(legacy_text, encoding="utf-8")
    client = JevApiClient(config, sdk_client=sdk_client)  # type: ignore[arg-type]

    result = client.decide(**values)  # type: ignore[arg-type]

    assert not result.cache_hit
    assert sdk_client.calls == 1
    assert legacy_path.read_text(encoding="utf-8") == legacy_text
    assert (config.cache_dir / f"{new_key}.json").is_file()
    assert len(list(config.cache_dir.glob("*.json"))) == 2


@pytest.mark.parametrize("long_context", [False, True])
def test_real_sdk_sends_long_context_header_only_when_opted_in(
    tmp_path: Path,
    long_context: bool,
) -> None:
    captured: list[tuple[httpx2.Headers, dict[str, object]]] = []
    response_payload = FakeSdkClient().system_one(None, None).model_dump(mode="json")

    def respond(outgoing: httpx2.Request) -> httpx2.Response:
        captured.append((outgoing.headers, json.loads(outgoing.content)))
        return httpx2.Response(
            200,
            json=response_payload,
            headers={"x-typesafe-request-id": "offline-request-1"},
        )

    config = replace(make_config(tmp_path), long_context=long_context)
    with TypeSafeClient(
        api_key="offline-test-key",
        model=config.model,
        transport=httpx2.MockTransport(respond),
        compression=False,
        retry=RetryPolicy(max_retries=0),
    ) as sdk_client:
        client = JevApiClient(config, sdk_client=sdk_client)
        result = client.decide(**request())  # type: ignore[arg-type]

    assert not result.cache_hit
    assert len(captured) == 1
    headers, body = captured[0]
    assert headers.get("TypeSafe-Danger-Zone-Long-Context") == (
        "true" if long_context else None
    )
    assert set(body) == {"state", "questions", "model"}
    assert body["state"] == request()["state"]
    assert body["questions"] == request()["questions"]
    assert body["model"] == config.model


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
