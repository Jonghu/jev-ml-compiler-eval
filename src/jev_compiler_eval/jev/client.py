"""Cached synchronous client for the official TypeSafe Jev SDK."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from time import perf_counter_ns
from typing import cast
from uuid import uuid4

from typesafe_sdk import (
    RetryPolicy,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
    TypeSafeClient,
    TypeSafeError,
)

CACHE_SCHEMA_VERSION = 3


class JevClientError(RuntimeError):
    """A secret-safe project error raised for TypeSafe client failures."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.request_id = request_id


class JevCacheError(JevClientError):
    """A cached Jev response could not be read or validated."""


@dataclass(frozen=True)
class JevClientConfig:
    """Network and cache settings for the Jev API client."""

    model: str
    cache_dir: Path
    max_retries: int = 3
    request_timeout_seconds: float = 30.0
    base_url: str = "https://api.typesafe.ai"
    long_context: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.long_context, bool):
            raise ValueError("Jev long_context must be a boolean")
        if not self.model.strip():
            raise ValueError("Jev model must not be empty")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise ValueError("Jev max_retries must be a non-negative integer")
        if self.max_retries < 0:
            raise ValueError("Jev max_retries must be a non-negative integer")
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or not isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("Jev request timeout must be a finite positive number")
        if not self.base_url.strip():
            raise ValueError("Jev base_url must not be empty")


@dataclass(frozen=True)
class JevApiResult:
    """Serializable response data returned by Jev or its local cache."""

    model: str
    answers: Mapping[str, Mapping[str, object]]
    input_tokens: int | None
    output_tokens: int | None
    request_id: str | None
    cache_hit: bool
    request_latency_ms: float
    request_sha256: str | None = None
    cache_path: str | None = None


class JevApiClient:
    """Call Jev through the official SDK and cache successful responses.

    Each cache file stores the request body next to the response so that the
    exact model input can be inspected later. The request body never contains
    API credentials.

    The cache key hashes the request in the exact key order that is sent to
    Jev. Order is part of the model input: Choice candidates presented in a
    different order are a different request and must not share a response.
    """

    def __init__(
        self,
        config: JevClientConfig,
        *,
        sdk_client: TypeSafeClient | None = None,
    ) -> None:
        self.config = config
        self._owns_sdk_client = sdk_client is None
        if sdk_client is not None:
            self._sdk_client = sdk_client
            return

        try:
            self._sdk_client = TypeSafeClient(
                model=config.model,
                retry=RetryPolicy(max_retries=config.max_retries),
                timeout=config.request_timeout_seconds,
                base_url=config.base_url,
            )
        except TypeSafeError as error:
            raise JevClientError("failed to initialize the TypeSafe client") from error

    def __enter__(self) -> "JevApiClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the SDK client when this wrapper created it."""

        if self._owns_sdk_client:
            self._sdk_client.close()

    def decide(
        self,
        *,
        state: object,
        questions: Mapping[str, Mapping[str, object]],
    ) -> JevApiResult:
        """Return one typed Jev decision response, using the cache when present."""

        started_at = perf_counter_ns()
        cache_request = self._cache_request(state=state, questions=questions)
        cache_key = self._cache_key(cache_request)
        cache_path = self.config.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            return self._read_cache(cache_path, cache_key, started_at)

        extra = (
            {"extra_headers": {"TypeSafe-Danger-Zone-Long-Context": "true"}}
            if self.config.long_context
            else {}
        )
        try:
            response = self._sdk_client.system_one(
                state=state,  # type: ignore[arg-type]
                questions=questions,  # type: ignore[arg-type]
                **extra,
            )
        except TypeSafeAPITimeoutError as error:
            raise JevClientError("Jev API request timed out") from error
        except TypeSafeAPIConnectionError as error:
            raise JevClientError("Jev API connection failed") from error
        except TypeSafeAPIError as error:
            raise JevClientError(
                f"Jev API request failed with HTTP status {error.status}",
                status=error.status,
                request_id=error.request_id,
            ) from error
        except TypeSafeError as error:
            raise JevClientError("Jev API request failed") from error

        if not isinstance(response, SystemOneResponse):
            raise JevClientError("Jev API returned an unexpected response type")
        result = replace(
            self._result_from_response(response, started_at),
            request_sha256=cache_key,
            cache_path=str(cache_path),
        )
        self._write_cache(cache_path, cache_key, cache_request, result)
        return result

    def _cache_request(
        self,
        *,
        state: object,
        questions: Mapping[str, Mapping[str, object]],
    ) -> dict[str, object]:
        request = {
            "base_url": self.config.base_url,
            "model": self.config.model,
            "state": state,
            "questions": questions,
        }
        if self.config.long_context:
            request["long_context"] = True
        return request

    @staticmethod
    def _cache_key(request: Mapping[str, object]) -> str:
        try:
            encoded = json.dumps(
                request,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        except (TypeError, ValueError) as error:
            raise JevClientError(
                "Jev state and questions must be JSON serializable"
            ) from error
        return hashlib.sha256(encoded).hexdigest()

    def _result_from_response(
        self,
        response: SystemOneResponse,
        started_at: int,
    ) -> JevApiResult:
        try:
            request_id = response.request_id
        except TypeSafeError:
            request_id = None
        answers = {
            name: cast(dict[str, object], answer.model_dump(mode="json"))
            for name, answer in response.answers.items()
        }
        return JevApiResult(
            model=response.model,
            answers=answers,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            request_id=request_id,
            cache_hit=False,
            request_latency_ms=(perf_counter_ns() - started_at) / 1_000_000,
        )

    def _read_cache(
        self,
        path: Path,
        expected_key: str,
        started_at: int,
    ) -> JevApiResult:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("cache root must be an object")
            if value.get("schema_version") != CACHE_SCHEMA_VERSION:
                raise ValueError("cache schema version does not match")
            if value.get("request_sha256") != expected_key:
                raise ValueError("cache request hash does not match")
            cached_request = value["request"]
            if not isinstance(cached_request, dict):
                raise ValueError("cached request must be an object")
            if self._cache_key(cached_request) != expected_key:
                raise ValueError("cached request does not match its hash")
            response = value["response"]
            if not isinstance(response, dict):
                raise ValueError("cached response must be an object")
            model = response["model"]
            answers = response["answers"]
            usage = response["usage"]
            request_id = response.get("request_id")
            if not isinstance(model, str) or not model:
                raise ValueError("cached model must be a non-empty string")
            if not isinstance(answers, dict) or not all(
                isinstance(name, str) and isinstance(answer, dict)
                for name, answer in answers.items()
            ):
                raise ValueError("cached answers must be an object of objects")
            if not isinstance(usage, dict):
                raise ValueError("cached usage must be an object")
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if input_tokens is not None and not isinstance(input_tokens, int):
                raise ValueError("cached input_tokens must be an integer or null")
            if output_tokens is not None and not isinstance(output_tokens, int):
                raise ValueError("cached output_tokens must be an integer or null")
            if request_id is not None and not isinstance(request_id, str):
                raise ValueError("cached request_id must be a string or null")
        except (OSError, KeyError, TypeError, ValueError, JevClientError) as error:
            raise JevCacheError(f'failed to read Jev cache file "{path}"') from error

        return JevApiResult(
            model=model,
            answers=cast(dict[str, dict[str, object]], answers),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_id=request_id,
            cache_hit=True,
            request_latency_ms=(perf_counter_ns() - started_at) / 1_000_000,
            request_sha256=expected_key,
            cache_path=str(path),
        )

    def _write_cache(
        self,
        path: Path,
        request_key: str,
        request: Mapping[str, object],
        result: JevApiResult,
    ) -> None:
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "request_sha256": request_key,
            "request": request,
            "response": {
                "model": result.model,
                "answers": result.answers,
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
                "request_id": result.request_id,
            },
        }
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(path)
        except OSError as error:
            temporary_path.unlink(missing_ok=True)
            raise JevCacheError(f'failed to write Jev cache file "{path}"') from error
