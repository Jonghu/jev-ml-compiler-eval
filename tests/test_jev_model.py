from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

import jev_compiler_eval.models.jev as jev_module
from jev_compiler_eval.jev import JevApiResult, JevClientConfig
from jev_compiler_eval.models import JevModel, get_model_class
from jev_compiler_eval.types import Candidate, Workload


class FakeJevClient:
    instances: list["FakeJevClient"] = []

    def __init__(self, config: JevClientConfig) -> None:
        self.config = config
        self.calls: list[dict[str, object]] = []
        self.closed = False
        self.__class__.instances.append(self)

    def decide(
        self,
        *,
        state: object,
        questions: Mapping[str, Mapping[str, object]],
    ) -> JevApiResult:
        self.calls.append({"state": state, "questions": questions})
        return JevApiResult(
            model="jev-1.14.0",
            answers={
                "decision": {
                    "type": "choice",
                    "choice": "candidate-b",
                    "confidence": 0.6,
                    "probabilities": {
                        "candidate-a": 0.1,
                        "candidate-b": 0.6,
                        "candidate-c": 0.3,
                    },
                }
            },
            input_tokens=100,
            output_tokens=20,
            request_id="request-1",
            cache_hit=False,
            request_latency_ms=12.5,
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_jev_client(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeJevClient.instances.clear()
    monkeypatch.setattr(jev_module, "JevApiClient", FakeJevClient)


def model_config(tmp_path: Path) -> dict[str, object]:
    return {
        "model": "jev-1.14.0",
        "cache_dir": str(tmp_path / "cache"),
        "max_retries": 3,
        "request_timeout_seconds": 30.0,
    }


def candidates() -> tuple[Candidate, ...]:
    return (
        Candidate("candidate-a", {"tile": 8}),
        Candidate("candidate-b", {"tile": 16}),
        Candidate("candidate-c", {"tile": 32}),
    )


def test_jev_model_is_registered() -> None:
    assert get_model_class("jev") is JevModel


def test_serializes_generic_workload_and_candidates(tmp_path: Path) -> None:
    model = JevModel(model_config(tmp_path))
    workload = Workload(
        "new-benchmark-workload",
        {"operation": "custom_op", "shape": [1, 2, 3]},
    )

    result = model.rank(workload, candidates())

    call = FakeJevClient.instances[0].calls[0]
    assert call["state"] == {
        "workload_id": "new-benchmark-workload",
        "context": {"operation": "custom_op", "shape": [1, 2, 3]},
    }
    assert call["questions"] == {
        "decision": {
            "type": "choice",
            "instructions": (
                "Which candidate is most likely to have the lowest runtime for this "
                "workload?"
            ),
            "criteria": {
                "candidate-a": {"tile": 8},
                "candidate-b": {"tile": 16},
                "candidate-c": {"tile": 32},
            },
        }
    }
    assert result.ranked_candidate_ids == (
        "candidate-b",
        "candidate-c",
        "candidate-a",
    )


def test_returns_api_metadata_and_reuses_client(tmp_path: Path) -> None:
    model = JevModel(model_config(tmp_path))
    workload = Workload("workload-1", {})

    first = model.rank(workload, candidates())
    model.rank(Workload("workload-2", {}), candidates())
    model.close()

    client = FakeJevClient.instances[0]
    assert len(FakeJevClient.instances) == 1
    assert len(client.calls) == 2
    assert client.closed is True
    assert first.metadata == {
        "requested_model": "jev-1.14.0",
        "resolved_model": "jev-1.14.0",
        "probabilities": {
            "candidate-a": 0.1,
            "candidate-b": 0.6,
            "candidate-c": 0.3,
        },
        "confidence": 0.6,
        "input_tokens": 100,
        "output_tokens": 20,
        "request_id": "request-1",
        "cache_hit": False,
        "api_latency_ms": 12.5,
    }


@pytest.mark.parametrize("long_context", [False, True])
def test_propagates_long_context_mode_and_records_opt_in_only(
    tmp_path: Path, long_context: bool
) -> None:
    config = {**model_config(tmp_path), "long_context": long_context}
    model = JevModel(config)

    result = model.rank(Workload("workload-1", {}), candidates())

    assert FakeJevClient.instances[0].config.long_context is long_context
    if long_context:
        assert result.metadata["long_context"] is True
    else:
        assert "long_context" not in result.metadata
    assert result.ranked_candidate_ids == (
        "candidate-b",
        "candidate-c",
        "candidate-a",
    )
    assert result.metadata["probabilities"] == {
        "candidate-a": 0.1,
        "candidate-b": 0.6,
        "candidate-c": 0.3,
    }


def test_default_model_keeps_long_context_disabled(tmp_path: Path) -> None:
    model = JevModel(model_config(tmp_path))

    result = model.rank(Workload("workload-1", {}), candidates())

    assert FakeJevClient.instances[0].config.long_context is False
    assert "long_context" not in result.metadata


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_rejects_non_boolean_long_context_before_constructing_client(
    tmp_path: Path, value: object
) -> None:
    config = {**model_config(tmp_path), "long_context": value}

    with pytest.raises(ValueError, match="long_context.*boolean"):
        JevModel(config)

    assert FakeJevClient.instances == []


def test_runtime_labels_cannot_be_passed_to_jev_rank(tmp_path: Path) -> None:
    model = JevModel(model_config(tmp_path))

    with pytest.raises(TypeError):
        model.rank(  # type: ignore[call-arg]
            Workload("workload-1", {}),
            candidates(),
            runtimes={"candidate-a": 1.0},
        )


def test_rejects_probabilities_that_do_not_match_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = JevModel(model_config(tmp_path))
    client = FakeJevClient.instances[0]

    def incomplete_response(**kwargs: object) -> JevApiResult:
        return JevApiResult(
            model="jev-1.14.0",
            answers={
                "decision": {
                    "type": "choice",
                    "choice": "candidate-b",
                    "confidence": 0.6,
                    "probabilities": {"candidate-b": 1.0},
                }
            },
            input_tokens=100,
            output_tokens=20,
            request_id=None,
            cache_hit=False,
            request_latency_ms=12.5,
        )

    monkeypatch.setattr(client, "decide", incomplete_response)

    with pytest.raises(
        ValueError,
        match="probabilities must match evaluation candidates exactly",
    ):
        model.rank(Workload("workload-1", {}), candidates())


@pytest.mark.parametrize(
    ("override", "value", "message"),
    [
        ("model", "", "Jev model must be a non-empty string"),
        ("cache_dir", "", "Jev cache_dir must be a non-empty string"),
        ("max_retries", -1, "Jev max_retries must be a non-negative integer"),
        (
            "request_timeout_seconds",
            0,
            "Jev request timeout must be a finite positive number",
        ),
    ],
)
def test_rejects_invalid_model_config(
    tmp_path: Path,
    override: str,
    value: object,
    message: str,
) -> None:
    config = model_config(tmp_path)
    config[override] = value

    with pytest.raises(ValueError, match=message):
        JevModel(config)


def test_defaults_keep_the_original_request_shape(tmp_path: Path) -> None:
    model = JevModel(model_config(tmp_path))

    result = model.rank(Workload("workload-1", {"op": "add"}), candidates())

    call = FakeJevClient.instances[0].calls[0]
    assert call["state"] == {"workload_id": "workload-1", "context": {"op": "add"}}
    criteria = call["questions"]["decision"]["criteria"]  # type: ignore[index]
    assert list(criteria) == ["candidate-a", "candidate-b", "candidate-c"]
    assert "presented_candidate_ids" not in result.metadata


def test_can_omit_workload_context(tmp_path: Path) -> None:
    config = model_config(tmp_path)
    config["include_context"] = False
    model = JevModel(config)

    model.rank(Workload("workload-1", {"op": "add"}), candidates())

    assert FakeJevClient.instances[0].calls[0]["state"] == {"workload_id": "workload-1"}


def test_shuffles_candidates_reproducibly_per_workload(tmp_path: Path) -> None:
    many = tuple(Candidate(f"candidate-{i}", {"tile": i}) for i in range(12))

    def presented(seed: int, workload_id: str) -> list[str]:
        config = model_config(tmp_path)
        config["candidate_order"] = "shuffled"
        config["order_seed"] = seed
        model = JevModel(config)
        client = FakeJevClient.instances[-1]

        def respond(**kwargs: object) -> JevApiResult:
            questions = cast(Mapping[str, Mapping[str, object]], kwargs["questions"])
            ids = list(cast(Mapping[str, object], questions["decision"]["criteria"]))
            client.calls.append({"criteria_order": ids})
            return JevApiResult(
                model="jev-1.14.0",
                answers={
                    "decision": {
                        "type": "choice",
                        "choice": ids[0],
                        "confidence": 0.5,
                        "probabilities": {
                            cid: (0.5 if cid == ids[0] else 0.0) for cid in ids
                        },
                    }
                },
                input_tokens=None,
                output_tokens=None,
                request_id=None,
                cache_hit=False,
                request_latency_ms=1.0,
            )

        client.decide = respond  # type: ignore[method-assign]
        result = model.rank(Workload(workload_id, {}), many)
        order = cast(list[str], client.calls[-1]["criteria_order"])
        assert result.metadata["presented_candidate_ids"] == order
        assert sorted(result.ranked_candidate_ids) == sorted(order)
        return order

    original = [c.candidate_id for c in many]
    first = presented(42, "workload-1")
    assert first != original
    assert sorted(first) == sorted(original)
    assert presented(42, "workload-1") == first
    assert presented(42, "workload-2") != first
    assert presented(7, "workload-1") != first


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"include_context": "no"}, "include_context must be a boolean"),
        ({"candidate_order": "reversed"}, "candidate_order must be"),
        ({"candidate_order": "shuffled"}, "requires order_seed"),
        ({"order_seed": 1}, 'order_seed requires candidate_order "shuffled"'),
        (
            {"candidate_order": "shuffled", "order_seed": True},
            "order_seed must be an integer",
        ),
        ({"temperature": 0}, "unknown Jev model config keys"),
    ],
)
def test_rejects_invalid_order_and_context_settings(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    config = model_config(tmp_path)
    config.update(overrides)

    with pytest.raises(ValueError, match=message):
        JevModel(config)
