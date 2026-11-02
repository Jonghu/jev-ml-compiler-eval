import json
from collections.abc import Mapping
from pathlib import Path

import pytest

import jev_compiler_eval.models.jev as jev_module
from jev_compiler_eval.config import load_experiment_config
from jev_compiler_eval.evaluation import run_experiment
from jev_compiler_eval.jev import JevApiResult, JevClientConfig

PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "experiments" / "mock" / "random-vs-jev.yaml"


class FakeJevClient:
    instances: list["FakeJevClient"] = []

    def __init__(self, config: JevClientConfig) -> None:
        self.config = config
        self.calls = 0
        self.closed = False
        self.__class__.instances.append(self)

    def decide(
        self,
        *,
        state: object,
        questions: Mapping[str, Mapping[str, object]],
    ) -> JevApiResult:
        self.calls += 1
        criteria = questions["decision"]["criteria"]
        assert isinstance(criteria, Mapping)
        candidate_ids = list(criteria)
        probabilities = {
            candidate_id: probability
            for candidate_id, probability in zip(
                candidate_ids,
                (0.6, 0.3, 0.1),
                strict=True,
            )
        }
        return JevApiResult(
            model="jev-1.14.0",
            answers={
                "decision": {
                    "type": "choice",
                    "choice": candidate_ids[0],
                    "confidence": 0.5,
                    "probabilities": probabilities,
                }
            },
            input_tokens=100,
            output_tokens=20,
            request_id="test-request",
            cache_hit=False,
            request_latency_ms=1.0,
        )

    def close(self) -> None:
        self.closed = True


def test_example_configuration_selects_random_and_jev() -> None:
    config = load_experiment_config(EXAMPLE_CONFIG_PATH)

    assert config.dataset.dataset_type == "mock"
    assert tuple(model.model_id for model in config.models) == (
        "random",
        "jev-1.14.0",
    )
    assert config.models[1].model_type == "jev"
    assert config.models[1].config["model"] == "jev-1.14.0"


def test_example_experiment_runs_without_live_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeJevClient.instances.clear()
    monkeypatch.setattr(jev_module, "JevApiClient", FakeJevClient)
    output_dir = tmp_path / "results"
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8").replace(
            "results/mock/random-vs-jev",
            str(output_dir),
        ),
        encoding="utf-8",
    )

    result = run_experiment(config_path)

    assert tuple(summary.model_id for summary in result.summaries) == (
        "random",
        "jev-1.14.0",
    )
    client = FakeJevClient.instances[0]
    assert client.calls == 2
    assert client.closed is True

    predictions = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    random_candidates = [
        prediction["candidate_ids"]
        for prediction in predictions
        if prediction["model_id"] == "random"
    ]
    jev_candidates = [
        prediction["candidate_ids"]
        for prediction in predictions
        if prediction["model_id"] == "jev-1.14.0"
    ]
    assert random_candidates == jev_candidates
