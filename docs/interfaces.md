# Extension interfaces

Benchmark workstreams extend the shared framework by implementing a dataset
adapter and, when required, a specialized baseline model.

## Dataset interface

Every dataset inherits from `BaseDataset` and returns `EvaluationCase` values.

```python
class ExampleDataset(BaseDataset):
    @property
    def dataset_type(self) -> str:
        return "example"

    def validate_config(self) -> None:
        ...

    def load(self) -> Sequence[EvaluationCase]:
        ...
```

An evaluation case separates model-visible data from measured runtime labels:

```python
EvaluationCase(
    workload=Workload(
        workload_id="workload-1",
        context={"operation": "matmul", "shape": [128, 256, 64]},
    ),
    candidates=(
        Candidate("candidate-1", {"tile_m": 8, "tile_n": 8}),
        Candidate("candidate-2", {"tile_m": 16, "tile_n": 16}),
    ),
    runtimes=(
        RuntimeRecord("workload-1", "candidate-1", 2.4),
        RuntimeRecord("workload-1", "candidate-2", 1.2),
    ),
)
```

`Workload.context` and `Candidate.configuration` must be JSON-serializable and
must not contain measured runtime labels.

## Model interface

Every model inherits from `BaseDecisionModel`:

```python
class ExampleModel(BaseDecisionModel):
    @property
    def model_type(self) -> str:
        return "example-model"

    def validate_config(self) -> None:
        ...

    def rank(
        self,
        workload: Workload,
        candidates: Sequence[Candidate],
    ) -> DecisionResult:
        ...

    def close(self) -> None:
        ...
```

`close()` is a no-op by default. Models that own network clients, GPU state, or
other resources override it. The runner calls it in a `finally` block.

## Registration

Built-in implementations are explicitly registered in their package
`__init__.py` files:

```python
register_dataset("example", ExampleDataset)
register_model("example-model", ExampleModel)
```

YAML structure is validated when loaded. Registered dataset, model, and metric
names are resolved when the experiment runner starts.

## Generic Jev serialization

`JevModel` works with any dataset that follows the common types. It sends the
workload ID and context as state, and maps candidate IDs to their configuration
objects in one Choice question. It converts returned probabilities into a full
ranking.

Dataset owners do not implement API, retry, cache, or ranking logic. They are
responsible for producing compact, semantically meaningful, JSON-friendly
workload and candidate representations within TypeSafe API limits.
