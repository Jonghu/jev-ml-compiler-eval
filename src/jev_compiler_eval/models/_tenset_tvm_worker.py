"""Standalone Python 3.8+ bridge to the original TenSet TVM cost model.

Run as a file, not an imported module: the original environment does not need
the Python 3.11 evaluation package installed. Checkpoints and all_tasks.pkl are
trusted local pickle artifacts; never use untrusted downloads for either.
"""

import argparse
import contextlib
import hashlib
import inspect
import json
import math
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

IMPLEMENTATION = "tlc-pack/tenset:auto_scheduler.cost_model.XGBModel"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _register_tasks(task_registry):
    import tvm
    from tvm import auto_scheduler

    with open(task_registry, "rb") as artifact:
        tasks = pickle.load(artifact)
    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors
        )
    return tasks, tvm


def _dependencies(task_registry):
    try:
        import numpy
        import tvm
        import xgboost
        from tvm.auto_scheduler.cost_model import XGBModel
        from tvm.auto_scheduler.cost_model.xgb_model import XGBModelInternal
        from tvm.auto_scheduler.measure import recover_measure_input
        from tvm.auto_scheduler.measure_record import load_record_from_string
    except ImportError as error:
        raise RuntimeError(
            "Install the original tlc-pack/tenset TVM fork and its XGBoost 1.x "
            "dependencies in the selected python_executable environment. "
            "Current apache-tvm wheels do not supply this historical model."
        ) from error
    if not hasattr(XGBModelInternal, "fit_base"):
        raise RuntimeError("the TVM installation is missing the official TenSet model")
    _register_tasks(task_registry)
    versions = {
        "python": sys.version.split()[0],
        "tvm": tvm.__version__,
        "xgboost": xgboost.__version__,
        "numpy": numpy.__version__,
        "cost_model_source_sha256": _sha256(inspect.getfile(XGBModel)),
    }
    return XGBModel, load_record_from_string, recover_measure_input, versions


def _dimension(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return str(value)


def _metadata(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("metadata destination must not exist")
    tasks, tvm = _register_tasks(args.task_registry)
    metadata = {}
    for task in tasks:
        tensors, loops, operation_names = [], [], []
        for tensor in task.compute_dag.tensors:
            operation = tensor.op
            name = str(operation.name)
            tensors.append(
                {
                    "name": name,
                    "shape": [_dimension(dim) for dim in tensor.shape],
                    "dtype": str(tensor.dtype),
                }
            )
        for operation in task.compute_dag.ops:
            name = str(operation.name)
            axes = list(getattr(operation, "axis", []))
            reductions = list(getattr(operation, "reduce_axis", []))
            if axes or reductions:
                operation_names.append(name)
                loops.append(
                    {
                        "operation": name,
                        "spatial": [
                            {
                                "variable": str(axis.var),
                                "extent": _dimension(axis.dom.extent),
                            }
                            for axis in axes
                        ],
                        "reduction": [
                            {
                                "variable": str(axis.var),
                                "extent": _dimension(axis.dom.extent),
                            }
                            for axis in reductions
                        ],
                    }
                )
        metadata[task.workload_key] = {
            "operation": "+".join(operation_names) or "placeholder",
            "tensors": tensors,
            "compute_dag": str(
                tvm.get_global_func("auto_scheduler.ComputeDAGPrintDAG")(
                    task.compute_dag, False
                )
            ),
            "loop_structure": loops,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "workloads": len(metadata)}


def _measure_record(candidate, include_label):
    runtime_seconds = candidate["runtime_ms"] / 1000.0 if include_label else 1.0
    return json.dumps(
        {
            "i": candidate.get("record_input", candidate.get("i")),
            "r": [[runtime_seconds], 0, 0.0, 0.0],
            "v": candidate.get("record_version", candidate.get("v")),
        }
    )


def _validate_training_input(entry, candidate):
    raw = candidate["record_input"]
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or not isinstance(raw[0], list)
        or raw[0][:2] != [entry["context"]["workload_key"], entry["context"]["target"]]
        or not isinstance(raw[1], list)
        or len(raw[1]) != 2
    ):
        raise ValueError("hidden TVM training input differs from the visible workload")
    state = raw[1]
    configuration = candidate["configuration"]
    expected_steps = [
        [step["code"]] + step["arguments"] for step in configuration["transformations"]
    ]
    if state[0] != configuration["initial_state"] or state[1] != expected_steps:
        raise ValueError("hidden TVM training input differs from the visible candidate")


def _train(args):
    output = Path(args.checkpoint)
    provenance = Path(str(output) + ".json")
    if output.exists() or provenance.exists():
        raise FileExistsError("checkpoint and provenance destinations must not exist")
    prepared = json.loads(Path(args.prepared).read_text(encoding="utf-8"))
    training = [entry for entry in prepared["workloads"] if entry["split"] == "train"]
    if not training:
        raise ValueError("prepared dataset has no training workloads")
    ids = [entry["workload_id"] for entry in training]
    if len(ids) != len(set(ids)):
        raise ValueError("training workload IDs must be unique")
    training_keys = {entry["context"]["workload_key"] for entry in training}
    if any(
        entry["context"]["workload_key"] in training_keys
        for entry in prepared["workloads"]
        if entry["split"] != "train"
    ):
        raise ValueError("train and held-out workload keys overlap")
    targets = {entry["context"]["target"] for entry in training}
    if len(targets) != 1:
        raise ValueError("train the hardware-specific XGBoost model on one target")
    model_class, read_record, _, versions = _dependencies(args.task_registry)
    model = model_class(seed=args.seed, num_warmup_sample=0)
    # The original fork uses `seed or 43`; preserve an explicitly requested zero.
    if hasattr(model.model, "xgb_params"):
        model.model.xgb_params["seed"] = args.seed
    inputs, results = [], []
    for entry in training:
        for candidate in entry["candidates"]:
            _validate_training_input(entry, candidate)
            runtime = candidate["runtime_ms"]
            if isinstance(runtime, bool) or not math.isfinite(runtime) or runtime <= 0:
                raise ValueError(
                    "training runtimes must be finite positive milliseconds"
                )
            inp, result = read_record(_measure_record(candidate, include_label=True))
            inputs.append(inp)
            results.append(result)
    model.update(inputs, results)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    # Loading also disables the official model's random warmup predictions.
    model.load(str(output))
    record = {
        "schema_version": 1,
        "implementation": IMPLEMENTATION,
        "checkpoint_sha256": _sha256(output),
        "prepared_sha256": _sha256(args.prepared),
        "task_registry_sha256": _sha256(args.task_registry),
        "training_workload_ids": sorted(ids),
        "training_candidates": len(inputs),
        "target": next(iter(targets)),
        "seed": args.seed,
        "versions": versions,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    provenance.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _emit(response):
    print(json.dumps(response, allow_nan=False), flush=True)


def _serve(args):
    with contextlib.redirect_stdout(sys.stderr):
        model_class, read_record, recover, versions = _dependencies(args.task_registry)
        model = model_class()
        model.load(args.checkpoint)
    _emit({"status": "ready", "versions": versions})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("op") != "predict" or not request.get("records"):
                raise ValueError("worker requires a nonempty predict request")
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                inputs = [
                    read_record(_measure_record(record, include_label=False))[0]
                    for record in request["records"]
                ]
                first_task = inputs[0].task
                if any(
                    inp.task.workload_key != first_task.workload_key
                    or str(inp.task.target) != str(first_task.target)
                    for inp in inputs
                ):
                    raise ValueError("a ranking must contain one workload and target")
                task = recover(inputs[0]).task
                scores = [
                    float(score)
                    for score in model.predict(task, [i.state for i in inputs])
                ]
            if not all(math.isfinite(score) for score in scores):
                raise ValueError(
                    "TVM failed to lower or predict a candidate; nonfinite score"
                )
            _emit(
                {
                    "scores": scores,
                    "feature_prediction_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
        except Exception as error:
            _emit({"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--prepared", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--task-registry", required=True)
    train.add_argument("--seed", type=int, default=42)
    serve = commands.add_parser("serve")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--task-registry", required=True)
    metadata = commands.add_parser("metadata")
    metadata.add_argument("--task-registry", required=True)
    metadata.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "train":
            with contextlib.redirect_stdout(sys.stderr):
                result = _train(args)
            _emit(result)
        elif args.command == "metadata":
            with contextlib.redirect_stdout(sys.stderr):
                result = _metadata(args)
            _emit(result)
        else:
            _serve(args)
    except Exception as error:
        _emit({"error": str(error)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
