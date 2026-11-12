"""Regenerate format fixtures with the pinned original TenSet TVM runtime.

These are synthetic tensor programs, not benchmark measurements. This script
is a developer validation tool; normal tests and metadata export need no TVM.
"""

import hashlib
import json
import pickle
from pathlib import Path

import tvm
from tvm import auto_scheduler, te, tir


def programs():
    a = te.placeholder((4, 8), name="a")
    b = te.placeholder((8, 6), name="b")
    k = te.reduce_axis((0, 8), name="k")
    c = te.compute(
        (4, 6), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), name="matmul"
    )
    yield "matmul", [a, b, c]

    for operator in ("add", "multiply"):
        a = te.placeholder((4, 4), name="left")
        b = te.placeholder((4, 4), name="right")
        body = (
            (lambda i, j: a[i, j] + b[i, j])
            if operator == "add"
            else (lambda i, j: a[i, j] * b[i, j])
        )
        c = te.compute((4, 4), body, name="same_name")
        yield operator, [a, b, c]

    a = te.placeholder((1, 3, 5, 5), name="image")
    w = te.placeholder((4, 3, 3, 3), name="weight")
    bias = te.placeholder((4,), name="bias")
    pad = te.compute(
        (1, 3, 7, 7),
        lambda n, c, y, x: tir.if_then_else(
            tir.all(y >= 1, y < 6, x >= 1, x < 6),
            a[n, c, y - 1, x - 1],
            tir.const(0, "float32"),
        ),
        name="padding",
    )
    rc = te.reduce_axis((0, 3), name="rc")
    ry = te.reduce_axis((0, 3), name="ry")
    rx = te.reduce_axis((0, 3), name="rx")
    conv = te.compute(
        (1, 4, 5, 5),
        lambda n, c, y, x: te.sum(
            pad[n, rc, y + ry, x + rx] * w[c, rc, ry, rx], axis=[rc, ry, rx]
        ),
        name="conv2d",
    )
    out = te.compute(
        conv.shape,
        lambda n, c, y, x: te.max(conv[n, c, y, x] + bias[c], 0),
        name="bias_relu",
    )
    yield "padded_conv", [a, w, bias, out]

    a = te.placeholder((3, 5), name="input")
    rmax = te.reduce_axis((0, 5), name="rmax")
    maximum = te.compute((3,), lambda i: te.max(a[i, rmax], axis=rmax), name="row_max")
    exponent = te.compute(
        (3, 5), lambda i, j: te.exp(a[i, j] - maximum[i]), name="exponent"
    )
    rsum = te.reduce_axis((0, 5), name="rsum")
    total = te.compute(
        (3,), lambda i: te.sum(exponent[i, rsum], axis=rsum), name="row_sum"
    )
    out = te.compute((3, 5), lambda i, j: exponent[i, j] / total[i], name="softmax")
    yield "softmax", [a, out]

    a = te.placeholder((2, 3, 4), name="input")
    out = te.compute((6, 4), lambda i, j: a[i // 3, i % 3, j], name="reshape")
    yield "reshape", [a, out]

    a = te.placeholder((4, 4), name="input")
    shared = te.compute((4, 4), lambda i, j: a[i, j] + 1, name="shared")
    left = te.compute((4, 4), lambda i, j: shared[i, j] * 2, name="left")
    right = te.compute((4, 4), lambda i, j: shared[i, j] + 3, name="right")
    out = te.compute((4, 4), lambda i, j: left[i, j] + right[i, j], name="merge")
    yield "shared_branch", [a, out]
    yield "multiple_outputs", [a, right, left]

    a = te.placeholder((3, 4), name="half_input", dtype="float16")
    k = te.reduce_axis((0, 4), name="k")
    out = te.compute(
        (3,), lambda i: te.sum(a[i, k].astype("float32"), axis=k), name="accumulate"
    )
    yield "mixed_precision", [a, out]

    a = te.placeholder((5,), name="input")
    k = te.reduce_axis((1, 4), name="slice_k")
    out = te.compute((), lambda: te.sum(a[k], axis=k), name="slice_sum")
    yield "nonzero_axis_min", [a, out]

    scalar = te.compute((), lambda: tir.const(2.5, "float32"), name="scalar")
    yield "scalar_constant", [scalar]

    a = te.placeholder((2,), name="input_λ")
    out = te.compute((2,), lambda i: a[i] + 0.25, name="compute_λ")
    yield "unicode", [a, out]

    a = te.placeholder((2, 4), name="input")
    k = te.reduce_axis((0, 4), name="k")
    reducer = te.comm_reducer(
        lambda x, y: (x[0] + y[0], x[1] + y[1]),
        lambda t0, t1: (tir.const(0, t0), tir.const(0, t1)),
        name="pair_sum",
    )
    first, second = te.compute(
        (2,), lambda i: reducer((a[i, k], a[i, k] * a[i, k]), axis=k), name="tuple_sum"
    )
    yield "tuple_reducer", [a, first, second]


def dimension(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return str(value)


def main():
    destination = Path(__file__).parent
    tasks, references = [], []
    for name, tensors in programs():
        dag = auto_scheduler.ComputeDAG(tensors)
        task = auto_scheduler.SearchTask(
            compute_dag=dag,
            workload_key=dag.workload_key(),
            target="llvm",
            task_inputs=["validation_input"],
        )
        graph = json.loads(dag.__getstate__()["tensors"])
        references.append(
            {
                "name": name,
                "workload_key": str(task.workload_key),
                "tensor_graph_sha256": hashlib.sha256(
                    json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "tensors": [
                    {
                        "name": str(tensor.name),
                        "shape": [dimension(dim) for dim in tensor.shape],
                        "dtype": str(tensor.dtype),
                    }
                    for tensor in tensors
                ],
                "stages": [
                    {
                        "name": str(operation.name),
                        "type": type(operation).__name__,
                        "spatial": [
                            {
                                "variable": str(axis.var.name),
                                "min": dimension(axis.dom.min),
                                "extent": dimension(axis.dom.extent),
                            }
                            for axis in getattr(operation, "axis", [])
                        ],
                        "reduction": [
                            {
                                "variable": str(axis.var.name),
                                "min": dimension(axis.dom.min),
                                "extent": dimension(axis.dom.extent),
                            }
                            for axis in getattr(operation, "reduce_axis", [])
                        ],
                    }
                    for operation in dag.ops
                ],
            }
        )
        tasks.append(task)
    with (destination / "native_registry.pkl").open("wb") as stream:
        pickle.dump(tasks, stream, protocol=4)
    reference = {
        "source": "synthetic TE programs generated by original TenSet TVM",
        "source_commit": "35774ed373f9693cc6efcbe83f5703141e8ab0c8",
        "tvm_version": tvm.__version__,
        "cases": references,
    }
    (destination / "native_reference.json").write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(tasks)} native TVM program fixtures")


if __name__ == "__main__":
    main()
