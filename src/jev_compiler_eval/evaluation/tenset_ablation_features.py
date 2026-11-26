"""Runtime-blind structural controls for the TenSet context ablation.

TenSet schedules have no single analogue of a TPU tile size. These fixed
features describe recorded transformations; their lexicographic order is a
position-bias control, never a predicted runtime order. The diversity filter
uses farthest-first selection in a min-max-normalized feature space to retain
different schedule structures. It neither trains on labels nor claims to
retain the fastest schedules. Filtered-pool Random and oracle diagnostics are
required to interpret its results.

Features are computed only from ``Candidate.configuration``. Initial graph
state, candidate IDs, and any extra configuration fields do not contribute to
the feature vector. Candidate IDs seed the starting point and selection ties.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from math import log

import numpy as np

from jev_compiler_eval.datasets.tenset import TRANSFORMATIONS
from jev_compiler_eval.types import Candidate

FEATURE_VERSION = "tenset-structural-diversity-v1"
_OPCODES = tuple(sorted(TRANSFORMATIONS))
FEATURE_NAMES = (
    "step_count",
    *(f"step_{code}" for code in _OPCODES),
    "step_unknown",
    "split_explicit_count",
    "split_inferred_count",
    "split_log_sum",
    "split_log_max",
    *(f"annotation_{code}" for code in range(12)),
    "annotation_other",
    "unroll_pragma_count",
    "unroll_log_max",
)
_UNROLL = re.compile(r"auto_unroll_max_step\$(\d+)\Z")


def schedule_features(candidate: Candidate) -> dict[str, float]:
    """Summarize schedule syntax without reading measurements or replaying TVM.

    SP argument positions follow the pinned record writer: lengths are the
    fourth argument. Positive integer lengths contribute ``log(1 + length)``;
    null and nonpositive integer lengths count as inferred/unspecified. This
    is a syntax summary, not a reconstructed tile volume. Boolean/floating
    lengths are rejected instead of being interpreted as integer factors.
    AN's third argument is its annotation enum; PR's third is its pragma.
    """
    features = dict.fromkeys(FEATURE_NAMES, 0.0)
    steps = candidate.configuration.get("transformations", [])
    if not isinstance(steps, (list, tuple)):
        raise ValueError("schedule transformations must be a sequence")
    for step in steps:
        if not isinstance(step, Mapping) or not isinstance(step.get("code"), str):
            raise ValueError("schedule transformations must have string opcodes")
        code = step["code"]
        args = step.get("arguments", [])
        if not isinstance(args, (list, tuple)):
            raise ValueError("schedule transformation arguments must be a sequence")
        features["step_count"] += 1.0
        features[f"step_{code}" if code in _OPCODES else "step_unknown"] += 1.0
        if code == "SP":
            if len(args) != 5 or not isinstance(args[3], (list, tuple)):
                raise ValueError("SP must contain five arguments with a lengths list")
            for length in args[3]:
                if length is None:
                    features["split_inferred_count"] += 1.0
                elif isinstance(length, bool) or not isinstance(length, int):
                    raise ValueError("SP lengths must be integers or null")
                elif length <= 0:
                    features["split_inferred_count"] += 1.0
                else:
                    value = log(length + 1)
                    features["split_explicit_count"] += 1.0
                    features["split_log_sum"] += value
                    features["split_log_max"] = max(features["split_log_max"], value)
        elif code == "AN":
            if (
                len(args) != 3
                or isinstance(args[2], bool)
                or not isinstance(args[2], int)
            ):
                raise ValueError("AN must contain three arguments ending in an integer")
            annotation = args[2]
            name = (
                f"annotation_{annotation}"
                if 0 <= annotation < 12
                else "annotation_other"
            )
            features[name] += 1.0
        elif code == "PR":
            if len(args) != 3 or not isinstance(args[2], str):
                raise ValueError("PR must contain three arguments ending in a pragma")
            if match := _UNROLL.fullmatch(args[2]):
                features["unroll_pragma_count"] += 1.0
                features["unroll_log_max"] = max(
                    features["unroll_log_max"], log(int(match[1]) + 1)
                )
    return features


def feature_signature(candidate: Candidate) -> tuple[float, ...]:
    """Return the fixed structural tuple used for ascending/descending controls."""
    features = schedule_features(candidate)
    return tuple(features[name] for name in FEATURE_NAMES)


def select_diverse_candidates(
    candidates: Sequence[Candidate], keep: int, seed: int, workload_id: str
) -> list[Candidate]:
    """Keep distinct structural vectors, breaking selection ties by a seeded hash.

    Each feature is scaled to [0, 1] within the input pool; constant columns
    become zero. Start with the smallest seeded candidate hash, then greedily
    maximize squared distance to the closest chosen vector. This is equivalent
    to farthest-first Euclidean selection without computing square roots.
    Selection is independent of input order, while returned candidates retain
    their input order for presentation. Identical feature vectors are allowed
    and are resolved by the same seeded hash. No global RNG state is changed.
    """
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("keep must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(workload_id, str) or not workload_id.strip():
        raise ValueError("workload_id must be a nonempty string")
    pool = list(candidates)
    identifiers = [candidate.candidate_id for candidate in pool]
    if len(set(identifiers)) != len(pool):
        raise ValueError("diversity selection requires unique candidate IDs")
    # Validate all feature records even when no candidate is pruned.
    vectors = [feature_signature(candidate) for candidate in pool]
    if keep >= len(pool):
        return pool

    def priority(index: int) -> tuple[bytes, str]:
        payload = json.dumps(
            [seed, workload_id, identifiers[index]], separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).digest(), identifiers[index]

    indices = sorted(range(len(pool)), key=priority)
    matrix = np.asarray([vectors[index] for index in indices], dtype=np.float64)
    minimum = matrix.min(axis=0)
    span = matrix.max(axis=0) - minimum
    matrix = (matrix - minimum) / np.where(span > 0, span, 1.0)
    selected = np.zeros(len(pool), dtype=bool)
    distance = np.full(len(pool), np.inf)
    choice = 0
    for _ in range(keep):
        selected[choice] = True
        delta = matrix - matrix[choice]
        distance = np.minimum(distance, np.einsum("ij,ij->i", delta, delta))
        distance[selected] = -np.inf
        choice = int(np.argmax(distance))
    chosen_ids = {identifiers[indices[index]] for index in np.flatnonzero(selected)}
    return [candidate for candidate in pool if candidate.candidate_id in chosen_ids]
