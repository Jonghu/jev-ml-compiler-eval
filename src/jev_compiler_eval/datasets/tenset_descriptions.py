"""Constant descriptions of the pinned TenSet schedule record format.

The descriptions explain syntax and index semantics only. They neither inspect
an evaluation case nor suggest which scheduling choices to make.
"""

import json

_FIELDS_JSON = json.dumps(
    {
        "format": "tenset-configuration-fields-v1",
        "source": "tlc-pack/tenset@35774ed373f9693cc6efcbe83f5703141e8ab0c8",
        "source_files": [
            "include/tvm/auto_scheduler/transform_step.h",
            "src/auto_scheduler/transform_step.cc",
            "src/auto_scheduler/measure_record.cc",
            "src/auto_scheduler/loop_state.cc",
            "src/auto_scheduler/compute_dag.cc",
            "include/tvm/te/schedule.h",
        ],
        "initial_state": (
            "The serialized State stage-array slot. The pinned writer always emits "
            "[] here; this omits stages, rather than declaring an empty program. "
            "Initial stages come from the workload's ComputeDAG."
        ),
        "transformations": (
            "An ordered sequence of schedule steps. Each step has code (the record "
            "opcode), name (its readable name), and arguments (positional values in "
            "the order documented below, excluding the opcode). Replay the sequence "
            "in order: arguments refer to the state immediately before that step."
        ),
        "indices": {
            "base": 0,
            "stage_id": (
                "Index in the current stage list, including placeholder stages. "
                "CHR, CHW and RF insert stages and can shift later stage indices."
            ),
            "iter_id": (
                "Index in a current stage's iterator list. Initially a compute "
                "stage has spatial axes followed by reduction axes; placeholders "
                "have no iterators. SP, FSP, FFSP, FU and RE change iterator indices."
            ),
            "src_step_id": (
                "Index in the complete transformation history, not a stage index "
                "or an index restricted to split steps; the referenced entry is SP."
            ),
            "graph_ids": (
                "compute_dag.nodes uses graph-local IDs. Initially "
                "compute_dag.stage_order[stage_id] identifies that stage's operation "
                "node; graph IDs are not stage indices. This initial mapping does "
                "not incorporate stage insertions by subsequent transformations."
            ),
        },
        "annotation_codes": {
            "0": "none (ordinary for loop)",
            "1": "unroll",
            "2": "vectorize",
            "3": "parallel",
            "4": "vthread",
            "5": "blockIdx.x",
            "6": "threadIdx.x",
            "7": "blockIdx.y",
            "8": "threadIdx.y",
            "9": "blockIdx.z",
            "10": "threadIdx.z",
            "11": "tensorize (enum entry; unsupported by this fork's AN replay)",
        },
        "steps": {
            "AN": {
                "name": "annotate",
                "arguments": ["stage_id", "iter_id", "annotation"],
                "meaning": (
                    "Annotate the selected iterator using annotation_codes. "
                    "The pinned schedule replay handles codes 0 through 10; "
                    "the enum's tensorize code 11 has no AN replay implementation."
                ),
            },
            "FU": {
                "name": "fuse",
                "arguments": ["stage_id", "fused_ids"],
                "meaning": (
                    "Fuse the listed consecutive, increasing current iterator "
                    "indices into one iterator at the first listed position. "
                    "The fused extent is the product of their extents when known. "
                    "An empty list creates a unit iterator."
                ),
            },
            "PR": {
                "name": "pragma",
                "arguments": ["stage_id", "iter_id", "pragma_type"],
                "meaning": (
                    "Attach the pragma string to the selected iterator. "
                    "auto_unroll_max_step$N encodes the integer auto-unroll step "
                    "limit N; schedule replay also sets unroll_explicit=true. "
                    "debug_skip_region is the other pragma supported by State replay."
                ),
            },
            "RE": {
                "name": "reorder",
                "arguments": ["stage_id", "after_ids"],
                "meaning": (
                    "Permute all current iterators: after_ids lists their old "
                    "indices in the new outer-to-inner order."
                ),
            },
            "SP": {
                "name": "split",
                "arguments": [
                    "stage_id",
                    "iter_id",
                    "extent",
                    "lengths",
                    "inner_to_outer",
                ],
                "meaning": (
                    "Split one iterator into len(lengths)+1 iterators. extent is "
                    "the recorded integer extent, or 0 when unspecified. lengths "
                    "contains explicit integer split values; the remaining extent "
                    "is inferred. Flag 1 replays lengths in reverse order with "
                    "factor splits; flag 0 replays them in forward order with "
                    "nparts splits. The resulting iterator list is outer-to-inner."
                ),
            },
            "FSP": {
                "name": "follow_split",
                "arguments": ["stage_id", "iter_id", "src_step_id", "n_split"],
                "meaning": (
                    "Follow a prior SP: take its first n_split-1 explicit lengths, "
                    "then append the product of its remaining explicit lengths "
                    "(empty product 1). This supplies n_split explicit lengths "
                    "for factor splitting, producing n_split+1 iterators."
                ),
            },
            "FFSP": {
                "name": "follow_fused_split",
                "arguments": [
                    "stage_id",
                    "iter_id",
                    "src_step_ids",
                    "level",
                    "factor_or_nparts",
                ],
                "meaning": (
                    "Multiply lengths[level] from each referenced prior SP. "
                    "level indexes the explicit lengths array, not resulting "
                    "iterators. Split into two iterators using that product: "
                    "flag 1 selects factor, flag 0 selects nparts."
                ),
            },
            "SA": {
                "name": "storage_align",
                "arguments": ["stage_id", "iter_id", "factor", "offset"],
                "meaning": (
                    "Set storage alignment on the selected current axis so its "
                    "stride equals k*factor+offset for some integer k. State replay "
                    "stores the offset; schedule replay applies factor and offset."
                ),
            },
            "CA": {
                "name": "compute_at",
                "arguments": ["stage_id", "target_stage_id", "target_iter_id"],
                "meaning": (
                    "Attach the source stage's computation at the selected iterator "
                    "of the target stage. State replay clears source iterator bounds "
                    "for later inference; this step does not insert a stage."
                ),
            },
            "CI": {
                "name": "compute_inline",
                "arguments": ["stage_id"],
                "meaning": (
                    "Inline the stage into its consumers. The State stage-list "
                    "entry remains and is marked inlined; later stages are not "
                    "renumbered by this step."
                ),
            },
            "CR": {
                "name": "compute_root",
                "arguments": ["stage_id"],
                "meaning": (
                    "Place the stage at the root, removing its compute-at "
                    "attachment. State replay clears iterator bounds for later "
                    "inference; this step does not insert or remove a stage."
                ),
            },
            "CHR": {
                "name": "cache_read",
                "arguments": ["stage_id", "scope_name", "reader_stage_ids"],
                "meaning": (
                    "Add a read-cache stage for output 0 of the source stage in "
                    "the named storage scope and redirect the listed reader stages "
                    "through it. The new stage is inserted at stage_id+1, shifting "
                    "existing stages at and after that position."
                ),
            },
            "CHW": {
                "name": "cache_write",
                "arguments": ["stage_id", "scope_name"],
                "meaning": (
                    "Cache writes of all outputs of the target stage in the named "
                    "storage scope. Insert a cache stage at stage_id before the "
                    "target stage and rebuild affected stages. The pinned fork "
                    "also handles two added stages for a multiple-output case, "
                    "so the shift is not always one stage."
                ),
            },
            "RF": {
                "name": "rfactor",
                "arguments": ["stage_id", "iter_id", "factor_iter_id"],
                "meaning": (
                    "Factor the selected reduction iterator into an intermediate "
                    "tensor. factor_iter_id is the insertion position of its new "
                    "axis, not an existing iterator reference. Insert the new "
                    "compute stage at stage_id before the original target stage."
                ),
            },
        },
    },
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
)


def tenset_configuration_fields() -> dict[str, object]:
    """Return a fresh, label-checked field guide for one workload context."""
    # Import lazily: the TenSet adapter also imports this description provider.
    from jev_compiler_eval.datasets.tenset import _reject_label_keys

    fields = json.loads(_FIELDS_JSON)
    _reject_label_keys(fields)
    return fields
