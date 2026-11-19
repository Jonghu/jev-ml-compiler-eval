# TpuGraphs tile:xla workstream

This document records how the TpuGraphs `tile:xla` collection is obtained,
which data is evaluated, and how kernels are presented to decision models.

## Source

- Dataset: [TpuGraphs](https://github.com/google-research-datasets/tpu_graphs),
  version `v0`, collection `tile:xla`.
- Paper: Phothilimthana et al., "TpuGraphs: A Performance Prediction Dataset
  on Large Tensor Computational Graphs", arXiv:2308.13490.
- Feature names and opcode IDs in
  `src/jev_compiler_eval/datasets/tpugraphs_features.py` were generated from
  the upstream README ("Node Features" and "Tile Config Features") and
  `tpu_graphs/process_data/xla/hlo_opcode.h`.

## Download

```bash
uv run python scripts/download_tpugraphs.py
```

The script downloads each archive to `data/tpugraphs/archives/`, verifies the
SHA-256 checksums below, and extracts to
`data/tpugraphs/npz/tile/xla/{train,valid,test}/`. Use `--splits valid` to
fetch only the evaluation split. Rerunning reuses verified archives and fully
extracted splits, and an existing archive with a different checksum is
reported rather than overwritten.

| Archive | Bytes | SHA-256 |
|---|---|---|
| `npz_tile_xla_train.tar` | 158,965,760 | `ef5c9b5a2e14ac5eb0c83e1f6421fe177e53fa72a65268ec58dcefa342fe25de` |
| `npz_tile_xla_valid.tar` | 15,923,200 | `2a56dc731f59f6ffa7d312e8d07aa15307c136dd6bc3bb7bc04dc5fa6d526803` |
| `npz_tile_xla_test.tar` | 15,912,960 | `a10487a99f33a1a64093507a02be1fe6006fc4480094d36549b00fbec87007d5` |

## Splits

| Split | Kernels | Programs | Configs per kernel (median / max) | Labels |
|---|---|---|---|---|
| `train` | 5,709 | 61 | 842 / 19,991 | Yes |
| `valid` | 676 | 7 | 604 / 9,997 | Yes |
| `test` | 844 | Hidden | 596 / 10,000 | No (all runtimes are zero) |

Experiments evaluate on the official `valid` split because the `test` split
has no runtime labels. The `train` split is reserved for training learned
baselines.

## Preprocessing

The `tpugraphs-tile` dataset adapter applies these rules:

- **One workload per kernel.** The workload ID is the `.npz` file stem.
- **Normalized runtimes.** Normalizers vary within 523 of 676 validation
  kernels, by up to 5%. The official target is
  `config_runtime / config_runtime_normalizers`. The adapter stores
  `config_runtime / normalizer × normalizer[0]`, converted from nanoseconds to
  milliseconds. Rankings and runtime regret therefore match the official
  target.
- **Duplicate configurations.** 580 of 676 validation kernels contain identical
  configuration rows with slightly different measurements. Identical rows are
  merged into one candidate, named `config_<first row index>`, that keeps the
  fastest normalized runtime.
- **Candidate limit.** Kernels with more than `max_candidates` distinct
  configurations are skipped. This keeps requests within API limits until
  shared candidate sampling exists. It biases the subset toward small kernels:
  `max_candidates: 50` keeps 120 validation kernels from 5 programs.

Configuration 0 is the compiler's default configuration in every validation
kernel. The best configuration improves on the default by a median of 0.7%
and by at most 34%.

## Model-visible serialization

`Workload.context` contains the collection, program name, node and edge
counts, and one object per node:

- `opcode`, `dtype`, `shape`, and `layout_minor_to_major`;
- `operands`, derived from `edge_index`;
- `is_root` and `parameter_number` when applicable;
- convolution dimension numbers for convolution nodes, where zero is a valid
  dimension index;
- every other nonzero feature group, with trailing zero padding removed.

`_sum` and `_product` features are omitted because they are derivable.
Zero padding is always trailing in the `train` and `valid` splits, so tensor
rank is inferred by stripping trailing zeros.

`Candidate.configuration` contains the nonzero `kernel_bounds`,
`output_bounds`, and `input_bounds` lists. Runtimes and normalizers are never
part of model-visible data.

The optional dataset setting `describe_config_fields: true` adds a
`configuration_fields` object to the context, before `nodes`. It restates the
official README definitions of the three bound groups and the tile size
selection task from the TpuGraphs paper. It does not state units, because the
official sources do not, and it does not say which tile sizes tend to be
faster. The default (`false`) leaves the context unchanged, so earlier
requests and their cache entries remain valid. The setting lives in the
dataset adapter rather than in the shared Jev model, so other benchmarks are
unaffected.

## Experiments

```bash
uv run python scripts/run_experiment.py \
  --config experiments/tpugraphs/tile-valid-random-smoke-test.yaml
uv run python scripts/run_experiment.py \
  --config experiments/tpugraphs/tile-valid-random-vs-jev.yaml
```
