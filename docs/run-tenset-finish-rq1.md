# Finish RQ1: Jev versus Random

The current research comparison is **Jev versus Random**. No specialized-model
training, checkpoint, or published-model comparison is required. Keep the two
completed studies separate:

| Study | Purpose | Metrics to report |
|---|---|---|
| E5-2673, 120 workloads x 50 valid schedules | Single-question schedule ranking | Top-1, Top-5, Pairwise Accuracy, runtime regret |
| Platinum-8272, 484 workloads and 187,679 test records | Larger tournament selection study | Record-weighted Top-1/Top-5, failure and elimination diagnostics |

The tournament's eliminated tail is a deterministic presentation order, not a
global learned ranking. Its retained failed-measurement costs are placeholders,
not physical execution latencies. Do not manufacture tournament Pairwise
Accuracy or interpret a mean sentinel-based regret as a measured slowdown.

The [RQ1 report](rq1-report.md), [run log](tenset-runlog.md), and
[reviewed Platinum results](results/tenset/tutorial-500-platinum-test/README.md)
contain the completed numerical results. The steps below prepare the remaining
workload-type examples and controlled size comparison.

The revised three-slide contribution is also available as an
[editable PowerPoint](presentations/tenset-rq1/tenset_rq1_slides.pptx),
[PDF](presentations/tenset-rq1/tenset_rq1_slides.pdf), and
[speaker notes](presentations/tenset-rq1/speaker_notes.md).

## 1. Update the Mac checkout

```bash
cd "$HOME/Projects/jev-ml-compiler-eval"
git fetch origin
git switch feat/tenset-rq1-completion
git pull --ff-only
uv sync --locked --extra tenset
```

Keep the existing prepared files, completed result bundles, and caches. Git
refuses a branch switch that would overwrite local edits; preserve those edits
before continuing. None of the commands in sections 2 or 3 makes API calls.

Historical completed runs were made at their recorded commits. The offline
analysis reads those saved artifacts without rerunning their models. Do not
resume an old execution under changed code: resume identities intentionally
require the original implementation and environment.

## 2. Export workload types and schedule examples offline

Run the context-aware analyzer on the complete Platinum study:

```bash
uv run python scripts/analyze_tenset_cases.py \
  --results results/tenset/tutorial-500-platinum-test \
  --dataset data/tenset/tutorial-500-platinum/index.json \
  --output-dir results/tenset/tutorial-500-platinum-case-analysis
```

Also analyze the earlier E5 study, keeping its separate target and metrics:

```bash
uv run python scripts/analyze_tenset_cases.py \
  --results results/tenset/random-vs-jev-w120-c50-s42-long-context \
  --dataset data/tenset/prepared-w120-c50-s42-metadata.json \
  --output-dir results/tenset/e5-w120-c50-case-analysis
```

The case exporter reports Top-1/Top-5 by type. For the E5 experiment's full
declared metric set, including Pairwise Accuracy and regret by operation, run
the generic offline analyzer separately:

```bash
uv run python scripts/analyze_tenset.py \
  --results results/tenset/random-vs-jev-w120-c50-s42-long-context \
  --dataset data/tenset/prepared-w120-c50-s42-metadata.json \
  --output-dir results/tenset/e5-w120-c50-full-metric-analysis
```

The script validates saved candidate pools and fingerprints against the actual
prepared data before joining tensor metadata. It exports per-workload scores,
operation/type and candidate-count tables, and post-hoc success/failure examples
with the Jev, Random, and oracle schedules. Labels, configurations, and source
record identities are read for offline analysis; none is sent to Jev.

Operation groups come from the actual program metadata. Any coarse family
classification is documented in the export; unknown types remain unknown.
Shapes and transformations in the case-study JSON can be inspected to explain
individual outcomes. These examples illustrate the observed results; selecting
large gains or losses is not a new representative evaluation.

Inspect the case export's `analysis-manifest.json`, `by_operation.csv`,
`by_operation_family.csv`, and `case_studies.md`. Preserve the original
`case_studies.json` for complete transformations. If an analysis output directory
already exists, choose a new output name rather than overwriting it.

The prepared shards are only on the Mac, so the cloud audit could not produce
these semantic exports. They are ready-to-run analysis, not fabricated finished
workload-type results.

## 3. Freeze the controlled candidate-size sweep offline

Use the same 120 E5 workloads at 16, 32, and 50 schedules. Smaller pools are
prefixes of the same seeded order; they are not selected by measured speed.
The target, seed 42, metadata, no-description input, long-context mode, model
version, and metric definitions stay fixed. This is separate from the Platinum
tournament and its record-level split.

```bash
uv run python scripts/prepare_tenset_size_sweep.py \
  --prepared data/tenset/prepared-w120-c50-s42-metadata.json \
  --baseline-results results/tenset/random-vs-jev-w120-c50-s42-long-context
```

The helper checks the audited prepared-file SHA-256 and the entire cohort. It
validates the 50-schedule baseline's candidate IDs, inputs, labels, rankings,
model settings, and metrics before reusing it. It never alters that bundle.
If the validation fails, investigate the difference instead of bypassing it.
If no reusable 50-schedule bundle is available, omit `--baseline-results` to
prepare a separate new 50-schedule run.

Local profiles and the frozen manifest are written to
`experiments/tenset/local/size-sweep/`. Canonical templates are committed under
`experiments/tenset/size-sweep/`; local machine paths are ignored by Git.
Preparation prints the applicable preflight, execution, and analysis commands.
Keep the data and generated profiles unchanged after freezing them.

Validate the 16- and 32-schedule members without credentials or model calls:

```bash
uv run python scripts/run_tenset_size_sweep.py \
  --manifest experiments/tenset/local/size-sweep/manifest.json \
  --candidate-count 16 --preflight-only

uv run python scripts/run_tenset_size_sweep.py \
  --manifest experiments/tenset/local/size-sweep/manifest.json \
  --candidate-count 32 --preflight-only
```

The wrapper rechecks the frozen data and configuration before execution. A
different file hash is not automatically interchangeable with the audited file.
Custom data can use an explicit expected checksum when preparing a new study;
that does not establish identity with the historical benchmark.

## 4. Run only the new size experiments when ready

The following commands make **paid Jev API calls**. They were prepared and
tested offline, not executed as part of this update. The earlier approximately
7.62 USD cost belongs to the completed Platinum study, not to this sweep.

The key is `TYPESAFE_API_KEY` in the repository-root `.env`. An exported shell
value takes precedence. To use the value in `.env` without displaying it:

```bash
unset TYPESAFE_API_KEY
```

```bash
uv run python scripts/run_tenset_size_sweep.py \
  --manifest experiments/tenset/local/size-sweep/manifest.json \
  --candidate-count 16

uv run python scripts/run_tenset_size_sweep.py \
  --manifest experiments/tenset/local/size-sweep/manifest.json \
  --candidate-count 32
```

The existing 50-schedule anchor requires no new API calls when reuse was
verified. If preparation instead scheduled a new 50-schedule member, execute
the printed command for that member. Nonempty result directories are protected;
keep successful cached responses and the exact frozen settings after an
interruption. The single-question runner does not checkpoint per-workload
completion like the tournament collection runner, so do not assume `--resume`
is available for this sweep. Consult the helper's printed status and preserve
any partial output before choosing a new destination.

After the new members finish, run the analysis command printed during
preparation. Use `by_candidate_count.csv` and paired per-workload rows to discuss
size effects. An overall average pooled across sizes is not the size comparison.
The plot and table should show the number of workloads and the 16/32/50 budgets.
Compare Jev and Random within each budget. Increasing the pool can change its
fastest schedule, so a cross-size score change alone is not a physical speedup.

Generate the dedicated candidate-size figure from the completed analysis:

```bash
uv run python scripts/plot_tenset_size_sweep.py \
  --analysis-dir results/tenset/size-sweep/analysis
```

The generic analyzer's overall comparison plot pools sizes. Use the dedicated
size figure for this ablation; it checks matched workload IDs and displays each
budget separately.

The current Platinum natural-size breakdown (468 workloads with 400 candidates,
16 with 6–88) is already published as descriptive analysis. It does not isolate
candidate-count effects, because workload identity changes between those groups.

## 5. Complete the report checklist

- [x] State the Jev-versus-Random scope and distinguish the two completed studies.
- [x] Publish Top-1/Top-5, the E5 Pairwise Accuracy/regret, and reproducible metrics.
- [x] Include paired uncertainty and a separately labelled exact Random expectation.
- [x] Record failure-sentinel choices, tournament attrition, tokens, and cached usage.
- [x] Prepare plots, updated slides, and the context-aware export workflow.
- [ ] Run the Mac metadata export and write the workload-type and schedule examples.
- [ ] Run the new 16/32 experiments and incorporate the controlled size comparison.

Fresh API repeatability checks, field-description/context/order ablations, and
expansion to all raw schedules are useful optional extensions. Finish the
metadata/case exports and the controlled size comparison before expanding scope.

For final review, the compact Mac exports to share are `by_operation.csv`,
`by_operation_family.csv`, `per_workload.csv`, and `case_studies.json` from each
case-analysis directory. The frozen prepared data and API key do not need to be
shared. Add the completed size analysis and its dedicated plot once available.
