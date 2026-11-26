# TenSet context ablation

Mean ± sample SD across fresh repeats; Top-5 below is an endpoint score.

| Phase | Variant | Repeats | Workloads | Top-1 score | Endpoint Top-5 score | Same pick |
|---|---|---:|---:|---:|---:|---:|
| full | a5 | 3 | 27 | 0.2678 ± 0.0179 | 0.4331 ± 0.0051 | 66.7% |
| small | a0 | 3 | 120 | 0.3582 ± 0.0045 | 0.6293 ± 0.0123 | 66.7% |
| small | a1 | 3 | 120 | 0.3218 ± 0.0021 | 0.6140 ± 0.0052 | 70.0% |
| small | a2 | 3 | 120 | 0.3557 ± 0.0024 | 0.6344 ± 0.0035 | 70.8% |
| small | a3 | 3 | 120 | 0.3319 ± 0.0076 | 0.6046 ± 0.0105 | 69.2% |
| small | a4-asc | 3 | 120 | 0.3380 ± 0.0049 | 0.6188 ± 0.0049 | 65.0% |
| small | a4-desc | 3 | 120 | 0.3176 ± 0.0203 | 0.6017 ± 0.0065 | 66.7% |
| small | a5 | 3 | 120 | 0.3410 ± 0.0037 | 0.5976 ± 0.0030 | 82.5% |

See baselines.csv for exact full-pool Random, filtered-pool Random and filter-oracle scores. All retain the original denominator.
random-reference.csv is one reproducible seeded Random draw, including the weighted network score where the paper fields exist. It has no repeated-model variation estimate.

Top-5 prefix scores are retained separately in the CSV files. Filtered tails have no reported pairwise quality. Paired tests are exploratory and unadjusted.

## Full-network weighted endpoints

These ratios use frequency-weighted sums of private paper normalizers and recovered selected latencies, separately from macro scores.

| Variant | Network Top-1 score | Network Top-5 score |
|---|---:|---:|
| a5 | 0.1134 ± 0.0078 | 0.3432 ± 0.0248 |
| random-42 (one draw) | 0.1373 | 0.4063 |
