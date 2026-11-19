# Operation-signature and case review

This review keeps the 16-, 32-, and 50-candidate experiments separate. It recomputes the supplied operation aggregates from the paired per-workload exports; it is not a new audit of raw measured records or API payloads.

## Coverage

There are 120 distinct workloads and 27 literal operation signatures at each budget. Eight signatures contain at least five workloads (85/120 workloads); nineteen signatures have fewer than five. The largest signature contains 40 workloads. Shape and transformation bodies are absent from these files.

## Larger descriptive groups by budget

These are fused-program operation-name signatures. The n >= 5 display threshold does not establish statistical reliability. All 27 signatures remain in by-operation-paired.csv.

### 16 candidates

| Operation signature | n | Jev Top-1 | Random Top-1 | Difference | Jev Top-5 | Random Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| PaddedInput+Conv2dOutput+T_add+T_relu | 40 | 0.3559 | 0.3388 | +0.0171 | 0.5795 | 0.5726 |
| PaddedInput+Conv2dOutput+T_add | 8 | 0.4435 | 0.3945 | +0.0490 | 0.6689 | 0.5940 |
| PaddedInput+Conv2dOutput+T_add+T_add | 7 | 0.3850 | 0.4577 | -0.0728 | 0.5037 | 0.6311 |
| tensor+tensor | 7 | 0.5372 | 0.5537 | -0.0165 | 0.7540 | 0.6414 |
| PaddedInput+Conv2dOutput | 6 | 0.4064 | 0.4235 | -0.0172 | 0.5750 | 0.6601 |
| PaddedInput+Conv2dOutput+T_add+T_add+T_relu | 6 | 0.5439 | 0.3752 | +0.1687 | 0.7068 | 0.6229 |
| PaddedInput+Conv3dOutput+T_multiply+T_add+T_relu | 6 | 0.3011 | 0.2964 | +0.0047 | 0.4712 | 0.4723 |
| PaddedInput+DepthwiseConv2d+T_add+compile_engine_const+T_add+compute+compile_engine_const+T_divide+T_multiply | 5 | 0.6379 | 0.3269 | +0.3110 | 0.7965 | 0.6997 |

### 32 candidates

| Operation signature | n | Jev Top-1 | Random Top-1 | Difference | Jev Top-5 | Random Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| PaddedInput+Conv2dOutput+T_add+T_relu | 40 | 0.3293 | 0.3079 | +0.0214 | 0.5032 | 0.5318 |
| PaddedInput+Conv2dOutput+T_add | 8 | 0.4237 | 0.2692 | +0.1545 | 0.6333 | 0.5361 |
| PaddedInput+Conv2dOutput+T_add+T_add | 7 | 0.2672 | 0.2052 | +0.0619 | 0.3344 | 0.3258 |
| tensor+tensor | 7 | 0.5045 | 0.3168 | +0.1876 | 0.7065 | 0.5954 |
| PaddedInput+Conv2dOutput | 6 | 0.2975 | 0.1471 | +0.1504 | 0.5335 | 0.4359 |
| PaddedInput+Conv2dOutput+T_add+T_add+T_relu | 6 | 0.3015 | 0.4907 | -0.1892 | 0.4596 | 0.6809 |
| PaddedInput+Conv3dOutput+T_multiply+T_add+T_relu | 6 | 0.3849 | 0.3163 | +0.0686 | 0.4555 | 0.5602 |
| PaddedInput+DepthwiseConv2d+T_add+compile_engine_const+T_add+compute+compile_engine_const+T_divide+T_multiply | 5 | 0.6023 | 0.5517 | +0.0506 | 0.7373 | 0.6805 |

### 50 candidates

| Operation signature | n | Jev Top-1 | Random Top-1 | Difference | Jev Top-5 | Random Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| PaddedInput+Conv2dOutput+T_add+T_relu | 40 | 0.3085 | 0.2564 | +0.0521 | 0.4566 | 0.4153 |
| PaddedInput+Conv2dOutput+T_add | 8 | 0.2927 | 0.3480 | -0.0553 | 0.5341 | 0.5698 |
| PaddedInput+Conv2dOutput+T_add+T_add | 7 | 0.2717 | 0.2673 | +0.0044 | 0.3657 | 0.4626 |
| tensor+tensor | 7 | 0.4348 | 0.5936 | -0.1587 | 0.6212 | 0.6890 |
| PaddedInput+Conv2dOutput | 6 | 0.1640 | 0.3185 | -0.1545 | 0.5158 | 0.4649 |
| PaddedInput+Conv2dOutput+T_add+T_add+T_relu | 6 | 0.3366 | 0.4737 | -0.1372 | 0.6076 | 0.5596 |
| PaddedInput+Conv3dOutput+T_multiply+T_add+T_relu | 6 | 0.2194 | 0.2895 | -0.0700 | 0.3697 | 0.4024 |
| PaddedInput+DepthwiseConv2d+T_add+compile_engine_const+T_add+compute+compile_engine_const+T_divide+T_multiply | 5 | 0.6221 | 0.5284 | +0.0937 | 0.7125 | 0.7712 |

## Supported descriptive observations

- The largest signature, PaddedInput+Conv2dOutput+T_add+T_relu (n=40), has positive average Top-1 differences at all three budgets: +0.0171, +0.0214, and +0.0521. Its Top-5 difference is slightly positive at 16, negative at 32, and positive at 50.
- The five-workload fused signature containing DepthwiseConv2d, constants, divide, and multiply has positive Top-1 differences at all three budgets (+0.3110, +0.0506, +0.0937). Its small count precludes a broad claim about all depthwise convolutions.
- PaddedInput+Conv2dOutput+T_add+T_add+T_relu (n=6) changes from a positive Top-1 difference at 16 (+0.1687) to negative differences at 32 (-0.1892) and 50 (-0.1372). This prevents a blanket claim that Jev consistently favors or outperforms Random on convolution signatures.
- The generic tensor+tensor signature (n=7) has insufficient names to identify a mathematical operation family. It should remain literally labelled rather than called matmul.

## Post-hoc examples

The table below shows one positive and one negative illustration per budget. Full top-three lists per direction are in posthoc-examples-by-budget.csv. Cost ratios are derived from the recorded regret; they do not reveal tensor shapes or schedule transformations.

| Budget | Outcome | Workload | Operation signature | Jev Top-1 | Random Top-1 | Difference | Jev / oracle cost | Random / oracle cost |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 16 | success | tenset-e6cd98ac634fba7bab1d171c | PaddedInput+Conv2dOutput+T_add | 1.0000 | 0.0343 | +0.9657 | 1.000 | 29.150 |
| 16 | failure | tenset-eaf65c7fa74cd53739bbc484 | PaddedInput+Conv2dOutput+T_add+T_relu | 0.0713 | 1.0000 | -0.9287 | 14.023 | 1.000 |
| 32 | success | tenset-e6cd98ac634fba7bab1d171c | PaddedInput+Conv2dOutput+T_add | 1.0000 | 0.0573 | +0.9427 | 1.000 | 17.453 |
| 32 | failure | tenset-218874f0ed0a9123e5f45364 | PaddedInput+Conv2dOutput+T_add+T_add+T_relu | 0.0658 | 1.0000 | -0.9342 | 15.207 | 1.000 |
| 50 | success | tenset-e89628f154df988428fa910a | PadInput+conv2d_transpose_nhwc+T_multiply+T_add+T_relu | 1.0000 | 0.0303 | +0.9697 | 1.000 | 33.025 |
| 50 | failure | tenset-1c06dd347db18aa8ffcdc91b | compute | 0.2009 | 1.0000 | -0.7991 | 4.977 | 1.000 |

The original jev_cases.csv contains ten cases selected across runs, rather than balanced examples from each budget. All ten entries agree with their source paired rows. The new per-budget selection avoids pooling budgets but remains post-hoc; it does not explain why Jev made a decision.

## Still missing for semantic case studies

- Tensor shapes and full program metadata.
- Selected, Random, and oracle candidate identities and complete scheduling transformations.
- Raw runtime records or request payloads for independent source/label-isolation verification.

The Mac-side analyze_tenset_cases.py exporter can supply these details without API calls. Do not infer them from hashed workload IDs or joined operation names.
