# TenSet contribution: three slides

- [Editable PowerPoint](tenset_rq1_slides.pptx)
- [PDF](tenset_rq1_slides.pdf)
- [Speaker notes and source details](speaker_notes.md)

These three slides cover Jev versus Random only:

1. The main Platinum selection study and the separate E5 candidate-size setup.
2. Record-weighted Top-1/Top-5 results for all 484 Platinum workloads, with the
   existing exploratory paired-workload intervals and logged input-cost estimate.
3. Completed E5 results at 16, 32, and 50 candidates on the same 120-workload
   cohort, including editable score charts and a Pairwise Accuracy/regret table.

The studies are not pooled. Platinum scores are test-record-weighted; E5 scores
are workload macro means within each candidate budget. The E5 selection-score
gain intervals cross zero at every budget, while strict Pairwise Accuracy gain
intervals are positive in the exploratory, unadjusted analysis. Cross-budget
advantage contrasts remain uncertain. The charts and text in the PowerPoint are
editable, and presenter notes include methods, source details, and limitations.

Sources: the [RQ1 report](../../rq1-report.md),
[reviewed Platinum results](../../results/tenset/tutorial-500-platinum-test/README.md),
the [reviewed E5 50-candidate results](../../results/tenset/w120-c50-s42-long-context/metrics.csv),
and the [completed candidate-size sweep](../../results/tenset/size-sweep/README.md).

Operation labels are available in the uploaded analysis: 27 literal ComputeOp
signatures across the E5 cohort. Full tensor-shape and selected/oracle schedule
explanations still require the prepared-data case export. Fresh API repeatability
has not been measured; archived response reuse is not an independent repeat.
Use the [finish guide](../../run-tenset-finish-rq1.md) for remaining offline exports.
