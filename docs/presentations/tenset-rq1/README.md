# TenSet contribution: three slides

- [Editable PowerPoint](tenset_rq1_slides.pptx)
- [PDF](tenset_rq1_slides.pdf)
- [Speaker notes and source details](speaker_notes.md)

These three slides report our paired Jev-versus-Random experiments and include a
separate published XGBoost tutorial reference for context:

1. The main Platinum selection study and the separate E5 candidate-size setup.
2. Record-weighted Top-1/Top-5 results for all 484 Platinum workloads, with
   exploratory paired-workload intervals and the logged input-cost estimate.
   A separate reference band shows published XGBoost Top-1 0.829146 and
   Top-5 0.904636 from the official TenSet 500-file tutorial.
3. Completed E5 results at 16, 32, and 50 candidates on the same 120-workload
   cohort, including editable score charts and a Pairwise Accuracy/regret table.

The studies are not pooled. Platinum scores are test-record-weighted; E5 scores
are workload macro means within each candidate budget. The E5 selection-score
gain intervals cross zero at every budget, while strict Pairwise Accuracy gain
intervals are positive in the exploratory, unadjusted analysis. Cross-budget
advantage contrasts remain uncertain. Presenter notes also explain the separate
leading-DAG-hash group sensitivity and its limitations. Charts and text in the
PowerPoint are editable.

XGBoost is an external published tutorial aggregate, not a third model measured
in our experiment. Its exact original test inventory and checkpoint have not
been independently matched; count agreement alone does not establish identical
pools. The tutorial does not publish a Random result. Every Random score shown
in these slides comes from our own experiment. No paired statistical comparison
with XGBoost is claimed.

Sources: the [RQ1 report](../../rq1-report.md),
[reviewed Platinum results](../../results/tenset/tutorial-500-platinum-test/README.md),
the [published XGBoost reference and provenance](../../results/tenset/tutorial-500-platinum-test/xgboost-tutorial-reference.json),
the [pinned official tutorial](https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/docs/try_pretrained_models_on_aws_c59x.md#L78),
the [reviewed E5 50-candidate results](../../results/tenset/w120-c50-s42-long-context/metrics.csv),
and the [completed candidate-size sweep](../../results/tenset/size-sweep/README.md).

Operation labels are available in the uploaded analysis: 27 literal ComputeOp
signatures across the E5 cohort. Full tensor-shape and selected/oracle schedule
explanations still require the prepared-data case export. Fresh API repeatability
has not been measured; archived response reuse is not an independent repeat.
Use the [finish guide](../../run-tenset-finish-rq1.md) for remaining offline exports.
