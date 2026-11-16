# TenSet contribution: three slides

Revised scope: Jev versus Random only. Designed as the TenSet portion of a five-slide group presentation. All PowerPoint text and charts are editable. Suggested speaking time: about 2–3 minutes.

## Slide 1: Experiment setup

Our comparison is Jev against Random. The main selection study uses 484 Platinum workloads and 187,679 test schedule records, reconstructed from a fixed 500-file sample. Jev and Random receive identical test pools. Jev uses field definitions, long context, and a tournament for larger pools. We also completed an earlier E5 study with 120 workloads and 50 valid schedules each. That single-question study provides full-ranking metrics and runtime regret. We report the two studies separately.

### Presenter reference

A workload is a tensor program; a schedule is a compiler configuration. Main study: Intel Platinum 8272 measurements; 500 batch-size-1 files sampled with seed 0, 484 retained after the 48-record minimum. The original reverse-order within-task split uses int(0.9*N) training records and the remaining approximately 10% test records. Neither Jev nor Random is trained on the training portion. These are not 484 previously unseen workloads. The pipeline preserves measurement-error records: 2,013 are reported in the test pools. 468 workloads used two rounds and 16 used one. The Mac runs the client; Jev inference is served by the API, and program runtimes are existing dataset labels.
Separate E5-2673 study: 120 project-held-out workloads, 50 seeded, valid distinct schedules each, seed 42, no field definitions, long context, one question per workload. Failed schedules are excluded. Do not pool this study with the Platinum record-level study or attribute their differences to one changed factor.
Sources: uploaded experiment.yaml and manifest.json, run ed73f19; earlier audited E5 run 6a4fe91.
Dataset: https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/README.md

## Slide 2: Two separately reported studies

In the main Platinum study, Jev's weighted Top-1 score increased from 0.2350 for Random to 0.2694. Top-5 was nearly tied, at 0.3850 and 0.3882. These are normalized throughput scores. The separate E5 ranking study provides the other required metrics: Pairwise Accuracy was 0.5195 for Jev versus 0.4920 for Random, and mean runtime regret was 5.0810 versus 6.0589. Those are separate results with different hardware and candidate policies, so we do not merge the aggregates.

### Presenter reference

Exact main-study record-weighted scores:
Random: Top-1 0.235045432987; Top-5 0.385020869616.
Jev: Top-1 0.269355564550; Top-5 0.388244862250.
Weighting is by test-record count. Separately reported macro values are not used in the main charts. Top-1 is optimal test-pool runtime / selected runtime; Top-5 averages prefix-best throughput ratios for the first five choices. Runtime-based recomputation agrees with the uploaded float32-label summary within 5.3e-10; private original throughput labels were not uploaded.
Earlier E5 run: Random Top-1 0.3324399162, Top-5 0.4987746998, Pairwise Accuracy 0.4919591837, runtime regret 6.0588537976. Jev Top-1 0.3368666101, Top-5 0.5128292812, Pairwise Accuracy 0.5194557823, runtime regret 5.0810106152. The E5 mean regret is sensitive to a few large gains; its paired interval against observed Random includes zero. The E5 pairwise signal is modest and pertains to its full predicted ranking. We do not invent global Pairwise Accuracy for the tournament's eliminated tail.
Sources: uploaded official-weighted-summary.csv and earlier reviewed docs/results/tenset/w120-c50-s42-long-context/metrics.csv. Main prediction checksum: 53e2662aeb7480af284abcd2b302264d3760288060c512803ed67c25ca978179.

## Slide 3: Interpretation, cost, and remaining work

The paired analysis supports a modest Top-1 signal on this cohort: the exploratory 95% interval is approximately plus 0.009 to plus 0.060. Jev had a better Top-1 score on 278 workloads. The Top-5 interval crosses zero, so its improvement remains uncertain. Logged input usage corresponds to about 7 dollars and 62 cents, including the pilot. We have prepared workload-type and schedule-case exports and a controlled candidate-size sweep. Those follow-up results and fresh-call repeatability are still pending.

### Presenter reference

The comparison is Jev versus Random only. Bootstrap: resample all 484 workload pairs together, retain each test-record weight, recalculate weighted means as a ratio of sums; 50,000 percentile resamples, seed 20261116. These intervals are exploratory, unadjusted and conditional on the saved responses, workload sample and Random seed. No family-cluster sensitivity could be computed from the provided artifacts. Fresh API responses have not been independently replicated; cache replay is not a live repeatability test.
An additional exact uniform-Random expectation check removes the particular Random seed's selection noise: expected Top-1 0.2394043892 versus Jev 0.2693555650; delta +0.029951176, exploratory 95% interval [+0.012579007,+0.048033600]. Expected Top-5 0.3783793707 versus Jev 0.3882448624; delta +0.009865492, interval [-0.005704909,+0.025933966]. This analytical reference does not replace the observed Random-0 result.
Main-study optimum hits: Jev 9/484 and Random 6/484; throughput scores are not hit-rate accuracy. Failure-sentinel selections are Jev 8 and Random 7, so physical mean runtime/regret is not claimed for this tournament. Of 468 two-round workloads, 442 lost the test-pool optimum in round 0; this identifies a saved-path failure mode, not proof that a single-question alternative would succeed.
New successful-call input tokens: 179,522,529; previously cached pilot inputs: 1,880,950; all stored inputs 181,403,479. New input estimate 7.539946 USD; total including the pilot 7.618946 USD. Do not add those two totals. Assumed rate is 0.042 USD/million input tokens, output free; no invoice supplied. Decision latency averaged 4.102 seconds per workload including cached cases; live API latency averaged 1.022 seconds per request.
Remaining analysis must use real prepared operation metadata and schedule configurations. The existing 6–88 versus 400-candidate breakdown is descriptive and confounded by workload: 16 small-pool workloads versus 468 with 400 candidates. A same-workload nested-size sweep is prepared but has no new paid results. Optional field/context/order controls and fresh repeats would strengthen evidence; an all-raw-schedule run is not necessary for the stated Jev-versus-Random scope.
Audit limits: raw measurements, private throughput labels, and actual API request payloads were not supplied. Saved-record consistency, matched candidate pools, scores, and usage were checked.
