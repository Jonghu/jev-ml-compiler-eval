# TenSet ResNet-50 contribution: four slides

Order: setup → workflow → results → diagnosis. This is the primary four-slide ResNet-50 contribution. The earlier 484-workload Platinum and E5 candidate-size studies remain available separately; their scores are not pooled with this network evaluation.

The PPTX uses native editable drawing bars, text and line shapes. Google account authorization was blocked, so no account-level Google Slides file was created; no cross-application import test was performed.

Primary evidence: [reviewed ResNet-50 results](../../results/tenset/paper-resnet50/README.md). Historical evidence: [Platinum tutorial study](../../results/tenset/tutorial-500-platinum-test/README.md), [E5 candidate-size study](../../results/tenset/size-sweep/README.md), and [E5 50-candidate study](../../results/tenset/w120-c50-s42-long-context/metrics.csv).

## Slide 1

Suggested narration:
We now evaluate one reconstructed ResNet-50 network rather than the earlier sampled-workload studies. It contains 27 distinct tensor tasks, occurring 56 times, and 100,126 measured schedule records. Jev and Random use the same local candidate pools. Jev is zero-shot, using field descriptions, long context and a tournament. Random is our paired baseline. The paper's trained MLP is a separate external reference.

Protocol and score definition:
ResNet-50, batch 1, input (1,3,224,224), declared Platinum-8272 CPU measurements. Twenty-five task pools have 4,000 records, one has 20, and one has 106. This run uses all available records in the declared task inventory: no within-task train/test split, error filtering or deduplication. It is not the 500-file tutorial test split. Declared source provenance is checked against supplied exports; original prepared task pickle/source records were not uploaded for an independent source inventory check.
Jev 1.14.0 uses structured program and schedule context, field descriptions, opaque candidate aliases, long context and a max-255-choice, top-5-promoting tournament. Candidate records/runtimes/input hashes agree across Jev and Random in supplied exports. Saved Random-0 rankings replay exactly. Jev inference is served by the API; the Mac runs the client, not the benchmarked CPU kernels. No task-specific Jev training is performed.
For task t let w_t be its occurrence count, m_t the oracle latency proxy and ell_t(c) the evaluator's latency proxy for schedule c. Score@K = sum_t(w_t*m_t) / sum_t(w_t*min_{c in first K choices} ell_t(c)). Weights sum to 56. The evaluator reconstructs normalized float32-origin labels and floors throughput at 1e-5, then accumulates in float64. This is a weighted network best-of-K ENDPOINT score; Top-5 is not the prefix-average Top-5 used in the earlier Platinum/E5 studies. It is neither an accuracy percentage nor a newly measured end-to-end ResNet speedup.
The published comparator is MLP + Ranking loss: a trained neural-network cost model; ranking loss learns to order schedules by performance. Its published result is not a locally evaluated paired baseline.

Sources:
resnet50-audit/audit-summary.json; numerical/numerical-audit.json; protocol/protocol-audit.json. Pinned evaluator: https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/scripts/eval_model_on_dataset.py

## Slide 2

Suggested narration:
We begin with all available records for the declared ResNet-50 task inventory.
The serializer provides program and tensor features plus schedule transformation
steps, field descriptions and opaque shuffled candidate IDs. Measured runtimes
bypass Jev and are used only by the scorer. Each candidate group becomes a Jev
question. Groups have at most 255 options and may be smaller because of token
limits. The top five per group advance, repeatedly, until one final question.
We then save Top-1 and Top-5 choices and score their network-level latency proxies.

Dataset and protocol:
27 distinct ResNet-50 tasks, 56 task occurrences, 100,126 schedule records from
the declared Platinum-8272 CPU source. Twenty-five pools have 4,000 records, one
has 20 and one has 106. This run uses full available task records, not a 90/10
within-task train/test split and not the earlier 500-file tutorial test portion.
The Mac runs the evaluation/API client; kernels are not newly timed on the Mac.

Serialization:
Structured JSON includes workload metadata, tensor shapes/DAG/target and schedule
transformation steps. Field descriptions accompany these inputs. Candidate IDs
are opaque aliases with seeded shuffled presentation order. The measured runtime
labels are excluded from the Jev decision input by the implementation contract.
The dashed bypass illustrates this contract; actual raw request bodies and raw
source/prepared records were not supplied for independent wire-content/source
identity verification. Saved input hashes/pools/rankings and implementation
consistency were checked, which is a narrower claim.

Tournament and ranking:
Jev 1.14.0, long context; each question returns an explicit choice and probabilities
for its presented options. Max 255 options is a ceiling, not an invariant group
size: token guards may split smaller. Within each question, the explicit choice
comes first, then descending probability; exact ties retain seeded presented
order. The top five per group advance, and rounds repeat until a single final
question. Actual tasks used 23 two-round, two one-round and two three-round
tournaments (673 total questions). Probabilities from different questions are
not globally comparable. Final Top-1/Top-5 saved choices use this same choice-first
ranking rule; this is not pure global probability sorting. A global tournament
Pairwise Accuracy is not reported for the synthetic tail ranking.

Score and comparisons:
For task t, w_t is its frequency in the network (weights sum to 56), m_t is its
oracle latency proxy and ell_t(c) the evaluator latency proxy of candidate c.
Score@K = sum_t(w_t*m_t) / sum_t(w_t*min_{c in first K}ell_t(c)), K in {1,5}.
The evaluator reconstructs normalized float32-origin labels with throughput floor
1e-5 and float64 accumulation. This is a task-frequency weighted NETWORK endpoint
ratio, not a record-weighted score or mean prefix-best score. Higher is better.
These proxies are not newly measured end-to-end ResNet wall times or accuracy.
Random uses the same local candidate pools. MLP + Ranking loss values are external
published paper aggregates, not a locally trained paired model; exact paper pool
and checkpoint identity remain unverified. Provenance stays in presenter notes
in accordance with the team's concise visible slide style. No new Jev inference,
timing experiment or fresh API replication was performed to make this diagram.

Sources:
resnet50-audit/audit-summary.json; protocol/protocol-audit.json;
tournament/tournament-audit.json; numerical/numerical-audit.json.
Pinned evaluator: https://github.com/tlc-pack/tenset/blob/35774ed373f9693cc6efcbe83f5703141e8ab0c8/scripts/eval_model_on_dataset.py

## Slide 3

Suggested narration:
On our fixed local pools, Jev's network Top-1 score is 0.0977 versus Random's 0.0915, a small lead. Jev's Top-5 is lower: 0.2746 versus 0.3771. Separately, the paper reports MLP plus ranking loss at 0.8823 and 0.9456. The table is an external published reference, not a paired locally trained comparator.

Exact scores:
Jev Top-1 0.097724226968282; Top-5 0.274582561989127.
Random-0 Top-1 0.091461940469530; Top-5 0.377121152526418.
Jev minus saved Random: Top-1 +0.006262286498751463; Top-5 -0.10253859053729059.
Published MLP + Ranking loss: Top-1 0.8823; Top-5 0.9456, Section 5.2, Table 4, ResNet50 column, printed page 7. It is a trained neural-network schedule-cost model. These published scores are numerically higher, but exact original task/candidate inventory and trained checkpoint identity have not been independently matched. The published model was not trained or run locally for this experiment. Do not interpret the comparison as a paired statistical effect or claim an accuracy/speedup percentage.
One reconstructed network and one saved Jev response set are available. The small saved Top-1 lead does not establish reliable superiority. Global tournament Pairwise Accuracy is not meaningful because unranked eliminated candidates have a synthetic tail order; no global PA is reported here. Runtime-weighted latency terms are evaluator proxies, not newly executed full-network timing.
Original source records, prepared shards/task pickle and actual request bodies were not supplied. Numerical audits establish consistency of saved scores, candidate pools, rankings, manifest and implementation, not independent original source or transmitted payload identity. The original paper pool/checkpoint remain unverified. No fresh Jev response replication has been run.

Sources:
resnet50-audit/audit-summary.json endpoint_scores and published_reference; numerical/numerical-audit.json; raw paper-network-summary.json.
Paper: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/a684eceee76fc522773286a895bc8436-Paper-round1.pdf#page=7

Chart display update:
Both native editable charts show all three point estimates: Random and Jev are paired local measurements, while the gray MLP + Ranking loss bar is an external published aggregate. Its label is inside the bar to keep 0.8823/0.9456 visible near the axis maximum. Shared display axes do not establish matched original candidate pools, checkpoints or a paired statistical comparison with MLP. Provenance and limitations remain in presenter notes to match the team slide style.

Google Slides import edition:
The two result charts use native editable rectangles, text boxes and line shapes on a 0–1 scale. Scores, model names, tables and provenance are unchanged. This preserves drawing-object editability when importing the PPTX into Google Slides; these are not data-linked Google Sheets chart objects. No account-level Google Slides file was created and no cross-application import test was performed.

## Slide 4

Suggested narration:
The audit points to a selection bottleneck. Twenty-five of 27 final questions have a tie at the Top-5 boundary, while the fastest surviving finalist makes the reported Top-5 in only two tasks. Questions have a median of four distinct probability levels. In a separate offline Random-choice reference, 95.3 percent of 100,000 draws beat this fixed Jev Top-5 score. This is not a statistical significance test or a fresh Jev repeat. A useful next experiment is randomized tie-breaking and presentation order, followed by independent Jev calls.

Diagnostics, precisely:
Across 673 saved questions, median distinct probability values per question = 4. Final Top-5 boundary ties occur in 25/27 tasks. The runtime-best surviving finalist is included in the actual Top-5 in 2/27 tasks; it is tied at the cutoff but excluded in 14 tasks and below the cutoff probability in 11. The best-finalist claim is based on runtime equivalence, not only one arbitrarily chosen tied candidate ID. These 25 boundary ties should not be confused with the 14 exclusions of a best finalist by a boundary tie. All 25 multi-round tasks lose an original oracle schedule in round 0: 22 below the probability cutoff and 3 tied at it but excluded. This is descriptive evidence from saved decisions, not a causal presentation-order experiment.
The retrospective oracle-aware network ceiling within surviving final pools is 0.742401408413135 versus actual Jev Top-5 0.2745825619891273. The final-ranking stage accounts for 86.866% of the saved Top-5 excess weighted latency proxy. These are post hoc upper bounds/decompositions requiring runtime labels, not attainable predictions, model reruns or measured speedups. They are deliberately omitted from the headline slide to avoid conflating oracle information with achieved results.
Random reference: 100,000 offline uniform ordered samples without replacement per task, NumPy default_rng seed 20261118, fixed local pools/occurrence weights/fixed Jev output. Exactly 95,320 draws have a strictly higher Top-5 network score than saved Jev; no ties were observed. The Random central 95% percentile range is [0.26344452871103907,0.4405937504224168]. This is a Random-choice reference distribution, not a confidence interval, p-value, population effect estimate or API repeatability result. Saved Jev Top-1 exceeds 77.934% of such draws; this does not establish a stable advantage.
Usage: 673 stored questions, 631 live requests and 42 cached prior responses. New successful live input count is 95,175,703 tokens; at the assumed USD 0.042/million input tokens this is USD 3.997379526. Cached inputs are 5,585,849 historical tokens and are not new charges for this run. The slide says new INPUT estimate, not total invoice. Live outputs (1,061,953 tokens), hidden retries/failures, and invoice reconciliation are outside this estimate; no output charge is assumed in the displayed input-only estimate. Decision latency totals 1,133.87 seconds in the saved metadata.
Limitations: one fixed reconstructed network and one saved Jev outcome; no fresh response replication, random-order rerun or causal tie-breaking test. Raw source records/task pickle/prepared shards and wire request/response files were not uploaded. Saved numerical/protocol/accounting consistency passed; raw source and transmitted input identity remain unverified. Artificial failure proxies must not be interpreted as physical inference time.

Sources:
resnet50-audit/tournament/tournament-audit.json; tournament/finalist-selection-diagnostics.json; random-reference/random-reference.json; audit-summary.json; numerical/numerical-audit.json.
