# TenSet contribution: four ResNet-50 slides

- [Editable PowerPoint](tenset_rq1_slides.pptx)
- [PDF](tenset_rq1_slides.pdf)
- [Speaker notes and source details](speaker_notes.md)
- [Artifact provenance and checksums](provenance.json)

The primary contribution now covers the completed reconstructed ResNet-50
network evaluation, in this order:

1. Setup: 27 distinct tasks, 56 task occurrences, and 100,126 schedule records.
2. Workflow: serialization, hidden runtime labels, Jev's choice tournament,
   saved rankings, and occurrence-weighted network endpoint scoring.
3. Results: Jev versus the saved Random baseline, with MLP + Ranking loss
   shown separately as an external published reference.
4. Diagnosis: probability ties, finalist selection, the offline Random-choice
   reference distribution, and possible follow-up experiments.

Jev's network Top-1/Top-5 scores are 0.097724/0.274583; Random's are
0.091462/0.377121. These are evaluator latency-proxy ratios, not accuracy
percentages or newly measured end-to-end network speedups. The paper's
0.8823/0.9456 MLP + Ranking loss values are external aggregates; the original
pool and checkpoint have not been matched and the model was not run locally.
One saved Jev outcome does not establish fresh-call repeatability. The fact
that 95.3% of 100,000 offline Random draws exceed Jev's fixed Top-5 is a
Random-choice reference result, not a p-value or a confidence interval.

The PowerPoint contains native editable rectangles, text boxes and line shapes
for its bar charts. It does not use chart screenshots or data-linked Google
Sheets objects. It can be uploaded for import into Google Slides; account
authorization was blocked during preparation, so no account-level Google Slides
file was created and no cross-application import test was performed. The PDF
has four pages. Presenter notes retain the numerical definitions, evidence
limits, and source details omitted from the concise visible slides.

Sources: [reviewed ResNet-50 results](../../results/tenset/paper-resnet50/README.md),
[RQ1 report](../../rq1-report.md), and the
[network protocol audit](../../tenset-paper-protocol.md).
The deck and PDF are exact copies of the reviewed supplied four-slide artifacts;
SHA-256 hashes are recorded in provenance.json.

The earlier studies remain separately documented: the
[484-workload Platinum tutorial tournament](../../results/tenset/tutorial-500-platinum-test/README.md),
[E5 16/32/50 candidate-size study](../../results/tenset/size-sweep/README.md), and
[E5 50-candidate audit](../../results/tenset/w120-c50-s42-long-context/metrics.csv).
Their cohorts, candidate policies, and score aggregation differ from this
network endpoint evaluation. Their scores are not substituted into this deck's
ResNet-50 comparison.
