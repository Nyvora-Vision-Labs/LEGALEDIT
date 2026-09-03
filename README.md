# LegalEdit: a dissociation diagnostic for legal meaning preservation metrics

A simplified legal clause either still says what the original said or it does
not, and the automatic metrics used to check that cannot currently tell the
difference. This repository holds the diagnostic that shows why, the validation
protocol built around it, the full experimental pipeline, and the paper.

The protocol is instantiated on
[FrJUDGE / JUDGEBERT (Beauchemin et al., EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.5/),
which is, to our knowledge, still the only annotated corpus for legal meaning
preservation in any language.

## The argument

Two automated checks are in current use for metrics of this kind: a sentence
paired with itself must score highest, and a sentence paired with an unrelated
one must score lowest. Both are necessary and both are easy to state. But moving
from an identical pair to an unrelated pair changes lexical overlap and legal
meaning together, in the same direction, at the same time. Any monotone function
of token overlap therefore satisfies both by construction, which is exactly why
the embedding metrics in the literature already pass them. The checks test
calibration at two endpoints; they do not test what the metric is reading.

The remedy is a **dissociation**: an item on which the two confounded variables
are pulled apart, so that legal force changes while surface form does not.
Replace *doit* (must) with *peut* (may) in a statutory clause and roughly ninety
percent of the tokens are untouched, the sentence stays fluent and statutory in
register, and an obligation has become a permission. A metric that reads legal
meaning must drop sharply; a metric that counts shared words cannot.

**LegalEdit** is 373 such perturbations (310 tier 1) generated from two Quebec
statutes, with mean token Jaccard 0.93 against their source and none below 0.75.
Each is scored through a harness that takes an arbitrary scoring function, so
untrained metrics, supervised regressors and prompted LLM judges are all
administered the same test. The statistic it is read with is the **margin
fraction**, the share of a metric's own identical-to-unrelated range that it
spends on a legally decisive edit.

Around the diagnostic sit three further requirements, because a diagnostic alone
does not discipline how corpus-correlation evidence is read:

1. **A measured ceiling.** Correlation with human judgment means nothing until
   you know what correlation a human achieves. Two bounds should be reported,
   the leave-one-annotator-out correlation and the Spearman-Brown reliability of
   the aggregate label, because they answer different questions.
2. **Identical supervision.** An unsupervised cosine and a regressor fitted to
   the label are not on the same scale. Fit a monotone map on the training split
   only, then compare.
3. **A trivial-feature control.** Simplifications get shorter, the dominant
   error category is omission, and a deduct-per-error rubric is therefore partly
   predictable from how many words disappeared. A surface-feature baseline
   belongs in the same table as the neural metrics.

## What we found

- **The field splits into two families.** Every similarity-based metric spends
  between 0.022 and 0.039 of its working range on an edit that reverses the law.
  BERTScore over CamemBERTv2 places a sentence saying the opposite of the
  original at 0.982, where the original itself sits at 1.000 and an unrelated
  statute at 0.193. Bidirectional NLI entailment spends 0.670 and ranks 93.3% of
  items correctly.
- **The existing check would have excluded the winner.** mDeBERTa assigns a mean
  entailment of 0.780 to a sentence paired with itself and clears the 99% bar the
  identical-pair convention requires on only 1.3% of items, against 100% for every
  BERTScore and embedding metric. The metric that best tracks legal meaning is
  the one metric convention would disqualify.
- **A trivial feature wins on the corpus.** A bare length difference reaches
  r = 0.641 on FrJUDGE, above every semantic metric tested, and a ridge over ten
  surface features reaches r = 0.62, at the measured human ceiling of r = 0.597.
  The length feature also has the lowest margin of anything we measure (0.011).
- **The two orderings come apart.** Over the 13 scorers carrying both a
  calibrated correlation and a margin, the rank correlation between them is
  rho = 0.24 with a 95% bootstrap interval of [-0.39, 0.75]. That is an absence
  of resolution rather than a demonstrated null, but the extremes are not in
  doubt: selecting on corpus correlation picks the length feature and the
  prompted judge, and the entailment family that moves on the edit sits
  mid-table.
- **A prompted judge passes both conventional checks perfectly and still fails
  the dissociation.** It scores identical pairs 10.00 and unrelated pairs 1.06,
  then awards a clause whose legal force has been reversed 6.69 out of ten, a
  margin of 0.370.
- **The rubric, not the model, is what limits the judge.** Holding the model
  fixed and varying only the wording, margin runs from 0.421 under the
  rubric-faithful prompt to 0.687 under a bare instruction, a range of 0.267
  against 0.002 for an independent redraw of the same prompt. Deducting one point
  per identified error caps the penalty for a single clause reversal near the top
  of the scale.
- **One judge behaviour survives every model and prompt.** Weakening an
  obligation (*doit* to *peut*) scores 4.72 while strengthening one scores 8.02,
  a gap of 3.30 that ranges only over 3.24 to 4.03 across all five
  configurations, though both edits reverse the law by the same construction.
  The judge is tracking legal severity rather than legal equivalence.
- **Negative results we report rather than bury.** A reasoning model under the
  identical rubric does not help (r = 0.580, margin 0.320). Of six rules for
  combining the judge with bidirectional NLI, none beats both parents on both
  axes. Across seven training objectives, no target moves correlation by more
  than the scalar baseline's own spread across seeds, and training on the sanity
  checks buys satisfaction of them in distribution and nothing across a change of
  source document.

## Layout

```
data/       FrJUDGE.jsonl, the raw Prodigy export, and the two reference statutes
src/        the whole pipeline: loaders, metric zoo, augmentation, training, experiments
results/    JSON results, per-metric score caches, LLM response cache, run logs
paper/      LaTeX sources, generated tables and figures, main.pdf, supplementary.pdf
docs/       errata and re-analysis for the upstream corpus, and the citation audit
```

## The paper

`paper/main.pdf` is the 8-page paper, including limitations, ethics and the
17-entry bibliography. `paper/supplementary.pdf` is the 8-page supplement,
holding twelve numbered floats (Tables S1 to S10 and Figures S1 to S2) and the
analyses the paper points at: the full calibrated baseline table, the per-metric
margin plot, one example per perturbation rule, the agreement and ceiling
tables, the disagreement figure, the per-configuration judge table, the per-rule
margin breakdown, the seven training variants, the Williams tests, the
document-grouped comparison, the length-change strata, and the reproducibility
statement.

Build either with the standard four-pass sequence (the bibliography is numbered,
so BibTeX has to run):

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
pdflatex supplementary.tex && bibtex supplementary && pdflatex supplementary.tex && pdflatex supplementary.tex
```

Tables and figures are generated, not hand-written: `src/make_tables.py` writes
`paper/tables/*.tex` and `src/make_figs.py` writes `paper/figs/*.pdf`, both from
the JSON in `results/`, so no number in the paper is typed by hand. The macros
in `paper/numbers.tex` carry every figure quoted in prose for the same reason.

Every one of the 17 references was verified against arXiv, DBLP, CrossRef or the
ACL Anthology: identifiers fetched by ID, author lists diffed position by
position, venues confirmed at the venue. `docs/RELATED_WORK_AUDIT.md` records the
method and the four passes, including the errors earlier passes found and fixed.

`paper/archive/` holds the superseded 23-page build, kept only for reference.

## Reproducing

```bash
cd src
python3 exp1_ceiling.py         # agreement, human ceiling, structure of disagreement
python3 exp2_baselines.py       # metric zoo and surface baselines, raw and calibrated
./run_all_exp3.sh               # both training grids: grouped split first, then random
python3 exp4_challenge.py       # the LegalEdit diagnostic over every scorer
python3 exp5_llm_judge.py       # prompted judge: panel position, self-consistency
python3 exp5b_replication.py    # independent redraw of the judge, run 1 against run 2
python3 exp6_hybrid.py          # judge x NLI combination rules (cached, no API cost)
python3 exp7_deepseek_panel.py  # reasoning model and three prompt wordings
python3 make_tables.py && python3 make_figs.py
```

`exp3_train.py` is the long pole. Use `run_all_exp3.sh`, which drives both grids
one process per (config, seed) and skips anything already banked in the results
JSON, so it is safe to interrupt and relaunch. It runs the document-grouped grid
(three configurations, five seeds, writing `results/exp3_train_grouped.json`),
then the random-split grid (seven configurations, five seeds, writing
`results/exp3_train.json`). Either stage runs on its own; for the grouped one:

```bash
CONFIGS='JudgeBERT (repro)|JudgeBERT-DA (repro)|JudgeBERT-DA+LegalEdit' \
GROUPED=1 TAG=_grouped ./run_exp3_grid.sh
```

Budget roughly 10 minutes per (config, seed) for the un-augmented heads and
three to four times that for the `-DA` variants. Grouped runs are slower because
the largest source form is 37% of the corpus, larger than the 30% test quota, so
it always lands in training and its long clauses push most batches into the
upper padding buckets.

`exp2_baselines.py` and `exp4_challenge.py` honour `METRICS_DEVICE`, so the
metric zoo can be pinned to the CPU and recomputed alongside a training run
without contending for the same MPS queue. Both cache per-metric scores; delete
`results/cache/`, `results/raw_metric_scores.pkl` and `results/exp4_raw.pkl` to
force a full recomputation.

### The prompted judge

`exp2`, `exp4` and `exp5` include the judge when a DeepSeek key is available and
skip it with a printed notice otherwise, so the rest of the pipeline runs
unchanged without one. Put the key in a `.env` at the repository root
(`DEEPSEEK_API_KEY=sk-...`, git-ignored) or export it; see `.env.example`.
`llm_judge.py` uses only the standard library for transport, so the judge adds no
dependency.

Every response is cached under `results/llm_cache/*.jsonl`, keyed by a hash of
model, prompt version, sample index and pair, so runs are resumable and a rerun
costs nothing. `SKIP_JUDGE=1` disables the judge; `JUDGE_K`, `JUDGE_TEMP`,
`JUDGE_MODEL` and `JUDGE_WORKERS` override the defaults (K=5, temperature 1.0,
`deepseek-chat`, 12 threads). Adding a wording means adding an entry to
`PROMPTS`, which changes the hash rather than silently mixing responses drawn
under two different rubrics.

`JUDGE_BUDGET_USD` (default 5) is a hard ceiling on a run. DeepSeek reports token
counts rather than dollars, so cost is derived from the `PRICES` table and the
guard is deliberately conservative: once the running estimate crosses the cap no
further request is issued and the run stops with everything already cached.
`results/llm_cache/spend_ledger.json` holds the running total. The reasoner is
the expensive model, billing its chain of thought as output, which is why `exp7`
runs it on a fixed 100-edit subsample. The main judge run reported in the paper
is 7,080 calls at an estimated \$0.99 (63 completion tokens per call); the
independent redraw is another 7,080 at \$0.87, and the spend ledger's running
total including the reasoner and the prompt variations stands at \$1.32. Replayed
from the cache, all of it is free.

### Apple Silicon notes

Three specifics cost us a day and are worth knowing:

- **Pad to fixed buckets, not to the longest member of each batch.** Pair lengths
  span 30 to 417 tokens, so per-batch padding produced a distinct tensor shape
  almost every step. MPS compiles a kernel per shape, and the growing graph cache
  slowed epochs from about 10 s to about 150 s within a single run.
  `model.Batcher` now pads to one of seven fixed lengths, masked out, so no
  result changes.
- **Release the MPS cache each epoch.** Without `torch.mps.empty_cache()` the
  process grew to roughly 70% of system memory over 20 epochs and later runs died
  with `MPS backend out of memory`.
- **`taskpolicy -B` the training process.** macOS puts processes spawned from a
  detached background shell into the background QoS class, which throttles MPS
  work about fourfold.

MPS kernels are not bitwise deterministic, so a fixed seed reproduces a run only
approximately. Report spreads over seeds, not single runs.

## What each module does

| File | Purpose |
| --- | --- |
| `data.py` | Rebuilds FrJUDGE from the raw export so annotator identity is available, and pseudonymises the five annotators as A to E on load. |
| `splits.py` | The 60/10/30 splits (seeds 42 to 51), plus document-grouped splits that prevent source-form leakage. |
| `features.py` | Ten trivial surface features: lengths, token and numeral overlap, modal, negation and connective deltas. |
| `metrics_zoo.py` | French-capable baselines: BERTScore over CamemBERTv2 and FlauBERT (F1 and recall), sentence-embedding cosine (LaBSE, Sentence-CamemBERT, mpnet, mE5), and bidirectional mDeBERTa-XNLI entailment. |
| `augment.py` | Identical and unrelated sanity pairs, and the LegalEdit perturbation set. Perturbations are screened by a French well-formedness check (elision, contraction, determiner-noun gender) so a metric cannot detect the edit by spotting disfluency; the check flags 0 of 400 unmodified source sentences. |
| `model.py`, `train.py` | CamemBERTv2 regressor with scalar, distributional (KL over the rating histogram), annotator-aware (five heads) and quantile (pinball) objectives, plus an auxiliary characterization head. |
| `stats_utils.py` | Bootstrap CIs, paired bootstrap tests, the Williams test, ordinal and nominal Krippendorff alpha, power analysis. |
| `llm_judge.py` | Prompted LLM-as-judge, exposed with the metric-zoo signature so it drops into `exp2` and `exp4` unchanged. One request per pair so the judge never sees the probe condition, K samples per pair for self-consistency, every response cached to an append-only JSONL. Carries the three prompt wordings (v1 French rubric, v2 English rubric, v3 bare instruction) and the spend ceiling. |
| `exp5b_replication.py` | Draws a fresh K samples under a separate cache tag and compares 14 headline statistics against tolerances fixed before the draw. Replaying the cache would reproduce them by construction, so it does not. |
| `exp6_hybrid.py` | Six rules for combining the judge with bidirectional NLI, calibration fitted on the training split alone, the diagnostic never entering the fit. Reads cached scores and costs nothing. |
| `exp7_deepseek_panel.py` | The two controls the replication cannot supply: a reasoning model under the identical rubric, and three prompt wordings on the same model. |
| `make_tables.py`, `make_figs.py` | Regenerate every table and figure in the paper from the result JSON. |
| `pseudonymise_frjudge.py` | Standalone utility that rewrites the raw export's real annotator names to stable pseudonyms and coarsens timestamps to the day, for anyone redistributing a reconstruction. |

## Key numbers

`python3 src/exp1_ceiling.py` prints, among others:

- Krippendorff alpha for legal meaning: **0.104 nominal**, the published figure,
  against **0.325 ordinal**, the coefficient appropriate to a ten-point ordered
  scale.
- Mean pairwise percent agreement **25.96%**, reproducing the published value
  exactly, which is what confirms the reconstruction.
- Leave-one-annotator-out human ceiling **r = 0.597** (RMSE 3.27), best-agreeing
  pair **r = 0.606**, mean pairwise **r = 0.462**, Spearman-Brown reliability of
  the five-rater mean **0.811**.
- The five annotators form two stable populations: B, C and D average 7.61 while
  A and E average 4.34, a gap of over three points on a ten-point scale.

`python3 src/exp4_challenge.py` runs the diagnostic and prints the margin table
that Table 1 of the paper is generated from.

`python3 src/exp5_llm_judge.py` and `exp7_deepseek_panel.py` produce the judge
results, including its calibrated r = 0.755 (an independent redraw moved it to
0.726, so it is quoted as a range), its ordinal alpha of 0.689 across its own
five samples against 0.325 for the five humans, and the prompt-sensitivity
spread.

## Data and ethics

`data/annotated.jsonl` is a reconstruction of the upstream raw export with the
annotators **already pseudonymised as A to E**, produced by
`src/pseudonymise_frjudge.py`, which also coarsens the per-annotation timestamps
to the day. The upstream export carries the annotators' real first names, and
`docs/ERRATA.md` item 11 reports that to its authors. Nothing here, in the code,
the results or the paper, carries a name: `data.py` derives the A to E labels
from the tokens in the file rather than from a stored mapping, so pointing it at
the upstream raw export reproduces exactly the same labels and the same numbers.
Anyone redistributing a copy of the raw export should run the script over it
first.

Note that git history still contains the pre-pseudonymisation copy of this file.
Removing it from history requires a rewrite and a force-push, which has not been
done.

The statutes LegalEdit is generated from are public legal texts. The perturbed
sentences are deliberate misstatements of Quebec law, released as an evaluation
artefact only and labelled as such in the distribution. They should not be quoted
as statements of the law, and nothing here constitutes legal advice. A metric
that cannot distinguish *doit* from *peut* must not be used as an automated gate
on whether a simplified clause is safe to publish; the appropriate deployment is
triage, the ranking of clauses for human review, rather than certification.

## Documents in `docs/`

| File | Contents |
| --- | --- |
| `ERRATA.md` | Errata and discrepancies found in the upstream JUDGEBERT paper and release, offered as corrigenda and shared with the authors. Items 1 to 3 affect a reported number. |
| `REANALYSIS.md` | Direct comparisons against the numbers reported upstream, held here rather than in the paper because they are corrections to another group's results. |
| `README_corrected.md` | A drop-in replacement for the upstream repository README, which documents the schema of a different dataset than the one released. |
| `RELATED_WORK_AUDIT.md` | The citation audit: how every reference was verified, what four passes found, and the record of what was corrected. |

## Requirements

`torch`, `transformers`, `sentence-transformers`, `scikit-learn`, `scipy`,
`statsmodels`, `krippendorff`, `matplotlib`. The judge needs no extra package.
Runs on Apple MPS or CPU with no GPU cluster: the largest model is
CamemBERTv2-base at 112M parameters, and the whole pipeline runs end to end on a
laptop.
