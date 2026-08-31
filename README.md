# LegalEdit: a dissociation diagnostic for legal meaning preservation metrics

Code, data pipeline, and paper for **LegalEdit**, a challenge set that holds
surface form fixed while reversing the legal force of a statutory clause, and
for the four-part validation protocol built around it. The protocol is
instantiated on
[FrJUDGE / JUDGEBERT (Beauchemin et al., EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.5/),
the only annotated corpus and metric for this construct.

## Layout

```
work/
  data/          FrJUDGE.jsonl + raw Prodigy export + reference statutes
  src/           all experiment code
  results/       JSON results, per-metric score cache, run logs
  paper/         LaTeX source, generated tables/figures, compiled main.pdf
  artifacts/     corrected README, errata, re-analysis notes, pseudonymisation script
```

## Reproducing

```bash
cd src
python3 exp1_ceiling.py       # agreement, human ceiling, disagreement structure
python3 exp2_baselines.py     # metric zoo + surface baselines, raw and calibrated
./run_all_exp3.sh             # both training grids: Table 9 first, then Table 8
python3 exp4_challenge.py     # LegalEdit diagnostic over the baseline metrics
python3 make_tables.py && python3 make_figs.py
cd ../paper && tectonic -X compile main.tex --outdir .
```

`exp3_train.py` is the long pole. Use `run_all_exp3.sh`, which drives both
grids one process per (config, seed) and skips anything already banked in the
results JSON, so it is safe to interrupt and re-launch. It runs two stages:

* **Table 9** (document-grouped split), three configurations against five
  seeds, writing `results/exp3_train_grouped.json`. Only the reproduction and
  the two augmented mixtures are rerun here; the other four heads differ in
  training target, not in what they can exploit about a source form.
* **Table 8** (random split), all seven configurations against five seeds,
  writing `results/exp3_train.json`.

Either stage can be run on its own. For Table 9 only:

```bash
CONFIGS='JudgeBERT (repro)|JudgeBERT-DA (repro)|JudgeBERT-DA+LegalEdit' \
GROUPED=1 TAG=_grouped ./run_exp3_grid.sh
```

Grouped runs are slower per seed than random-split ones: the largest source
form is 37% of the corpus, larger than the 30% test quota, so it always lands
in training, and its clauses are long enough to push most batches into the
upper padding buckets. Budget roughly 10 min per (config, seed) for
the un-augmented heads and 3-4x that for the `-DA` variants, whose training set
is three to four times larger.

Three Apple-Silicon specifics cost us a day and are worth knowing:

* **Pad to fixed buckets, not to the longest member of each batch.** Pair
  lengths span 30-417 tokens, so per-batch padding produced a distinct tensor
  shape almost every step. MPS compiles a kernel per shape, and the growing
  graph cache slowed epochs from ~10 s to ~150 s *within a single run*, turning
  a 10-minute run into an 80-minute one. `model.Batcher` now pads to one of
  seven fixed lengths. Padding is masked out, so no result changes.
* **Release the MPS cache each epoch.** Without `torch.mps.empty_cache()` the
  process grew to ~70 % of system memory over 20 epochs and later runs died
  with `MPS backend out of memory`.
* **`taskpolicy -B` the training process.** macOS puts processes spawned from a
  detached background shell into the background QoS class, which throttles MPS
  work about 4x.

MPS kernels are not bitwise deterministic, so a fixed seed reproduces a run only
approximately; report spreads over seeds, not single runs.

`exp2_baselines.py` and `exp4_challenge.py` honour `METRICS_DEVICE`, so the
metric zoo can be pinned to the CPU (`METRICS_DEVICE=cpu python3 exp2_baselines.py`)
and re-computed alongside a training run without contending for the same MPS
queue. Both cache per-metric scores under `results/`; delete `results/cache/`,
`results/raw_metric_scores.pkl` and `results/exp4_raw.pkl` to force a full
re-computation.

Requires `torch`, `transformers`, `sentence-transformers`, `scikit-learn`,
`scipy`, `statsmodels`, `krippendorff`, `matplotlib`. Runs on Apple MPS or CPU;
no GPU cluster needed (the largest model is CamemBERTv2-base, 112M parameters).

## What each module does

| File | Purpose |
|---|---|
| `data.py` | Rebuilds FrJUDGE from the **raw** export so annotator identity is available; pseudonymises the five annotators as A–E on load. |
| `splits.py` | The paper's 60/10/30 splits (seeds 42–51), plus document-grouped splits that prevent source-form leakage. |
| `features.py` | Ten trivial surface features (lengths, token/numeral overlap, modal/negation/connective deltas). |
| `metrics_zoo.py` | French-capable baselines: BERTScore over CamemBERTv2 and FlauBERT (F1 and recall), sentence-embedding cosine (LaBSE, Sentence-CamemBERT, mpnet, mE5), and bidirectional mDeBERTa-XNLI entailment. |
| `augment.py` | Identical / unrelated sanity pairs, and the **LegalEdit** perturbation set. Perturbations are screened by a French well-formedness check (elision, contraction, determiner-noun gender) so that a metric cannot detect the edit by spotting disfluency; the check flags 0/400 unmodified source sentences. |
| `model.py`, `train.py` | CamemBERTv2 regressor with scalar, distributional (KL over the rating histogram), annotator-aware (5 heads), and quantile (pinball) objectives, plus an auxiliary characterization head. |
| `stats_utils.py` | Bootstrap CIs, paired bootstrap tests, Williams test, ordinal/nominal Krippendorff α, power analysis. |

## Key numbers

Run `python3 src/exp1_ceiling.py`; it prints, among others:

- Krippendorff α for legal meaning: **0.104 nominal** (the published figure)
  vs **0.325 ordinal** (the appropriate coefficient).
- Mean pairwise percent agreement **25.96%**, reproducing the published value
  exactly, confirming the reconstruction.
- Leave-one-annotator-out human ceiling: **r = 0.597**, RMSE 3.27.
  Best-agreeing annotator pair: **r = 0.606**.

Run `python3 src/exp4_challenge.py` for the LegalEdit diagnostic (373 edits, 310
tier-1, mean token Jaccard 0.93). The headline: every similarity-based metric
spends 0.022-0.039 of its identical-to-unrelated range on an edit that reverses
the law, while bidirectional NLI spends **0.670** and ranks 93% of items
correctly. Corpus correlation does not predict that margin.

## Note on the data

`data/annotated.jsonl` is a copy of the upstream raw export and **contains the
annotators' real first names**. `data.py` pseudonymises on load and nothing in
`results/` or `paper/` contains a name. If this pipeline is redistributed, run
`artifacts/pseudonymise_frjudge.py` over the data directory first. See
`artifacts/ERRATA.md`, item 11.
