# JUDGEBERT (arXiv:2508.16870v1 / EMNLP 2025 main, pp. 92–118) — Assessment and 10 Ideas to Improve

**Paper:** Beauchemin, Albert-Rochette, Khoury, Déziel — *JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences*, Université Laval.
**Artifacts:** <https://github.com/GRAAL-Research/JUDGEBERT> (FrJUDGE released as JSONL; **no model weights released**).
**Status checked on the web:** published at EMNLP 2025 (main conference), so the ideas below are framed as a *v2 / journal extension / follow-up paper* agenda rather than pre-submission fixes.

---

## Short assessment

### What the paper gets right

- **A genuinely new construct.** Separating "legal meaning preservation" (LMP) from generic "meaning preservation" is well motivated. The *automobile* vs. *véhicule* example (mechanical force vs. mechanical-or-human force) is a crisp demonstration that synonymy in general language is not synonymy in law.
- **A real, expensive resource.** 297 sentence pairs annotated by five law students on three dimensions, with a documented three-step protocol (characterization → bracket → error deductions), a Human Evaluation Datasheet, an 18-class legal characterization taxonomy grounded in Quebec insurance law, and a released annotation recipe. This is the first legal-meaning judgment dataset in any language, and the raw per-annotator ratings plus free-text justifications are actually published.
- **Honest framing.** The limitations section concedes small data, sentence-level decontextualization, no out-of-domain split, and the unfairness of running English metrics on French. The ethics section is substantive.
- **The right instinct on asymmetry.** Table 6 (percentage of predictions *above* the human label) is the paper's sharpest idea: in a legal setting, an over-permissive metric is much more dangerous than a strict one. That deserves to be a headline result, not a secondary table.

### Where it is weak

1. **The gold label is barely a signal.** Krippendorff's α = 0.10 for LMP, 25.96 % raw agreement, and two annotator clusters whose means are 7.5 vs. 4.35 — a 3-point gap on a 10-point scale. The paper acknowledges this and then averages the five ratings anyway, producing a target that no annotator actually endorsed.
2. **The headline correlation may exceed the human ceiling.** Recomputing from the released `FrJUDGE.jsonl` (1,485 rows = 297 pairs × 5 ratings): holding out one rating per pair and correlating it with the mean of the other four gives **r ≈ 0.60 (range 0.52–0.65 over the five held-out positions) and RMSE ≈ 3.27**. JUDGEBERT reports r = 0.74, RMSE = 1.72. A model that beats a human rater at predicting the *average of humans* is very likely exploiting the shrinkage of a bimodal, low-agreement mean — not doing better legal reasoning.
3. **The +DA numbers are not comparable to the no-DA numbers.** Adding 594 identical/unrelated pairs (labelled effectively 10 and 1) to *both* train and test inflates label variance at the easy extremes. Every metric's Pearson jumps (BERTScore 0.46 → 0.94, SBERT-Multi 0.06 → 0.90) while BERTScore's RMSE simultaneously gets *worse* (3.61 → 5.09) — the classic signature of variance inflation rather than improved judgment. JUDGEBERT-DA's r = 0.97 is therefore not evidence about legal meaning.
4. **The sanity checks are trained on.** JUDGEBERT scores 0.00/0.00 on both checks; JUDGEBERT-DA scores 100/100 *because the DA training set is exactly those two cases*. Passing a test you were fine-tuned on is not a sanity check. The abstract's "it always returns a score of 100 % … 0 % for two unrelated sentences" does not distinguish the two variants and overclaims.
5. **Baselines are uncalibrated and no trivial baseline is reported.** The comparison metrics are decimal-scaled into [0,10] but never *fitted* to the human scale, so their RMSE and the Table 6 over-prediction rates are largely an artifact of never having seen the label distribution — an unsupervised similarity score is being scored against a supervised regressor with no common calibration. Concretely, on the released corpus: raw word-count difference between original and simplification correlates **r = −0.62** with the gold label; a 6-feature linear model over lengths and token overlap reaches **r = 0.61 / RMSE = 1.90 under 10-fold CV** — beating every Transformer baseline in Table 5 and matching the human ceiling. Much of what JUDGEBERT (0.74 / 1.72) adds over that is currently unquantified.
6. **One generator, one domain, one jurisdiction.** Every simplification comes from `gpt-4-turbo-2024-04-09`, zero-shot, one prompt, `max_new_tokens=100`. A metric meta-evaluation needs outputs spanning several systems and quality levels; otherwise the model may be learning "what GPT-4-turbo gets wrong on Quebec property-insurance forms."
7. **Statistical reporting is thin.** No confidence intervals on Pearson, no significance test between metrics, no Spearman/Kendall (the label is ordinal and bimodal — Pearson is the wrong default), test split ≈ 89 items. Card et al. (2020), *With Little Power Comes Great Responsibility*, is in the bibliography but not applied.
8. **Rich annotations are collected and then thrown away.** Annotators label characterization (18 classes) and four error types (hallucination, omission, consistency, confusion), and these drive the score arithmetic — yet the model predicts only a scalar, the error-type labels are not in the release (`FrJUDGE.jsonl` has `original`, `simplification`, `simplicity`, `qualification`, `legal_meaning`, `justifications` only), and there is **no annotator ID field**, so annotator-aware modelling is impossible for downstream users.
9. **The artifact undercuts the paper.** The repository README is copy-pasted from a different project: it documents `label` as "binary … 0 means ungrammatical and 1 means grammatical", a `source` URL field and an "aggregated BDL category" — none of which exist in `FrJUDGE.jsonl`. The load snippet points at the wrong path. The license also disagrees with the paper: §3 states CC-BY 4.0, the README states CC-BY-NC-SA 4.0. And a metric paper that ships no weights and no `pip`-installable scorer will not get adopted (contrast MeaningBERT, which did).
10. **Small factual slips.** Table 7 lists "Top K 0.9" (0.9 is a `top_p` value; OpenAI's API exposes no `top_k`). §3.2.2 cites Kandel & Moles (1958) as the "French Flesch-Kincaid grade level" with a threshold of ≤ 50 and a discarded example scoring 69.87 — that is French Flesch **Reading Ease** (0–100, higher = easier), not a grade level. §6.2 says "only three metrics" and then names four. Table 5's fourth column is headed `%>1%` but the text defines it as the ratio scoring *at or below* 1 %. SBERT-Multi's identical-pair score changes between the DA-False and DA-True blocks (0.00 → 100.00) even though it is an untrained metric whose behaviour on identical pairs cannot depend on how JUDGEBERT was trained — one of the two is wrong.

**Bottom line.** The construct and the corpus are the contribution and they are worth building on. The metric's evaluation, as it stands, does not yet establish that JUDGEBERT measures legal meaning rather than length-preservation plus label shrinkage. Ideas 1–4 below are the ones that would change the paper's conclusions; 5–10 broaden and harden it.

---

## 10 ideas to improve the paper

### 1. Report the human ceiling, and evaluate against individual annotators, not just their mean

**Problem.** With α = 0.10 there is no established upper bound, so r = 0.74 is uninterpretable — it may already be *above* what a human achieves.

**Evidence.** Recovered annotator IDs from `datastore/annotated/insurance_text_simplifications_annotated.jsonl` (field `_annotator_id`) and computed the true leave-one-annotator-out ceiling on all 297 pairs:

| Held-out annotator | r vs. mean of other 4 | RMSE |
|---|---|---|
| Clara | 0.524 | 3.93 |
| Eve | 0.569 | 3.05 |
| Simon | 0.680 | 2.75 |
| William | 0.601 | 3.08 |
| Yanis | 0.612 | 3.56 |
| **Mean ceiling** | **0.597** | **3.27** |

Pairwise inter-annotator Pearson ranges **0.34–0.61** (best pair: Simon–William 0.61). JUDGEBERT reports **0.74 / 1.72** — above the leave-one-out ceiling *and* above the strongest human–human correlation in the study. Annotator means split cleanly into a lenient cluster (Eve 7.57, Simon 7.55, William 7.72) and a strict one (Clara 4.33, Yanis 4.36).

**Do this.**
- Add a "human ceiling" row to Table 5: leave-one-annotator-out correlation and RMSE against the consensus of the rest, with bootstrap CIs.
- Report per-annotator correlations, `r(JUDGEBERT, annotator_i)` for all five, alongside `r(annotator_i, annotator_j)`. If the model correlates better with the mean than any human does, say so explicitly and interpret it as regression-to-the-mean rather than superiority.
- Ship annotator IDs in the *public* release (`FrJUDGE.jsonl` drops them) so others can reproduce this — pseudonymised, see idea 10.

**Effort:** low (analysis only). **Impact:** very high — it reframes the central claim honestly and pre-empts the most obvious reviewer objection.

---

### 2. Stop averaging a bimodal label; model the disagreement instead

**Problem.** Annotators A/B/C average ≈ 7.5 and D/E average ≈ 4.35. The gold label distribution on the released data is strongly bimodal (302 ratings of 1 and 452 of 10 out of 1,485), and the **median spread across the five annotators on a single pair is 7 points; 188 of 297 pairs (63 %) have a range ≥ 6**. Their mean is a number that describes no one's legal opinion.

**Do this.**
- Predict the full 5-rating distribution, not the mean: soft-label / distributional regression trained with KL or Earth Mover's distance, evaluated with cross-entropy and the *Distance to Human Distribution* used in the annotator-disagreement literature.
- Report a **calibrated uncertainty band**, e.g. "LMP ≈ 6.2, annotator range 2–9". For a legal deployment, "the experts disagreed sharply about this clause" is more actionable than a point estimate and is precisely the flag a practitioner needs.
- Alternatively, model annotator identity explicitly (per-annotator bias/severity term, à la annotator-embedding or Dawid–Skene-style aggregation) so the "strict" cluster and the "lenient" cluster are represented rather than averaged out.
- Investigate *why* D and E are strict: the paper speculates about domain expertise inferring case law. A short qualitative analysis of their free-text `justifications` (all 1,485 rows have one, and they are currently unused) would resolve this and is essentially free.

**Effort:** medium. **Impact:** very high — turns the paper's biggest weakness into its most interesting scientific finding.

---

### 3. Fix the data-augmentation confound: keep sanity checks out of training *and* out of the correlation set

**Problem.** Identical and unrelated pairs are added to both train and test. Passing the checks is then guaranteed, and the correlation jump is a variance artifact: BERTScore goes 0.46 → 0.94 in Pearson while its RMSE *degrades* 3.61 → 5.09.

**Do this.**
- Report Pearson/RMSE **only on the human-annotated test split**, for every metric, in a single un-augmented table. Report sanity-check pass rates as a separate, orthogonal diagnostic — never mixed into the correlation.
- Make the sanity-check hold-out genuinely held out: different legal sources, different clause types, and near-duplicate (not just unrelated) negatives.
- Add a *hard* third check the DA cannot trivially teach: **minimal legal edits** — pairs differing by one legally decisive token (`doit`/`peut`, `automobile`/`véhicule`, `et`/`ou`, a negation, a numeric cap `2 000 $` → `20 000 $`, an inclusive/exclusive list marker). A metric that scores these near 100 % is unusable for law regardless of its Pearson. This is the single most diagnostic experiment the paper is missing, and it can be built semi-automatically from the existing corpus.
- Correct the abstract to distinguish JUDGEBERT (0/0 on the checks) from JUDGEBERT-DA (100/100), and disclose that the latter was trained on that distribution.

**Effort:** low–medium. **Impact:** very high — currently the headline r = 0.97 will not survive scrutiny.

---

### 4. Add trivial and calibrated baselines

**Problem.** All comparison metrics are unsupervised and never fitted to the Likert scale; JUDGEBERT is supervised on it. The RMSE column and Table 6 therefore mostly measure "was this metric allowed to see the label distribution", not metric quality. And no length/overlap baseline is reported at all.

**Evidence.** On the released corpus: `r(|len(simp) − len(orig)|, label) = −0.62`; `r(len ratio, label) = 0.50`; `r(token Jaccard, label) = 0.36`. A 6-feature linear model over lengths and set overlap gets **r = 0.61 / RMSE = 1.90 (10-fold CV)** — above every Transformer baseline in Table 5 and level with the human ceiling.

**Do this.**
- Add a **trivial-features row** to Table 5 and report how much JUDGEBERT gains over it. If the gap is small, the honest conclusion is that most current LMP signal is omission-as-length, which is itself a publishable finding about the task.
- **Calibrate every baseline**: fit a monotone map (isotonic regression or a 1-D linear fit) from each metric's raw score to the human scale *on the training split*, then report RMSE and over-prediction rate on test. Report both raw and calibrated numbers.
- Add a same-supervision control: fine-tune CamemBERTv2 on a *non-legal* meaning-preservation target and test on FrJUDGE, to separate "supervised on this data" from "learned legal meaning".

**Effort:** low. **Impact:** very high — this is what a determined reviewer will do, so do it first.

---

### 5. Compare against modern, language-appropriate, and LLM-judge baselines

**Problem.** Six of seven baselines are monolingual **English** models applied to French. The paper concedes this makes the comparison unfair, which unfortunately also makes it uninformative — none of the numbers tell you whether a competent French metric would have sufficed.

**Do this.**
- **Language-matched embeddings:** BERTScore with CamemBERTv2/FlauBERT, multilingual BERTScore (XLM-R), Sentence-CamemBERT, LaBSE, E5-multilingual, Solon/BGE-M3.
- **Adequacy metrics built for exactly this:** COMET / COMET-Kiwi (reference-free), BLEURT, and the factual-consistency family — AlignScore, SummaC, and QAFactEval-style QA metrics run with French QG/QA models (a fair version of the QuestEval baseline).
- **Bidirectional NLI**, which matches the construct better than similarity: score entailment(original → simplification) *and* entailment(simplification → original) with mDeBERTa-XNLI or a French NLI model. Omissions and hallucinations are precisely the two directions of entailment failure, so this is the natural strong baseline for the paper's own error taxonomy.
- **LLM-as-judge**, given the *same annotation guide the law students received* — GPT-4/5-class, Claude, and an open French model — zero-shot, few-shot, and with the three-step bracket-then-deduct procedure. This is the baseline every reader will ask about, and if a 112M-parameter CamemBERT beats it, that is a strong, cheap, deployable result worth foregrounding. Also report cost/latency per 1,000 pairs to make the efficiency argument concrete.
- Report the **encoder ablation you already ran**: the repo contains `sweep_bert_base.sh`, `sweep_flaubert_base*.sh`, `sweep_camembert_base*.sh`, `sweep_bert_large.sh`. Those results belong in the paper.

**Effort:** medium. **Impact:** high — converts a conceded-unfair comparison into a real one.

---

### 6. Predict the error taxonomy, not just a scalar — make the metric explainable

**Problem.** The score is *constructed* as `bracket_max − #errors` over four error types (hallucination, omission, consistency, confusion), yet the model regresses only the scalar, and the error labels are not in the release. A legal practitioner told "6.2/10" cannot act; told "omission: the CAD 2,000 cap is missing" they can.

**Do this.**
- Multi-task head: scalar LMP + 4-way multi-label error prediction + (auxiliary) 18-class characterization. Report whether the auxiliary tasks improve the scalar — the deterministic scoring rule means the error labels are close to a full decomposition of the target, so this should help *and* it gives the model a legally meaningful inductive bias.
- **Span-level output:** highlight the omitted or hallucinated span. The free-text `justifications` field (populated for all 1,485 rows) is a ready-made weak-supervision source for this, and could be converted into spans with modest additional annotation.
- Sanity-check the arithmetic: verify on the released data that `label = bracket_max − #errors` actually holds, and report any deviation as a measure of protocol drift.

**Effort:** medium. **Impact:** high — explainability is the difference between a leaderboard metric and something a legal team will deploy, and it is directly aligned with the paper's own Ethics section warning about "blind faith in users who trust such metrics".

---

### 7. Establish generalization: multiple ATS systems, and a real out-of-domain split

**Problem.** Every simplification is from one model, one prompt, `max_new_tokens=100`, `temperature=1.0`. The limitations section admits no OOD split was tested. As written, JUDGEBERT could be a GPT-4-turbo error detector for Quebec property-insurance forms.

**Do this.**
- **Vary the generator** to spread the quality distribution: a small open French model (Croissant/Mistral-fr), a fine-tuned mT5/BARThez simplifier, a rule-based/sentence-splitting system, a human simplification, and deliberately degraded outputs. A metric meta-evaluation is only meaningful across systems of differing quality.
- **Leave-one-source-out** evaluation between the BAC home-insurance form and the AMF auto form, and **leave-one-characterization-class-out** (18 classes are already annotated) — do exclusions transfer to definitions?
- **Cross-domain probe:** score a handful of non-insurance French legal texts (the paper's own future-work suggestion of arbitration/mediation clauses, plus EUR-Lex-Sum FR) and report degradation, even without gold labels, via agreement with a small expert spot-check.
- Report the **length-controlled** result: bucket the test set by `|len(simp) − len(orig)|` and show correlation within each bucket. This directly answers idea 4's challenge.

**Effort:** medium–high (needs some new annotation). **Impact:** high — this is what "is the metric real?" reduces to.

---

### 8. Fix the statistics: CIs, ordinal correlations, significance, and power

**Problem.** Point estimates only, on a test split of roughly 89 items, with an ordinal bimodal target scored by Pearson.

**Do this.**
- Bootstrap 95 % CIs on every Pearson and RMSE in Tables 5 and 6, and test JUDGEBERT vs. the runner-up with a paired bootstrap / Williams test for dependent correlations. The 10-fold std devs already reported (±0.02) measure seed variance, not sampling variance — say so.
- Add **Spearman ρ and Kendall τ** (the primary numbers, arguably) plus **Krippendorff's α for ordinal data** between metric and humans, so the metric is judged on the same scale type as the annotations.
- Report a **power analysis** in the spirit of Card et al. (2020), which is already cited: what correlation gap is detectable at n ≈ 89? This also motivates idea 9.
- Report the *asymmetric* result properly: Table 6 deserves a full over/under-prediction breakdown with CIs and a cost-weighted score (over-prediction penalized more heavily than under-prediction), since the paper argues that permissiveness is the dangerous failure mode. Consider making a quantile/pinball-loss objective the training target so the model is deliberately conservative by construction.

**Effort:** low. **Impact:** medium–high — cheap credibility.

---

### 9. Grow and rebalance the corpus, targeting agreement rather than raw volume

**Problem.** 297 pairs at α = 0.10 is a hard ceiling on what can be concluded. More data at the same agreement level does not help much; more *reliable* data does.

**Do this.**
- **Adjudication round:** re-annotate the ~63 % of pairs where the five ratings span ≥ 6 points, with discussion between the two annotator clusters and a senior lawyer adjudicating. Report agreement before/after; the delta quantifies how much of the disagreement is protocol ambiguity vs. genuine legal indeterminacy — which is itself a contribution.
- **Sharpen the guidelines** on the specific phenomena that split the clusters (sentence splitting, generic-vs-specific term substitution, implied context/case law). The paper notes D and E may be importing external legal context; the guide should state explicitly whether that is in scope.
- **Anchor examples** on the scale: give annotators 3–4 calibrated reference pairs per bracket. Low α on a 10-point scale usually means the scale lacked anchors.
- **Reduce the scale.** Since annotators naturally used the three brackets (as Garneau et al. found), consider making the *bracket* the primary label and the deduction a secondary one, then evaluate ordinal classification. This is likely to raise α substantially at no cost in usefulness.
- Then expand: group insurance, other provinces, and — the highest-value addition — **document-context pairs**, since the limitations section correctly notes that clause-level judgment out of context is not how contracts are read.

**Effort:** high (new annotation budget). **Impact:** high — everything else is capped by label quality.

---

### 10. Ship a usable artifact and fix the paper–repo discrepancies

**Problem.** A metric is adopted only if it is one `pip install` away. Today: no weights are published, the README is copy-pasted from an unrelated grammaticality project (it documents a binary `label` of "0 = ungrammatical / 1 = grammatical", a `source` URL and an "aggregated BDL category" — none of which exist in `FrJUDGE.jsonl`, whose real fields are `original`, `simplification`, `simplicity`, `qualification`, `legal_meaning`, `justifications`), the `load_dataset` snippet points at the wrong path, the repo's own `TODO` still lists "Add the corpus on HG", and the license disagrees with the paper (§3 says CC-BY 4.0; README says CC-BY-NC-SA 4.0 — for a corpus derived from BAC/AMF forms this needs to be stated once, correctly, and consistently).

**Do this.**
- Publish **model weights on HuggingFace** (both JUDGEBERT and JUDGEBERT-DA, all 10 seeds or at least the median seed) plus a `judgebert` package with a two-line scoring API, mirroring what MeaningBERT did — that is the reason MeaningBERT is the metric this paper had to compete with.
- Publish the **dataset as a HuggingFace dataset** with a proper card, the official train/dev/test splits, **annotator IDs**, and the **error-type labels** used in the scoring rule.
- **Fix a de-anonymisation leak.** The paper pseudonymises the five annotators as A–E, but `datastore/annotated/insurance_text_simplifications_annotated.jsonl` ships raw Prodigy output whose `_annotator_id` / `_session_id` fields carry their **real first names** (Clara, Eve, Simon, William, Yanis), each joined to per-annotation Unix `_timestamp`s and free-text justifications. That is a re-identifiable record of five named law students' individual judgments and working pace, published under the paper's own pseudonymisation promise. Replace the names with stable pseudonyms (`A`–`E`) in the committed file, and rewrite git history if the consent form did not cover named release.
- Rewrite the README against the actual schema; resolve the license conflict; note that `datastore/released/FrJUDGE.jsonl` contains 1,485 rows (297 pairs × 5 annotators, un-aggregated) and document how the paper's aggregated label is derived.
- **Small corrections for the camera-ready/v2:** Table 7 "Top K 0.9" should be `top_p = 0.9` (OpenAI exposes no `top_k`); §3.2.2 describes Kandel & Moles (1958) as a "Flesch-Kincaid grade level" with `≤ 50` meaning *harder* and a discarded example at 69.87 — that is French Flesch **Reading Ease** (0–100, higher = easier), so rename it and restate the threshold; §6.2 says "only three metrics" then lists four; Table 5's `%>1%` header contradicts the text's "equal or below 1 %"; and SBERT-Multi's identical-pair score differs between the DA-False (0.00) and DA-True (100.00) blocks although it is an untrained metric that cannot be affected by JUDGEBERT's training regime — one of those two cells is an error.

**Effort:** low. **Impact:** high — this is the difference between a cited paper and a used one, and several of the fixes take minutes.

---

## Suggested priority

| Priority | Ideas | Rationale |
|---|---|---|
| **Do first (changes the conclusions)** | 1, 3, 4 | Human ceiling, DA confound, trivial baselines. Cheap, analysis-only, and they determine what the paper can honestly claim. |
| **Do next (strengthens the contribution)** | 2, 5, 8, 10 | Disagreement modelling, fair baselines, statistics, artifact release. |
| **Follow-up paper** | 6, 7, 9 | Explainable error prediction, generalization study, corpus v2 with adjudication. |

## Sources

- [arXiv:2508.16870 — JUDGEBERT](https://arxiv.org/abs/2508.16870)
- [ACL Anthology 2025.emnlp-main.5](https://aclanthology.org/2025.emnlp-main.5/)
- [GRAAL-Research/JUDGEBERT (GitHub)](https://github.com/GRAAL-Research/JUDGEBERT)
- [MeaningBERT: Assessing Meaning Preservation Between Sentences](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10557945/)
- [InfoLossQA: Characterizing and Recovering Information Loss in Text Simplification](https://arxiv.org/pdf/2401.16475)
- [Text Simplification for Legal Domain: Insights and Challenges (NLLP 2022)](https://aclanthology.org/2022.nllp-1.28.pdf)
- [DETECT: Determining Ease and Textual Clarity of German Text Simplifications](https://arxiv.org/pdf/2510.22212)
- [From jargon to clarity: graded simplification of legal data (AI & Law)](https://link.springer.com/article/10.1007/s10506-026-09503-y)

*All quantitative claims above marked "on the released corpus" were recomputed directly from the repository: `datastore/released/FrJUDGE.jsonl` (1,485 rows, 297 unique pairs, 5 ratings each) and `datastore/annotated/insurance_text_simplifications_annotated.jsonl` (same annotations with `_annotator_id`, `_timestamp` and the Prodigy option set; note the file is concatenated JSON objects, not strict one-object-per-line JSONL). The ceiling in idea 1 is a true leave-one-annotator-out ceiling over all 297 complete pairs.*
