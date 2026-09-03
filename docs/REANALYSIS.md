# Re-analysis notes: JUDGEBERT / FrJUDGE

These notes hold the material that was removed from the paper
(`paper/main.tex`) when it was restructured around the **LegalEdit**
diagnostic. They are direct comparisons against the numbers reported in
[Beauchemin et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.5/)
(arXiv:2508.16870v1) and are recorded here rather than in the paper for two
reasons: they are corrections to another group's reported results rather than
findings about metrics in general, and every one of them has been communicated
to those authors directly. `ERRATA.md` holds the smaller textual and artefact
discrepancies; this file holds the four that change how a reported number should
be read.

Every figure below is reproduced by the pipeline in `src/`. Where the
paper still reports the underlying quantity, the section reference points at it.

## 1. The reliability coefficient is nominal where the scale is ordinal

The published α = 0.10 for legal meaning is the **nominal** coefficient. We
reproduce it exactly (0.1042) from the raw export. Legal meaning is a ten-point
**ordinal** scale, and nominal α counts a 9-vs-10 disagreement as fully as a
1-vs-10 disagreement; the ordinal coefficient is **α = 0.325** (interval:
0.315). The reported values for simplicity (0.18) and characterization (0.55)
are correct, since those dimensions are genuinely nominal.

This correction runs in the corpus's favour: FrJUDGE is more reliable than its
published coefficient makes it look. The paper reports both coefficients in
§6.1 without characterising the published value as an error.

## 2. The reported correlation sits above the measurable human ceiling

FrJUDGE ships five ratings per item, so a leave-one-annotator-out ceiling is
computable. A held-out annotator predicts the mean of the other four at
**r = 0.597** (RMSE 3.27); the best-agreeing *pair* of annotators reaches
**r = 0.606**. JUDGEBERT is reported at r = 0.74.

A model above 0.597 is not thereby superhuman: predicting a five-rater mean is
an easier task than being one of the raters, and it rewards predicting the
middle of a bimodal distribution. The appropriate upper bound for predicting the
observed mean is the Spearman–Brown reliability of that mean (0.811, so a metric
may approach √0.811 = 0.90). A reported 0.74 sits between the two bounds: it is
evidence for "as good as the aggregate allows" and evidence about nothing at all
with respect to "as good as an expert."

The paper reports both bounds (§6.2) and recommends reporting against both,
without staging the comparison against the reported 0.74.

## 3. The augmentation gain does not survive a clean evaluation split

The published result has data augmentation raising r from 0.74 to 0.97. That
figure is computed on an evaluation set that **contains the augmentation**:
identical pairs carry label 10 and unrelated pairs label 1, and between them
they supply most of the label variance, which raises Pearson r for every metric
without any improvement in judgment.

The tell is in the published table itself: BERTScore moves from r = 0.46 to
0.94 while its RMSE simultaneously **worsens** from 3.61 to 5.09. Correlation
and RMSE improving and worsening together is the signature of variance
inflation, not of better measurement.

Retrained under an evaluation that scores only the human-annotated test split:

| configuration | r (5 seeds) | vs. un-augmented baseline |
|---|---|---|
| JudgeBERT-Scalar (repro) | 0.542 ± 0.082 | n/a |
| JudgeBERT-DA | 0.192 ± 0.254 | −0.350, on 5 seeds of 5 |
| JudgeBERT-DA+LegalEdit | 0.122 | −0.420, on 5 seeds of 5 |

DA reaches r = −0.188 on one seed. Scored on the human items alone, the
augmentation costs more than half the correlation rather than adding a fifth.
The paper reports this in §7 and Appendix A as a property of the evaluation
protocol, without contrasting it with the published 0.97.

Correlation should be reported on the human-annotated split only, with
sanity-check pass rates kept separate.

## 4. Sanity-check pass rates reported on training-distribution probes

JUDGEBERT-DA is reported at 100% on both the identical-pair and unrelated-pair
checks, with those exact distributions in its training mixture. Our augmented
models pass the unrelated check on 77% and 69% of probes and the identical check
on **none** of them. The only difference between those two outcomes is whether
the probe comes from the same documents as the training pairs: our unrelated
probes are drawn from the same two statutes as the mixture's unrelated pairs
(in-distribution), while our identical probes are statutory sentences and the
mixture's identical pairs are FrJUDGE insurance clauses (out-of-distribution).

Training on a constraint buys satisfaction of it in distribution and nothing
across a change of source. A report should separate "satisfies the constraint by
construction" from "generalises to the constraint"; only the second is evidence
about a metric. The paper makes this point in §7 about our own models.

## Annotator de-anonymisation

See `ERRATA.md`, item 11. This was raised with the authors directly and is not
discussed in the paper; the paper's ethics section states only that our own
release pseudonymises annotators on load and ships the pseudonymisation script.
