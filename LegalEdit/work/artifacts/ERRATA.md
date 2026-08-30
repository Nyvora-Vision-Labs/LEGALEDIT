# Errata and discrepancies: JUDGEBERT (arXiv:2508.16870v1 / EMNLP 2025 main, pp. 92–118)

Offered as corrigenda for a v2 or journal extension, not as criticism of the
study's conduct. Items 1–3 affect a reported result; the rest are textual or
artefact issues. Every item was verified against the paper and the released
data.

## Affecting a reported number

**1. Krippendorff's α for legal meaning uses the wrong coefficient.**
The reported α = 0.10 is the **nominal** coefficient. We reproduce it exactly
(0.1042). But legal meaning is a ten-point **ordinal** scale, and nominal α
counts a 9-vs-10 disagreement as fully as a 1-vs-10 disagreement. The
appropriate ordinal coefficient is **α = 0.325** (interval: 0.315). The
reported values for simplicity (0.18) and characterization (0.55) are correct,
because those are genuinely nominal. *This correction favours the paper, since the
corpus is more reliable than it claims.*

**2. Table 5's DA=True block is not comparable to the DA=False block.**
Adding identical pairs (label 10) and unrelated pairs (label 1) to the
*evaluation* set inflates label variance at the extremes, raising Pearson *r*
for every metric without any improvement in judgment. The tell is BERTScore:
*r* rises 0.46 → 0.94 while its RMSE simultaneously **worsens** 3.61 → 5.09.
Correlation should be reported on the human-annotated split only, with
sanity-check pass rates kept separate.

**3. SBERT-Multi's identical-pair score is inconsistent between blocks.**
Table 5 gives SBERT-Multi 0.00 on the identical check in the DA=False block and
100.00 in the DA=True block. SBERT-Multi is an untrained cosine metric; its
behaviour on identical pairs cannot depend on how JUDGEBERT was trained. One of
the two cells is wrong.

## Textual

**4. Generation parameters (Table 7).** "Top K 0.9": 0.9 is a `top_p` value,
and the OpenAI API exposes no `top_k` parameter for these models.

**5. Readability metric (§3.2.2).** Kandel and Moles (1958) is described as the
"French Flesch-Kincaid grade level" with a threshold of ≤ 50 selecting harder
text, and an example is discarded for scoring 69.87. That behaviour identifies
it as the French adaptation of Flesch **Reading Ease** (0–100, *higher = easier*),
not a grade level. The name and the direction of the threshold should be
restated.

**6. Count mismatch (§6.2).** "only three metrics always return the expected
value … BERTScore, SBERT, SBERT-Multi and MeaningBERT", but four are listed.

**7. Column header (Table 5).** The fourth column is headed `%>1%` with ↑, but
§5.2 defines it as the ratio of unrelated pairs scoring *at or below* 1%.

**8. Uniqueness.** The corpus contains 297 pairs over **296** unique original
sentences; one original appears with two different simplifications.

## Artefacts

**9. README documents the wrong schema.** The repository README describes
`label` (binary, "0 = ungrammatical, 1 = grammatical"), `sentence`, `source`
(URL) and an "aggregated BDL category", the schema of a different project.
`FrJUDGE.jsonl`'s actual fields are `original`, `simplification`, `simplicity`,
`qualification`, `legal_meaning`, `justifications`. The `load_dataset` snippet
also points at a non-existent path. See `README_corrected.md`.

**10. License conflict.** §3 of the paper states CC-BY 4.0; the README states
CC-BY-NC-SA 4.0.

**11. Annotator de-anonymisation.** The paper pseudonymises annotators as A–E,
but `datastore/annotated/insurance_text_simplifications_annotated.jsonl` records
their real first names in `_annotator_id` / `_session_id`, joined to
per-annotation Unix timestamps and free-text justifications. See
`pseudonymise_frjudge.py`. Note that repairing the working file does not remove
the names from git history.

**12. File format.** The annotated export is *concatenated* JSON objects, not
JSONL; naive line-by-line parsing fails with `JSONDecodeError: Extra data`.
Worth a note in the README.

**13. Annotator IDs and error-type labels are not in the public release.**
`FrJUDGE.jsonl` drops the annotator, making inter-annotator agreement and
human-ceiling analysis irreproducible from the public file. The four error
types that drive the scoring rule (hallucination, omission, consistency,
confusion) are not released at all.

**14. Unreported ablation.** `article_src/src/` contains sweep scripts for
BERT-base, BERT-large, FlauBERT and CamemBERT, with and without DA. The encoder
comparison those imply is not in the paper and would strengthen it.

**15. No model weights.** The metric is not distributed, which is the principal
obstacle to it being adopted the way MeaningBERT was.
