# JUDGEBERT / FrJUDGE: corrected repository README

> Drop-in replacement for the current `README.md` of
> [GRAAL-Research/JUDGEBERT](https://github.com/GRAAL-Research/JUDGEBERT).
> The published README documents the schema of a *different* project (a binary
> grammaticality dataset with `label` / `sentence` / `source` / `category`
> fields and an "aggregated BDL category"). None of those fields exist in
> `FrJUDGE.jsonl`. The text below documents the files that are actually in the
> repository.

# JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences

Official dataset and source code for
[JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences](https://aclanthology.org/2025.emnlp-main.5/)
(EMNLP 2025, pages 92–118).

## Dataset

FrJUDGE contains **297 sentence pairs**, each pairing a legal sentence from a
Quebec property-damage insurance form with a GPT-4-turbo simplification of it,
rated by **five** law students on three dimensions.

### Files

| Path | Rows | Contents |
|---|---|---|
| `datastore/released/FrJUDGE.jsonl` | 1,485 | One row per (pair, annotator) = 297 × 5. **No annotator identifier.** |
| `datastore/annotated/insurance_text_simplifications_annotated.jsonl` | 1,485 | Raw Prodigy export: same annotations plus annotator id, source document, timestamps, and the full option set. |
| `datastore/training/*.jsonl` | n/a | The train/dev/test splits used in the paper, with and without data augmentation. |
| `datastore/references/A-25.jsonl` | 254 | Quebec Automobile Insurance Act, used to build unrelated-sentence pairs. |
| `datastore/references/code_securite_routiere.jsonl` | 875 | Quebec Highway Safety Code, likewise. |

> **Format warning.** `insurance_text_simplifications_annotated.jsonl` is
> *concatenated JSON objects*, not one object per line. Reading it with
> `for line in f: json.loads(line)` raises `JSONDecodeError: Extra data`.
> Use `json.JSONDecoder().raw_decode` in a loop.

### Fields of `FrJUDGE.jsonl`

| Field | Type | Description |
|---|---|---|
| `original` | str | The legal sentence (1–5 sentences) from the insurance form. |
| `simplification` | str | The GPT-4-turbo zero-shot simplification. |
| `simplicity` | str | One of `Plus simple à lire`, `Aussi simple à lire`, `Plus difficile à lire`, `Pas de simplification`. |
| `qualification` | str | Legal characterization class, `"1"`–`"18"` (see the paper, Appendix D). |
| `legal_meaning` | str | This annotator's legal-meaning rating, `"1"`–`"10"`. |
| `justifications` | str | The annotator's free-text rationale (French). Populated for all rows. |

Each `(original, simplification)` pair appears **five times**, once per
annotator. The paper's gold label is the **mean** of the five `legal_meaning`
values; `simplicity` and `qualification` are resolved by majority vote with
ties broken at random.

### Loading

```python
import json, collections

rows = [json.loads(l) for l in open("datastore/released/FrJUDGE.jsonl")]

pairs = collections.defaultdict(list)
for r in rows:
    pairs[(r["original"], r["simplification"])].append(int(r["legal_meaning"]))

gold = {k: sum(v) / len(v) for k, v in pairs.items()}   # 297 items
```

The HuggingFace snippet in the previous README pointed at a non-existent path.
The working equivalent is:

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="datastore/released/FrJUDGE.jsonl")
```

### Known issues

- **Annotator identity.** `FrJUDGE.jsonl` omits the annotator, so
  annotator-level analysis (inter-annotator agreement, disagreement modelling,
  human ceilings) is not reproducible from the public file alone. The raw export
  retains it, but in the form of the annotators' **real first names**, which the
  paper pseudonymises as A–E. See `pseudonymise_frjudge.py` in this directory
  for a repair script that assigns stable pseudonyms and coarsens timestamps.
- **Error-type labels.** The annotation protocol has annotators identify four
  error types (hallucination, omission, consistency, confusion) and deduct one
  point per error. Those labels are not in either released file; only the
  resulting score and the free-text justification survive.
- **License.** The paper states CC-BY 4.0 (§3); this README previously stated
  CC-BY-NC-SA 4.0. These need to be reconciled, and the source forms are
  published by the BAC and the AMF, so the applicable terms should be stated
  explicitly.

## Model

The paper's trained metric is not currently distributed. For the metric to be
usable as a metric it needs published weights and a scoring API, as its
predecessor MeaningBERT has
(<https://huggingface.co/davebulaval/MeaningBERT>).

## Source code

`article_src/` contains the code used to clean the dataset, compute the
statistics, and train the models. Encoder sweeps for BERT, FlauBERT and
CamemBERT (`article_src/src/sweep_*.sh`) were run but are not reported in the
paper.

## Citation

```bibtex
@inproceedings{beauchemin2025judgebert,
  title     = {JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences},
  author    = {Beauchemin, David and Albert-Rochette, Michelle and
               Khoury, Richard and D{\'e}ziel, Pierre-Luc},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in
               Natural Language Processing},
  pages     = {92--118},
  year      = {2025}
}
```
