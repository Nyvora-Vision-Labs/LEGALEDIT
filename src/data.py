"""Canonical loader for FrJUDGE, rebuilt from the raw Prodigy export.

The public release (datastore/released/FrJUDGE.jsonl) drops annotator identity;
the raw export (datastore/annotated/...) keeps it in `_annotator_id`. We rebuild
from the raw file so annotator-level analysis is possible, and we pseudonymise
the five annotators as A-E on load (the raw file carries their real first names).
"""
import json, re, os, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# real first names -> stable pseudonyms, assigned by sorted name for determinism
PSEUDONYMS = ["A", "B", "C", "D", "E"]


def _read_concat_json(path):
    """The raw export is concatenated JSON objects, not strict JSONL."""
    dec = json.JSONDecoder()
    s = open(path, encoding="utf-8").read()
    out, i = [], 0
    while i < len(s):
        while i < len(s) and s[i] in " \n\r\t":
            i += 1
        if i >= len(s):
            break
        o, i = dec.raw_decode(s, i)
        out.append(o)
    return out


def _raw_id(rec):
    """The annotator token as the export writes it, name or pseudonym."""
    a = rec["_annotator_id"]
    m = re.match(r"legal_annotations_\w+?_\d+-(\w+)", a)
    return m.group(1) if m else a


def _pseudonyms(rows):
    """Map each annotator token to A-E, ordered by the token itself.

    The released data/annotated.jsonl is already pseudonymised, so this is the
    identity there. Pointed at the upstream raw export, which carries the
    annotators' real first names, it reproduces the same A-E assignment that
    pseudonymise_frjudge.py makes, so results are identical either way and no
    name is ever hard-coded here.
    """
    return {n: PSEUDONYMS[i] for i, n in enumerate(sorted({_raw_id(r) for r in rows}))}


def load_pairs():
    """Return list of dicts, one per (original, simplification) pair.

    Keys: original, simplification, source, ratings {annot: int},
          simplicity {annot: str}, characterization {annot: str},
          justifications {annot: str}, label (mean legal meaning).
    """
    rows = _read_concat_json(os.path.join(DATA, "annotated.jsonl"))
    pseudo = _pseudonyms(rows)
    by_pair = collections.OrderedDict()
    for r in rows:
        k = (r["original"], r["simplified"])
        d = by_pair.setdefault(
            k,
            dict(original=r["original"], simplification=r["simplified"],
                 source=r.get("meta", {}).get("source", ""),
                 ratings={}, simplicity={}, characterization={}, justifications={}),
        )
        a = pseudo[_raw_id(r)]
        d["ratings"][a] = int(r["evaluation"])
        d["simplicity"][a] = r["difficulty_evaluation"]
        acc = r.get("accept") or []
        d["characterization"][a] = acc[0] if acc else None
        d["justifications"][a] = r.get("justifications", "")
    pairs = [p for p in by_pair.values() if len(p["ratings"]) == 5]
    for p in pairs:
        p["label"] = float(np.mean([p["ratings"][a] for a in "ABCDE"]))
        # majority-vote characterization, ties broken by lowest class id
        c = collections.Counter(v for v in p["characterization"].values() if v)
        p["char_major"] = min(c.most_common(), key=lambda kv: (-kv[1], int(kv[0])))[0] if c else None
    return pairs


ANNOTATORS = list("ABCDE")

if __name__ == "__main__":
    P = load_pairs()
    print("pairs:", len(P))
    print("sources:", collections.Counter(p["source"] for p in P))
    M = np.array([[p["ratings"][a] for a in ANNOTATORS] for p in P], float)
    print("per-annotator mean:", dict(zip(ANNOTATORS, M.mean(0).round(2))))
    print("label mean/sd:", round(np.mean([p["label"] for p in P]), 2),
          round(np.std([p["label"] for p in P]), 2))
    print("char classes:", len(set(p["char_major"] for p in P)))
