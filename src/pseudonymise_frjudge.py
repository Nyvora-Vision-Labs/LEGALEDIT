#!/usr/bin/env python3
"""Repair the annotator de-anonymisation in the released FrJUDGE raw export.

The paper pseudonymises its five annotators as A-E, but
`datastore/annotated/insurance_text_simplifications_annotated.jsonl` records
their real first names in `_annotator_id` and `_session_id`, joined to
per-annotation timestamps. This script rewrites those fields to stable
pseudonyms and coarsens timestamps to the day, preserving every field a
researcher needs (annotator-level analysis stays possible) while removing the
personal identifiers.

Usage:
    python3 pseudonymise_frjudge.py IN.jsonl OUT.jsonl [--map map.json]

Note: the file is *concatenated* JSON objects, not one object per line, so it
cannot be read with a naive `for line in f: json.loads(line)`.
"""
import argparse, hashlib, json, re, sys


def read_concat_json(path):
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


def real_name(rec):
    a = rec.get("_annotator_id", "")
    m = re.match(r"legal_annotations_\w+?_\d+-(\w+)", a)
    return m.group(1) if m else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--map", default=None, help="write the name->pseudonym map here")
    a = ap.parse_args()

    rows = read_concat_json(a.infile)
    names = sorted({real_name(r) for r in rows})
    # deterministic, order-independent assignment
    mapping = {n: chr(ord("A") + i) for i, n in enumerate(names)}

    for r in rows:
        p = mapping[real_name(r)]
        r["_annotator_id"] = p
        if "_session_id" in r:
            # keep the batch label, drop the name
            r["_session_id"] = re.sub(r"-\w+$", f"-{p}", r["_session_id"])
            r["_session_id"] = re.sub(r"legal_annotations_\w+?_(\d+)",
                                      r"legal_annotations_batch\1", r["_session_id"])
        if "_timestamp" in r:
            r["_timestamp"] = int(r["_timestamp"]) // 86400 * 86400   # day resolution

    with open(a.outfile, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")   # proper JSONL
    if a.map:
        json.dump(mapping, open(a.map, "w"), indent=2)
    print(f"{len(rows)} records; {len(names)} annotators pseudonymised -> "
          f"{sorted(mapping.values())}", file=sys.stderr)
    print("NOTE: rewriting the working file does not remove the names from git "
          "history; a history rewrite is also required.", file=sys.stderr)


if __name__ == "__main__":
    main()
