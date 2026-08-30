"""Sanity-check pairs and the new LEGALEDIT challenge set.

Three regimes, deliberately chosen so surface similarity and legal meaning
dissociate:

  identical   high surface similarity, meaning fully preserved   -> gold 10
  legaledit   high surface similarity, legal meaning FLIPPED     -> gold  1
  unrelated   low  surface similarity, no shared meaning         -> gold  1

The original paper only used `identical` and `unrelated`; those two are jointly
solvable by any lexical-overlap function, which is why every embedding metric
"passes" the identical check. `legaledit` is the discriminating case.
"""
import json, os, re, random
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


# ---------------------------------------------------------------- unrelated --
def _sents(path, lo=60, hi=400):
    out = []
    for line in open(os.path.join(DATA, path), encoding="utf-8"):
        t = json.loads(line)["text"]
        for s in re.split(r"(?<=[.;])\s+", t.replace("\n", " ")):
            s = re.sub(r"\s+", " ", s).strip()
            s = re.sub(r"^\d+(\.\d+)*\.?\s*", "", s)
            if lo <= len(s) <= hi and s[0].isupper():
                out.append(s)
    return out


def _tok(s):
    return re.findall(r"\w+", s.lower())


def _rouge1(a, b):
    A, B = _tok(a), _tok(b)
    if not A or not B:
        return 0.0
    import collections
    ca, cb = collections.Counter(A), collections.Counter(B)
    ov = sum((ca & cb).values())
    p, r = ov / len(B), ov / len(A)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def unrelated_pairs(n, seed=0, max_rouge=0.25):
    """Paper's protocol: A-25 sentence x Code de la securite routiere sentence,
    kept only if lexical overlap stays under threshold (similar register,
    unrelated content)."""
    a25, csr = _sents("A-25.jsonl"), _sents("code_securite_routiere.jsonl")
    rng = random.Random(seed)
    out, tries = [], 0
    while len(out) < n and tries < n * 200:
        tries += 1
        x, y = rng.choice(a25), rng.choice(csr)
        if _rouge1(x, y) <= max_rouge:
            out.append((x, y))
    return out


# ---------------------------------------------------------------- legaledit --
# Each rule rewrites a legally decisive token. Every rewrite changes who is
# bound, what is covered, how much, or whether a condition is conjunctive --
# while leaving >90% of the surface form intact.
_RULES = [
    ("modal_obligation_to_permission", [(r"\bdoit\b", "peut"), (r"\bdoivent\b", "peuvent"),
                                        (r"\bdevra\b", "pourra"), (r"\bdevront\b", "pourront")]),
    ("modal_permission_to_obligation", [(r"\bpeut\b", "doit"), (r"\bpeuvent\b", "doivent")]),
    ("connective_and_to_or", [(r"\bet\b", "ou")]),
    ("connective_or_to_and", [(r"\bou\b", "et")]),
    ("party_insured_to_insurer", [(r"\bl'assuré\b", "l'assureur"), (r"\bassuré\b", "assureur"),
                                  (r"\bl’assuré\b", "l’assureur")]),
    ("party_insurer_to_insured", [(r"\bl'assureur\b", "l'assuré"), (r"\bassureur\b", "assuré"),
                                  (r"\bl’assureur\b", "l’assuré")]),
    ("coverage_include_to_exclude", [(r"\bcouvre\b", "ne couvre pas"), (r"\binclut\b", "exclut"),
                                     (r"\bcomprend\b", "exclut"), (r"\bgaranti\b", "exclu")]),
    ("coverage_exclude_to_include", [(r"\bexclut\b", "inclut"), (r"\bne couvre pas\b", "couvre"),
                                     (r"\bexclus\b", "inclus"), (r"\bexclue\b", "incluse")]),
    ("negation_drop", [(r"\bne (\w+) pas\b", r"\1"), (r"\bn'(\w+) pas\b", r"\1"),
                       (r"\bn’(\w+) pas\b", r"\1")]),
    ("scope_all_to_some", [(r"\btous les\b", "certains"), (r"\btoutes les\b", "certaines"),
                           (r"\btout\b", "certain"), (r"\btoute\b", "certaine")]),
    ("vehicle_class", [(r"\bl['\u2019]automobile\b", "le véhicule"),
                       (r"\ble véhicule\b", "l'automobile"),
                       (r"\bun véhicule\b", "une automobile"),
                       (r"\bune automobile\b", "un véhicule"),
                       (r"\bautomobile\b", "véhicule"), (r"\bvéhicule\b", "automobile")]),
    ("temporal_flip", [(r"\bavant\b", "après"), (r"\baprès\b", "avant")]),
    ("condition_unless_to_if", [(r"\bsauf si\b", "si"), (r"\bà moins que\b", "pourvu que"),
                                (r"\bsauf\b", "y compris")]),
    ("obligation_of_result", [(r"(?<![nN]['\u2019] )\best tenu\b", "n'est pas tenu"),
                              (r"\bn['\u2019]est pas tenu\b", "est tenu")]),
]

_AMOUNT = re.compile(r"(\d[\d\s ]*(?:,\d+)?)\s*(\$|%|jours?|mois|ans?)")


def _amount_edit(s):
    """Multiply the first monetary/temporal quantity by ten."""
    m = _AMOUNT.search(s)
    if not m:
        return None
    raw = m.group(1)
    digits = re.sub(r"[\s ]", "", raw).replace(",", ".")
    try:
        v = float(digits)
    except ValueError:
        return None
    new = v * 10
    txt = ("%d" % new) if new == int(new) else ("%.2f" % new).replace(".", ",")
    # keep any trailing whitespace of the original number so spacing survives
    trail = re.match(r".*?([\s ]*)$", raw, re.S).group(1)
    return s[:m.start(1)] + txt + trail + s[m.end(1):]


# Tier 1: the edit unambiguously changes who is bound, what is covered, how
# much, or whether an obligation exists. Tier 2: the edit shifts legal scope
# (conjunctive vs disjunctive conditions; genus vs species of vehicle) -- still
# a legal change, but a milder one. Reported separately.
TIER2 = {"connective_and_to_or", "connective_or_to_and", "vehicle_class"}


def tier(rule):
    return 2 if rule in TIER2 else 1


# Elision, contraction and gender-agreement violations. A perturbation that
# trips any of these is discarded: we need metrics to react to the legal change,
# not to a fluency artefact that a language model can spot without reading the
# law. The list is deliberately conservative -- it flags 0/400 of the unmodified
# source sentences, so it costs us only genuine artefacts.
_BAD = re.compile(
    r"\bn['\u2019]\s*n['\u2019]"                                   # double elided negation
    r"|\b(?:le|la|de|ne|que|se|ce|je|me|te|du|au)\s+[aàâeéèêëiîïoôuùû]"  # failed elision
    r"|\bde\s+(?:le|les)\b|\bà\s+(?:le|les)\b"                    # missing contraction
    r"|\b(?:un|le|du|au|cet?)\s+automobile\b"                     # automobile is feminine
    r"|\b(?:une|la|cette)\s+v[eé]hicule\b"                        # véhicule is masculine
    r"|\s{2,}",
    re.UNICODE | re.IGNORECASE)


def _wellformed(s):
    """Reject perturbations whose only obvious defect is grammatical -- we want
    metrics to react to the legal change, not to a fluency artefact."""
    return _BAD.search(s) is None


def legal_edits(sentences, seed=0, per_sentence=1):
    """Return list of (original, perturbed, rule). Tier-1 rules are preferred."""
    rng = random.Random(seed)
    out = []
    for s in sentences:
        cands = []
        for name, subs in _RULES:
            for pat, rep in subs:
                if re.search(pat, s, flags=re.IGNORECASE):
                    new = re.sub(pat, rep, s, count=1, flags=re.IGNORECASE)
                    if new != s:
                        cands.append((new, name))
                    break
        a = _amount_edit(s)
        if a:
            cands.append((a, "amount_x10"))
        cands = [(n, r) for n, r in cands if _wellformed(n)]
        if not cands:
            continue
        rng.shuffle(cands)
        cands.sort(key=lambda c: tier(c[1]))   # stable: tier-1 first
        for new, name in cands[:per_sentence]:
            out.append((s, new, name))
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load_pairs
    P = load_pairs()
    ed = legal_edits([p["original"] for p in P], per_sentence=1)
    print("legal edits:", len(ed), "/", len(P))
    import collections
    print(collections.Counter(r for _, _, r in ed).most_common())
    for o, n, r in ed[:6]:
        print(f"\n[{r}]\n  - {o[:150]}\n  + {n[:150]}")
    u = unrelated_pairs(20)
    print("\nunrelated sample:", len(u))
    print("  A:", u[0][0][:120], "\n  B:", u[0][1][:120])
