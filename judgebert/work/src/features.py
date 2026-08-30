"""Trivial surface features -- the baseline the original paper omits."""
import re
import numpy as np

_W = re.compile(r"\w+", re.UNICODE)
_NUM = re.compile(r"\d[\d\s.,]*")
# tokens whose presence/absence flips legal force in French insurance text
_MODAL = {"doit", "doivent", "peut", "peuvent", "devra", "pourra", "faut"}
_NEG = {"ne", "pas", "non", "aucun", "aucune", "ni", "sauf", "sans", "exclu", "exclut", "exclus"}
_CONN = {"et", "ou"}

def toks(s):
    return _W.findall(s.lower())

FEATURE_NAMES = [
    "len_ratio", "abs_len_diff", "jaccard", "orig_recall", "simp_precision",
    "num_overlap", "modal_diff", "neg_diff", "conn_diff", "char_len_ratio",
]

def featurize(orig, simp):
    O, S = toks(orig), toks(simp)
    A, B = set(O), set(S)
    uni = len(A | B) or 1
    numO, numS = set(_NUM.findall(orig)), set(_NUM.findall(simp))
    numu = len(numO | numS) or 1
    cnt = lambda X, V: sum(1 for t in X if t in V)
    return np.array([
        len(S) / max(1, len(O)),
        abs(len(S) - len(O)),
        len(A & B) / uni,
        len(A & B) / max(1, len(A)),
        len(A & B) / max(1, len(B)),
        len(numO & numS) / numu,
        cnt(S, _MODAL) - cnt(O, _MODAL),
        cnt(S, _NEG) - cnt(O, _NEG),
        cnt(S, _CONN) - cnt(O, _CONN),
        len(simp) / max(1, len(orig)),
    ], float)

def matrix(pairs):
    return np.array([featurize(p["original"], p["simplification"]) for p in pairs])
