"""Experiment 4: LEGALEDIT -- do metrics notice a minimal edit that flips the law?

Three probe sets over the same sentence pool, so surface similarity is held
roughly constant while legal meaning varies:

  identical   (s, s)                  legal meaning preserved      target high
  legaledit   (s, minimal_legal_edit) legal meaning FLIPPED        target low
  unrelated   (s, other legal sent.)  no shared meaning            target low

A metric that scores `legaledit` as high as `identical` is measuring lexical
overlap, not legal meaning -- regardless of how well it correlates with human
judgment on ordinary simplifications.
"""
import json, os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_pairs
from augment import legal_edits, unrelated_pairs, tier, _sents
import stats_utils as su

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)

# probe pool: held-out statutory sentences, disjoint from FrJUDGE
ref = _sents("A-25.jsonl") + _sents("code_securite_routiere.jsonl")
rng = np.random.default_rng(7)
pool = [ref[i] for i in rng.choice(len(ref), size=400, replace=False)]
ED = legal_edits(pool, seed=7, per_sentence=1)
print(f"probe pool={len(pool)} legal edits={len(ED)} "
      f"(tier1={sum(tier(r)==1 for _,_,r in ED)})")

ident = [dict(original=o, simplification=o) for o, _, _ in ED]
edit = [dict(original=o, simplification=p) for o, p, _ in ED]
unrel_raw = unrelated_pairs(len(ED), seed=77)
unrel = [dict(original=a, simplification=b) for a, b in unrel_raw]
T1 = np.array([tier(r) == 1 for _, _, r in ED])
RULES = [r for _, _, r in ED]

ALL = ident + edit + unrel
n = len(ED)


def split3(v):
    v = np.asarray(v, float)
    return v[:n], v[n:2 * n], v[2 * n:]


def run_metrics():
    import metrics_zoo as M
    S = {}
    S["BERTScore-CamemBERTv2"] = M.bertscore_f1(ALL, "almanach/camembertv2-base")
    print("  camembert done", flush=True)
    S["BERTScore-CamemBERTv2-Recall"] = M.bertscore_recall(ALL, "almanach/camembertv2-base")
    S["BERTScore-FlauBERT"] = M.bertscore_f1(ALL, "flaubert/flaubert_base_cased")
    print("  flaubert done", flush=True)
    for nm, hf in [("LaBSE", "sentence-transformers/LaBSE"),
                   ("Sentence-CamemBERT", "dangvantuan/sentence-camembert-base"),
                   ("Para-mUSE-mpnet", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")]:
        S[nm] = M.sbert_cosine(ALL, hf); print("  %s done" % nm, flush=True)
    S["mE5-base"] = M.sbert_cosine(ALL, "intfloat/multilingual-e5-base", prefix="query: ")
    fwd, bwd, mn = M.nli_bidirectional(ALL)
    S["NLI-fwd (no hallucination)"] = fwd
    S["NLI-bwd (no omission)"] = bwd
    S["NLI-bidirectional (min)"] = mn
    print("  nli done", flush=True)
    return S


CACHE = os.path.join(OUT, "exp4_raw.pkl")
if os.path.exists(CACHE):
    S = pickle.load(open(CACHE, "rb"))
else:
    S = run_metrics()
    pickle.dump(S, open(CACHE, "wb"))

# surface baselines on the same probes
import features as F
X = np.array([F.featurize(d["original"], d["simplification"]) for d in ALL])
S["surface: token Jaccard"] = X[:, F.FEATURE_NAMES.index("jaccard")]
S["surface: -|len diff|"] = -X[:, F.FEATURE_NAMES.index("abs_len_diff")]

rows = {}
for name, v in S.items():
    i_, e_, u_ = split3(v)
    lo, hi = float(np.min(v)), float(np.max(v))
    nz = lambda a: (a - lo) / (hi - lo + 1e-12)           # 0-1 for comparability
    rows[name] = dict(
        identical=float(nz(i_).mean()), legaledit=float(nz(e_).mean()),
        legaledit_t1=float(nz(e_[T1]).mean()), unrelated=float(nz(u_).mean()),
        # discrimination: does the metric rank identical strictly above the edit?
        discrim=float((i_ > e_).mean()),
        # margin as a fraction of the identical-vs-unrelated dynamic range
        margin_frac=float((i_ - e_).mean() / max(1e-9, (i_ - u_).mean())),
        auc_edit_vs_ident=float(su.pearson((i_ > e_).astype(float), np.ones(n))
                                if False else np.nan),
    )
    d, l, h, p = su.boot_diff_test(lambda a, b: float(np.mean(a)), i_, e_, np.zeros(n))
    rows[name]["ident_minus_edit"] = d
    rows[name]["ident_minus_edit_ci"] = [l, h]

json.dump(dict(rows=rows, n=n, rules=RULES, tier1=T1.tolist()),
          open(os.path.join(OUT, "exp4_challenge.json"), "w"), indent=2, default=float)

print(f"\n{'metric':32s} {'ident':>6s} {'edit':>6s} {'edit_t1':>7s} {'unrel':>6s} | "
      f"{'discrim%':>8s} {'margin_frac':>11s}")
print("-" * 92)
for k in sorted(rows, key=lambda k: -rows[k]["margin_frac"]):
    r = rows[k]
    print(f"{k:32s} {r['identical']:6.3f} {r['legaledit']:6.3f} {r['legaledit_t1']:7.3f} "
          f"{r['unrelated']:6.3f} | {r['discrim']*100:7.1f}% {r['margin_frac']:11.3f}")
