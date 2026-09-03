"""Experiment 6: can one scorer hold both ranks?

Experiment 5 leaves the field split. The prompted judge is first of eleven on
agreement with human ratings and fourth on the LEGALEDIT diagnostic;
bidirectional NLI is fourth and first. The two are not merely different in
degree -- they fail in opposite directions, which is what makes combining them
worth trying rather than merely convenient.

The construction has to survive the objection that it is fitted. Two rules
therefore hold throughout:

  1. Every calibration map is fitted on the training split only, per seed, and
     evaluated on the held-out split -- the same discipline exp2 applies to
     every unsupervised metric.
  2. The diagnostic sees no labels at all. Whatever rule the corpus selects is
     transferred unchanged to the probes, exactly as it would be in deployment.
     A combination that only looks good after peeking at LEGALEDIT would be
     worthless, so LEGALEDIT never enters the fit.

Both components are first mapped to the 1-10 human scale by isotonic
regression, which is monotone and so cannot change either parent's ranking.
Combination happens on that shared scale. Parents are re-reported through the
identical pipeline, so the hybrid is never compared against a differently
processed baseline.
"""
import json, os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_pairs
from splits import make_split, SEEDS
from augment import legal_edits, unrelated_pairs, tier, _sents
import stats_utils as su
import llm_judge as J
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results")
NLI_KEY = "NLI-bidirectional (min)"

# ------------------------------------------------------------------ inputs --
P = load_pairs()
y = np.array([p["label"] for p in P])
S_corpus = pickle.load(open(os.path.join(OUT, "raw_metric_scores.pkl"), "rb"))
nli_c = np.asarray(S_corpus[NLI_KEY], float)

def judge_corpus(tag):
    with np.errstate(invalid="ignore"):
        v = np.nanmean(J.score_matrix(P, tag=tag, verbose=False), axis=1)
    return np.where(np.isfinite(v), v, np.nanmean(v))

ref = _sents("A-25.jsonl") + _sents("code_securite_routiere.jsonl")
rng = np.random.default_rng(7)
pool = [ref[i] for i in rng.choice(len(ref), size=400, replace=False)]
ED = legal_edits(pool, seed=7, per_sentence=1)
n = len(ED)
ALL = ([dict(original=o, simplification=o) for o, _, _ in ED]
       + [dict(original=o, simplification=p) for o, p, _ in ED]
       + [dict(original=a, simplification=b) for a, b in unrelated_pairs(n, seed=77)])
T1 = np.array([tier(r) == 1 for _, _, r in ED])
S_probe = pickle.load(open(os.path.join(OUT, "exp4_raw.pkl"), "rb"))
nli_p = np.asarray(S_probe[NLI_KEY], float)

def judge_probe(tag, npy=None):
    if npy and os.path.exists(npy):
        return np.load(npy)
    with np.errstate(invalid="ignore"):
        return np.nanmean(J.score_matrix(ALL, tag=tag, verbose=False), axis=1)

# ------------------------------------------------------------------- rules --
# Each takes two vectors already on the 1-10 scale and returns one.
# `min` is the conjunctive reading of the construct: a clause is meaning-
# preserving only if BOTH readings say so, so either component can veto.
RULES = {
    "min(judge, NLI)":        lambda a, b: np.minimum(a, b),
    "mean(judge, NLI)":       lambda a, b: (a + b) / 2,
    "geometric mean":         lambda a, b: np.sqrt(np.clip(a, 1e-9, None) * np.clip(b, 1e-9, None)),
    "harmonic mean":          lambda a, b: 2 * a * b / np.clip(a + b, 1e-9, None),
    "max(judge, NLI)":        lambda a, b: np.maximum(a, b),
}
WGRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def iso(x_fit, y_fit, x_apply):
    m = IsotonicRegression(y_min=1, y_max=10, out_of_bounds="clip").fit(x_fit, y_fit)
    return m.predict(x_apply), m


def evaluate(judge_c, judge_p, label):
    """Corpus correlation per seed + the diagnostic, for parents and every rule."""
    names = list(RULES) + ["weighted w*judge+(1-w)*NLI"]
    acc = {k: dict(r=[], rmse=[]) for k in names + ["judge (calibrated)", "NLI (calibrated)"]}
    ws = []
    for sd in SEEDS:
        tr, va, te = make_split(len(P), sd)
        fit = np.concatenate([tr, va])
        cj, _ = iso(judge_c[fit], y[fit], judge_c)
        cn, _ = iso(nli_c[fit], y[fit], nli_c)
        acc["judge (calibrated)"]["r"].append(su.pearson(cj[te], y[te]))
        acc["judge (calibrated)"]["rmse"].append(su.rmse(cj[te], y[te]))
        acc["NLI (calibrated)"]["r"].append(su.pearson(cn[te], y[te]))
        acc["NLI (calibrated)"]["rmse"].append(su.rmse(cn[te], y[te]))
        for k, f in RULES.items():
            v = f(cj, cn)
            acc[k]["r"].append(su.pearson(v[te], y[te]))
            acc[k]["rmse"].append(su.rmse(v[te], y[te]))
        # the one fitted rule: w chosen on the TRAIN split only
        best = max(WGRID, key=lambda w: su.pearson(w * cj[fit] + (1 - w) * cn[fit], y[fit]))
        ws.append(float(best))
        v = best * cj + (1 - best) * cn
        acc["weighted w*judge+(1-w)*NLI"]["r"].append(su.pearson(v[te], y[te]))
        acc["weighted w*judge+(1-w)*NLI"]["rmse"].append(su.rmse(v[te], y[te]))

    # ---- diagnostic: maps fitted on the whole corpus, no probe labels ever ----
    pj, _ = iso(judge_c, y, judge_p)
    pn, _ = iso(nli_c, y, nli_p)
    wbar = float(np.mean(ws))

    def diag(v):
        i_, e_, u_ = v[:n], v[n:2 * n], v[2 * n:]
        den = max(1e-9, (i_ - u_).mean())
        return dict(identical=float(i_.mean()), legaledit=float(e_.mean()),
                    unrelated=float(u_.mean()),
                    margin_frac=float((i_ - e_).mean() / den),
                    margin_frac_t1=float((i_[T1] - e_[T1]).mean()
                                         / max(1e-9, (i_[T1] - u_[T1]).mean())),
                    discrim=float((i_ > e_).mean()))

    D = {"judge (calibrated)": diag(pj), "NLI (calibrated)": diag(pn)}
    for k, f in RULES.items():
        D[k] = diag(f(pj, pn))
    D["weighted w*judge+(1-w)*NLI"] = diag(wbar * pj + (1 - wbar) * pn)

    rows = {}
    for k in acc:
        rows[k] = dict(r_cal_mean=float(np.mean(acc[k]["r"])),
                       r_cal_sd=float(np.std(acc[k]["r"])),
                       rmse_cal=float(np.mean(acc[k]["rmse"])), **D[k])
    return rows, wbar, ws


print("=" * 96)
print("EXPERIMENT 6: hybrid scorers -- can one metric hold rank 1 on both axes?")
print("=" * 96)

ROWS, WBAR, WS = evaluate(judge_corpus("frjudge"),
                          judge_probe("exp4", os.path.join(OUT, "cache", "llm_judge_exp4.npy")),
                          "run 1")
print("\nfitted weight w chosen per seed on the training split: mean %.2f, range %.2f-%.2f"
      % (WBAR, min(WS), max(WS)))

print("\n%-30s %14s %8s | %8s %8s %8s"
      % ("scorer", "r_cal", "RMSE", "margin", "marg_t1", "discrim"))
print("-" * 96)
order = sorted(ROWS, key=lambda k: -ROWS[k]["r_cal_mean"])
for k in order:
    v = ROWS[k]
    print("%-30s %7.3f+-%.3f %8.2f | %8.3f %8.3f %7.1f%%"
          % (k, v["r_cal_mean"], v["r_cal_sd"], v["rmse_cal"],
             v["margin_frac"], v["margin_frac_t1"], v["discrim"] * 100))

# --------------------------------------------------- does it beat both parents? --
pj, pn = ROWS["judge (calibrated)"], ROWS["NLI (calibrated)"]
print("\nparents (same pipeline):  judge r=%.3f margin=%.3f   NLI r=%.3f margin=%.3f"
      % (pj["r_cal_mean"], pj["margin_frac"], pn["r_cal_mean"], pn["margin_frac"]))
print("\ndominance test -- a hybrid must beat BOTH parents on BOTH axes:")
win = []
for k in ROWS:
    if k.endswith("(calibrated)"):
        continue
    v = ROWS[k]
    br = v["r_cal_mean"] >= max(pj["r_cal_mean"], pn["r_cal_mean"])
    bm = v["margin_frac"] >= max(pj["margin_frac"], pn["margin_frac"])
    flag = "DOMINATES BOTH" if (br and bm) else ("beats on r" if br else
           ("beats on margin" if bm else "beats neither"))
    print("   %-30s r %+.3f vs best parent, margin %+.3f  -> %s"
          % (k, v["r_cal_mean"] - max(pj["r_cal_mean"], pn["r_cal_mean"]),
             v["margin_frac"] - max(pj["margin_frac"], pn["margin_frac"]), flag))
    if br and bm:
        win.append(k)

# ------------------------------------------------- rank against the full field --
B = json.load(open(os.path.join(OUT, "exp2_baselines.json")))
C = json.load(open(os.path.join(OUT, "exp4_challenge.json")))["rows"]
field = [(k, float(np.mean([d["r"] for d in B[k]["per_seed_cal"]])), C[k]["margin_frac"])
         for k in C if k in B and not k.startswith("surface:")]
best_rule = max((k for k in ROWS if not k.endswith("(calibrated)")),
                key=lambda k: min(ROWS[k]["r_cal_mean"] / max(pj["r_cal_mean"], 1e-9),
                                  ROWS[k]["margin_frac"] / max(pn["margin_frac"], 1e-9)))
cand = field + [("HYBRID: " + best_rule, ROWS[best_rule]["r_cal_mean"],
                 ROWS[best_rule]["margin_frac"])]
rk = sorted(cand, key=lambda t: -t[1]); rm = sorted(cand, key=lambda t: -t[2])
hr = [i for i, t in enumerate(rk) if t[0].startswith("HYBRID")][0] + 1
hm = [i for i, t in enumerate(rm) if t[0].startswith("HYBRID")][0] + 1
print("\nbest-balanced rule '%s' placed in the field of %d scorers:" % (best_rule, len(cand)))
print("   rank by corpus r = %d of %d,   rank by margin = %d of %d" % (hr, len(cand), hm, len(cand)))

# ------------------------------------------------------ replication of the hybrid --
print("\n" + "-" * 96)
print("stability: the same construction on the independent run-2 judge draw")
R2, W2, _ = evaluate(judge_corpus("frjudge_rep2"), judge_probe("exp4_rep2"), "run 2")
print("%-30s %16s %16s" % ("scorer", "r_cal (1 -> 2)", "margin (1 -> 2)"))
for k in order:
    print("%-30s   %.3f -> %.3f      %.3f -> %.3f"
          % (k, ROWS[k]["r_cal_mean"], R2[k]["r_cal_mean"],
             ROWS[k]["margin_frac"], R2[k]["margin_frac"]))

json.dump(dict(run1=ROWS, run2=R2, w_mean=WBAR, w_per_seed=WS,
               dominating_rules=win, best_balanced=best_rule,
               hybrid_rank_r=hr, hybrid_rank_margin=hm, field_size=len(cand)),
          open(os.path.join(OUT, "exp6_hybrid.json"), "w"), indent=2, default=float)
print("\nwrote results/exp6_hybrid.json")
