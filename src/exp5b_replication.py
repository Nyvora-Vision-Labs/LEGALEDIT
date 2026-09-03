"""Experiment 5b: independent replication of the prompted-judge run.

The judge is sampled, not deterministic, so every number in Experiment 5 is a
draw. Replaying the cache would reproduce those numbers by construction and
establish nothing; this script draws a completely fresh set of K samples under a
separate cache tag, identical in model, prompt version, temperature and K, and
reports run 1 against run 2 side by side.

Everything else in the pipeline -- splits, the metric zoo, the probe
construction -- is deterministic and is reused unchanged, so any difference
between the two columns is judge sampling variance and nothing else.
"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_pairs, ANNOTATORS
from splits import make_split, SEEDS
from augment import legal_edits, unrelated_pairs, tier, _sents
import stats_utils as su
import llm_judge as J
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results")
K = J.K
REP = os.environ.get("REP_TAG", "rep2")
A = json.load(open(os.path.join(OUT, "exp5_llm_judge.json")))   # run 1

P = load_pairs()
y = np.array([p["label"] for p in P])
H = np.array([[p["ratings"][a] for a in ANNOTATORS] for p in P], float)

ref = _sents("A-25.jsonl") + _sents("code_securite_routiere.jsonl")
rng = np.random.default_rng(7)
pool = [ref[i] for i in rng.choice(len(ref), size=400, replace=False)]
ED = legal_edits(pool, seed=7, per_sentence=1)
n = len(ED)
ALL = ([dict(original=o, simplification=o) for o, _, _ in ED]
       + [dict(original=o, simplification=p) for o, p, _ in ED]
       + [dict(original=a, simplification=b) for a, b in unrelated_pairs(n, seed=77)])
RULES = [r for _, _, r in ED]
T1 = np.array([tier(r) == 1 for r in RULES])

print("drawing an independent K=%d sample for %d corpus pairs and %d probes ..."
      % (K, len(P), len(ALL)), flush=True)
M1 = J.score_matrix(P, k=K, tag="frjudge")                 # run 1 (cached)
M2 = J.score_matrix(P, k=K, tag="frjudge_" + REP)          # run 2 (fresh)
E1 = J.score_matrix(ALL, k=K, tag="exp4")
E2 = J.score_matrix(ALL, k=K, tag="exp4_" + REP)

def summarise(M, E):
    """Every headline statistic, from one run's raw sample matrices."""
    with np.errstate(invalid="ignore"):
        jm = np.nanmean(M, axis=1); ve = np.nanmean(E, axis=1)
    ok = np.isfinite(jm)
    d = {}
    d["judge_mean"] = float(np.nanmean(jm))
    d["judge_sd"] = float(np.nanstd(jm))
    d["mean_judge_human_r"] = float(np.mean(
        [su.pearson(jm[ok], H[ok, i]) for i in range(len(ANNOTATORS))]))
    d["alpha_self"] = float(su.krippendorff_ordinal(M))
    d["r_full"] = float(su.pearson(jm[ok], y[ok]))
    rc, rr = [], []
    for sd in SEEDS:
        tr, va, te = make_split(len(P), sd)
        fit = np.concatenate([tr, va])
        f = fit[np.isfinite(jm[fit])]; t = te[np.isfinite(jm[te])]
        rr.append(su.pearson(jm[t], y[t]))
        iso = IsotonicRegression(y_min=1, y_max=10, out_of_bounds="clip").fit(jm[f], y[f])
        rc.append(su.pearson(iso.predict(jm[t]), y[t]))
    d["r_raw_mean"] = float(np.mean(rr)); d["r_cal_mean"] = float(np.mean(rc))
    i_, e_, u_ = ve[:n], ve[n:2 * n], ve[2 * n:]
    d["identical"] = float(i_.mean()); d["legaledit"] = float(e_.mean())
    d["legaledit_t1"] = float(e_[T1].mean()); d["unrelated"] = float(u_.mean())
    d["margin_frac"] = float((i_ - e_).mean() / max(1e-9, (i_ - u_).mean()))
    d["margin_frac_t1"] = float((i_[T1] - e_[T1]).mean() / max(1e-9, (i_[T1] - u_[T1]).mean()))
    d["discrim"] = float((i_ > e_).mean())
    by = collections.defaultdict(list)
    for r, s in zip(RULES, e_):
        by[r].append(s)
    d["_rules"] = {r: float(np.mean(v)) for r, v in by.items()}
    d["_jm"] = jm; d["_ve"] = ve
    return d

S1, S2 = summarise(M1, E1), summarise(M2, E2)

FIELDS = [
    ("judge_mean",         "mean rating on FrJudge",        "%.2f",  0.15),
    ("judge_sd",           "sd of that rating",             "%.2f",  0.15),
    ("mean_judge_human_r", "mean r with the 5 annotators",  "%.3f",  0.03),
    ("alpha_self",         "ordinal alpha among K samples", "%.3f",  0.05),
    ("r_full",             "r vs human mean, full corpus",  "%.3f",  0.03),
    ("r_raw_mean",         "r raw, mean over 10 splits",    "%.3f",  0.03),
    ("r_cal_mean",         "r calibrated, over 10 splits",  "%.3f",  0.03),
    ("identical",          "LegalEdit: identical probes",   "%.2f",  0.10),
    ("legaledit",          "LegalEdit: edited probes",      "%.2f",  0.30),
    ("legaledit_t1",       "LegalEdit: edited, tier 1",     "%.2f",  0.30),
    ("unrelated",          "LegalEdit: unrelated probes",   "%.2f",  0.15),
    ("margin_frac",        "margin fraction",               "%.3f",  0.04),
    ("margin_frac_t1",     "margin fraction, tier 1",       "%.3f",  0.04),
    ("discrim",            "discrimination rate",           "%.3f",  0.04),
]

print("\n" + "=" * 84)
print("REPLICATION: independent K=%d draw, same model / prompt / temperature" % K)
print("=" * 84)
print("\n%-34s %8s %8s %9s   %s"
      % ("statistic", "run 1", "run 2", "delta", "within tolerance"))
print("-" * 84)
rep = {}
allok = True
for key, label, fmt, tol in FIELDS:
    a, b = S1[key], S2[key]
    ok = abs(a - b) <= tol
    allok &= ok
    print(("%-34s " + fmt + " " + fmt + "   %+8.3f   %s")
          % (label, a, b, b - a, "yes" if ok else "NO  (tol %.2f)" % tol))
    rep[key] = dict(run1=a, run2=b, delta=b - a, tol=tol, ok=bool(ok))

# item-level stability: do the two runs order the same items the same way?
ok1 = np.isfinite(S1["_jm"]) & np.isfinite(S2["_jm"])
r_items = float(su.pearson(S1["_jm"][ok1], S2["_jm"][ok1]))
rho_items = float(su.spearman(S1["_jm"][ok1], S2["_jm"][ok1]))
r_probe = float(su.pearson(S1["_ve"], S2["_ve"]))
print("\nitem-level agreement between the two runs:")
print("   FrJudge pairs   r = %.3f   rho = %.3f   (n=%d)" % (r_items, rho_items, ok1.sum()))
print("   LegalEdit probes r = %.3f                (n=%d)" % (r_probe, len(S1["_ve"])))
rep["item_stability"] = dict(frjudge_r=r_items, frjudge_rho=rho_items, probe_r=r_probe)

# the finding that mattered: the directional modal asymmetry
print("\ndirectional asymmetry (mean judge score on the edited clause):")
for r in ("modal_obligation_to_permission", "modal_permission_to_obligation",
          "negation_drop", "scope_all_to_some"):
    print("   %-34s %5.2f -> %5.2f" % (r, S1["_rules"][r], S2["_rules"][r]))
gap1 = S1["_rules"]["modal_permission_to_obligation"] - S1["_rules"]["modal_obligation_to_permission"]
gap2 = S2["_rules"]["modal_permission_to_obligation"] - S2["_rules"]["modal_obligation_to_permission"]
print("   asymmetry gap (peut->doit minus doit->peut):  %.2f -> %.2f" % (gap1, gap2))
rep["asymmetry_gap"] = dict(run1=gap1, run2=gap2)
rep["by_rule_run2"] = S2["_rules"]

# the headline: does the rank inversion survive?
B = json.load(open(os.path.join(OUT, "exp2_baselines.json")))
C = json.load(open(os.path.join(OUT, "exp4_challenge.json")))["rows"]
rows = []
for k in C:
    if k in B and not k.startswith("surface:") and not k.startswith("LLM-judge"):
        rows.append((k, float(np.mean([d["r"] for d in B[k]["per_seed_cal"]])), C[k]["margin_frac"]))
rows.append(("LLM-judge (run 2)", S2["r_cal_mean"], S2["margin_frac"]))
rk = sorted(rows, key=lambda t: -t[1]); rm = sorted(rows, key=lambda t: -t[2])
jr = [i for i, t in enumerate(rk) if t[0].startswith("LLM-judge")][0] + 1
jm_ = [i for i, t in enumerate(rm) if t[0].startswith("LLM-judge")][0] + 1
nr = [i for i, t in enumerate(rk) if t[0] == "NLI-bidirectional (min)"][0] + 1
nm = [i for i, t in enumerate(rm) if t[0] == "NLI-bidirectional (min)"][0] + 1
print("\nrank inversion (run 2 judge against the unchanged deterministic metrics):")
print("   judge              rank by r = %d,  rank by margin = %d   (run 1: 1 and 4)" % (jr, jm_))
print("   NLI-bidirectional  rank by r = %d,  rank by margin = %d   (run 1: 4 and 1)" % (nr, nm))
inv = (jr < nr) and (jm_ > nm)
print("   inversion holds:   %s" % ("YES" if inv else "NO"))
rep["ranks"] = dict(judge_r=jr, judge_margin=jm_, nli_r=nr, nli_margin=nm, inversion=bool(inv))

u = J.usage_summary(tag="frjudge_" + REP); v = J.usage_summary(tag="exp4_" + REP)
tot = {k: u.get(k, 0) + v.get(k, 0) for k in set(u) | set(v)}
cost = (tot.get("prompt_tokens", 0) - tot.get("prompt_cache_hit_tokens", 0)) / 1e6 * 0.27 \
     + tot.get("prompt_cache_hit_tokens", 0) / 1e6 * 0.07 \
     + tot.get("completion_tokens", 0) / 1e6 * 1.10
print("\n" + "=" * 84)
print("VERDICT: %s" % ("all %d statistics within tolerance; the run replicates." % len(FIELDS)
                       if allok else "at least one statistic moved outside tolerance -- see NO rows above."))
print("replication cost: %d calls, ~$%.2f" % (tot.get("calls", 0), cost))
print("=" * 84)
rep["all_within_tolerance"] = bool(allok)
rep["usage"] = dict(total=tot, est_usd=round(cost, 2))
json.dump(rep, open(os.path.join(OUT, "exp5b_replication.json"), "w"), indent=2, default=float)
print("\nwrote results/exp5b_replication.json")
