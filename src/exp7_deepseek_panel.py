"""Experiment 7: how much of the judge result is the model, and how much the prompt?

Experiment 5 used one model under one rubric wording. Two objections follow,
and neither is answered by drawing more samples -- the replication already
settled sampling variance, which is a different axis from both of these:

  A. Capability. Was a mid-tier chat model asked to do a reasoning model's job?
     `deepseek-reasoner` runs the identical rubric on the identical items.
  B. Prompt. How much of the behaviour is the rubric rather than the model?
     Three wordings on the same model: the original French rubric (v1), the
     same rubric in English over the same French text (v2), and a bare French
     instruction with the four-type error taxonomy stripped out (v3).

Both run on a fixed 100-edit subsample so the reasoner stays affordable -- it
emits ~1,770 completion tokens per call, all billed, making it about 15x the
cost of a chat call. The existing chat run is recomputed on exactly those items
from cache at no cost, so every column is like-for-like.
"""
import json, os, sys, collections
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
N_EDITS, SUB_SEED = 100, 11
K_REASON, K_PROMPT = 1, 2

P = load_pairs()
y = np.array([p["label"] for p in P])
ref = _sents("A-25.jsonl") + _sents("code_securite_routiere.jsonl")
rng = np.random.default_rng(7)
pool = [ref[i] for i in rng.choice(len(ref), size=400, replace=False)]
ED_FULL = legal_edits(pool, seed=7, per_sentence=1)
UN_FULL = unrelated_pairs(len(ED_FULL), seed=77)
NF = len(ED_FULL)
sub = np.sort(np.random.default_rng(SUB_SEED).choice(NF, size=N_EDITS, replace=False))
ED = [ED_FULL[i] for i in sub]
n = len(ED)
PROBES = ([dict(original=o, simplification=o) for o, _, _ in ED]
          + [dict(original=o, simplification=p) for o, p, _ in ED]
          + [dict(original=a, simplification=b) for a, b in [UN_FULL[i] for i in sub]])
RULES = [r for _, _, r in ED]


def stats(jc, jp):
    d = {}
    ok = np.isfinite(jc)
    rc = []
    for sd in SEEDS:
        tr, va, te = make_split(len(P), sd)
        fit = np.concatenate([tr, va])
        f = fit[np.isfinite(jc[fit])]; t = te[np.isfinite(jc[te])]
        m = IsotonicRegression(y_min=1, y_max=10, out_of_bounds="clip").fit(jc[f], y[f])
        rc.append(su.pearson(m.predict(jc[t]), y[t]))
    d["r_cal_mean"] = float(np.mean(rc)); d["r_cal_sd"] = float(np.std(rc))
    d["r_full"] = float(su.pearson(jc[ok], y[ok]))
    d["mean_rating"] = float(np.nanmean(jc))
    i_, e_, u_ = jp[:n], jp[n:2 * n], jp[2 * n:]
    d["identical"] = float(np.nanmean(i_)); d["legaledit"] = float(np.nanmean(e_))
    d["unrelated"] = float(np.nanmean(u_))
    d["margin_frac"] = float((np.nanmean(i_) - np.nanmean(e_))
                             / max(1e-9, np.nanmean(i_) - np.nanmean(u_)))
    d["discrim"] = float(np.nanmean(i_ > e_))
    by = collections.defaultdict(list)
    for r, sc in zip(RULES, e_):
        by[r].append(sc)
    d["_rules"] = {r: float(np.nanmean(v)) for r, v in by.items()}
    d["asym"] = (d["_rules"].get("modal_permission_to_obligation", np.nan)
                 - d["_rules"].get("modal_obligation_to_permission", np.nan))
    return d


def chat_cached(k, prompt="v1"):
    """The existing chat run restricted to the subsample -- costs nothing.

    Corpus scores and probe scores are both drawn at the requested K, from the
    same cached samples, so a row of the table is internally like-for-like.
    Reading the K=5 mean that experiment 5 saved instead would print a
    five-sample diagnostic beside a one- or two-sample correlation, label it
    K=1, and turn the prompt comparison below into a comparison of K as much as
    of wording: v2 and v3 score their own probes at K_PROMPT.

    tag="exp4" is experiment 5's own cache file, which already holds samples
    0..4 of every v1 probe, so at k <= 5 this is served entirely from disk and
    issues no request.
    """
    with np.errstate(invalid="ignore"):
        c = np.nanmean(J.score_matrix(P, k=k, tag="frjudge", prompt=prompt,
                                      verbose=False)[:, :k], axis=1)
        p_ = np.nanmean(J.score_matrix(PROBES, k=k, tag="exp4", prompt=prompt,
                                       verbose=False), axis=1)
    return c, p_


print("=" * 90)
print("EXPERIMENT 7: model capability and prompt sensitivity")
print("=" * 90)
print("items: %d corpus pairs + %d probes (%d edits, seed %d)"
      % (len(P), len(PROBES), n, SUB_SEED))
print("budget ceiling $%.2f, spent so far $%.2f" % (J.BUDGET_USD, J.spend()))

R = {}
# ------------------------------------------------------- A. chat vs reasoner --
print("\n--- A. capability: deepseek-chat vs deepseek-reasoner (K=%d) ---" % K_REASON)
cc, cp = chat_cached(K_REASON)
R["chat (K=1)"] = stats(cc, cp)
print("  chat: reused from cache ($0.00)")
try:
    with np.errstate(invalid="ignore"):
        rc_ = np.nanmean(J.score_matrix(P, k=K_REASON, model="deepseek-reasoner",
                                        tag="panel_corpus"), axis=1)
        rp_ = np.nanmean(J.score_matrix(PROBES, k=K_REASON, model="deepseek-reasoner",
                                        tag="panel_probe"), axis=1)
    R["reasoner (K=1)"] = stats(rc_, rp_)
    print("  reasoner done; running spend $%.2f" % J.spend())
except J.BudgetExceeded as e:
    print("  ! %s" % e)

# ------------------------------------------------------ B. prompt sensitivity --
print("\n--- B. prompt sensitivity on deepseek-chat (K=%d) ---" % K_PROMPT)
cc2, cp2 = chat_cached(K_PROMPT, "v1")
R["chat v1 (rubric, FR)"] = stats(cc2, cp2)
for var, lab in [("v2", "chat v2 (rubric, EN)"), ("v3", "chat v3 (no taxonomy)")]:
    try:
        with np.errstate(invalid="ignore"):
            c = np.nanmean(J.score_matrix(P, k=K_PROMPT, tag="prompt_corpus",
                                          prompt=var), axis=1)
            p_ = np.nanmean(J.score_matrix(PROBES, k=K_PROMPT, tag="prompt_probe",
                                           prompt=var), axis=1)
        R[lab] = stats(c, p_)
        print("  %s done; running spend $%.2f" % (var, J.spend()))
    except J.BudgetExceeded as e:
        print("  ! %s" % e); break

# ------------------------------------------------------------------- report --
print("\n" + "=" * 90)
print("%-24s %13s %7s %8s %8s %7s %7s"
      % ("configuration", "r_cal", "mean", "margin", "discrim", "edit", "asym"))
print("-" * 90)
for k in R:
    v = R[k]
    print("%-24s %6.3f+-%.3f %7.2f %8.3f %7.1f%% %7.2f %7.2f"
          % (k, v["r_cal_mean"], v["r_cal_sd"], v["mean_rating"], v["margin_frac"],
             v["discrim"] * 100, v["legaledit"], v["asym"]))

print("\ndirectional asymmetry across every configuration:")
for k, v in R.items():
    r = v["_rules"]
    print("   %-24s doit->peut %5.2f   peut->doit %5.2f   gap %+.2f"
          % (k, r.get("modal_obligation_to_permission", float("nan")),
             r.get("modal_permission_to_obligation", float("nan")), v["asym"]))

pv = [k for k in R if k.startswith("chat v")]
if len(pv) > 1:
    rs = [R[k]["r_cal_mean"] for k in pv]; ms = [R[k]["margin_frac"] for k in pv]
    print("\nspread across the %d prompt wordings:" % len(pv))
    print("   r_cal   %.3f - %.3f  (range %.3f)" % (min(rs), max(rs), max(rs) - min(rs)))
    print("   margin  %.3f - %.3f  (range %.3f)" % (min(ms), max(ms), max(ms) - min(ms)))
    print("   for scale: run-to-run sampling moved r_cal by 0.029 and margin by 0.002")

print("\ntotal estimated spend this run: $%.2f (ceiling $%.2f)" % (J.spend(), J.BUDGET_USD))
json.dump(dict(configs={k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                       for k, v in R.items()},
               rules={k: v["_rules"] for k, v in R.items()},
               n_edits=n, subsample_seed=SUB_SEED, spend=J.spend()),
          open(os.path.join(OUT, "exp7_panel.json"), "w"), indent=2, default=float)
print("wrote results/exp7_panel.json")
