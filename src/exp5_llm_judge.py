"""Experiment 5: the prompted LLM judge, analysed as a scorer and as an annotator.

Experiments 2 and 4 place the judge in the metric tables beside every other
scorer. This script asks the three questions those tables cannot answer:

  A. Panel position. Treated as a sixth annotator, where does the judge sit
     relative to the five humans -- how strict is it, how well does it agree
     with each of them, and what does it do to the panel's Krippendorff alpha?
  B. Self-consistency. A sampled judge has its own variance. We measure the
     ordinal alpha *among its own K samples* -- directly comparable to the
     humans' alpha on the same scale -- and how much averaging K samples buys.
  C. Diagnostic behaviour. On LEGALEDIT, the raw 1-10 output is interpretable
     in a way a cosine is not, so we report the actual scores it assigns to
     identical, edited and unrelated probes, broken out by tier and by rule,
     plus what the judge *says* is wrong -- whether it names the error type the
     perturbation actually introduced.
"""
import json, os, sys, unicodedata, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_pairs, ANNOTATORS
from splits import make_split, SEEDS
from augment import legal_edits, unrelated_pairs, tier, _sents
import stats_utils as su
import llm_judge as J

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results")
K = J.K
R = {}

# ============================================================== A. the corpus ==
P = load_pairs()
y = np.array([p["label"] for p in P])
H = np.array([[p["ratings"][a] for a in ANNOTATORS] for p in P], float)  # items x 5
M = J.score_matrix(P, k=K, tag="frjudge")          # items x K, all cached by now
jm = np.nanmean(M, axis=1)

print("=" * 78)
print("A. THE JUDGE AS A SIXTH ANNOTATOR   (n=%d pairs, K=%d samples)" % (len(P), K))
print("=" * 78)

means = {a: float(H[:, i].mean()) for i, a in enumerate(ANNOTATORS)}
means["JUDGE"] = float(jm.mean())
sds = {a: float(H[:, i].std()) for i, a in enumerate(ANNOTATORS)}
sds["JUDGE"] = float(jm.std())
print("\nmean rating (leniency):")
for k in sorted(means, key=lambda k: -means[k]):
    print("   %-6s %5.2f   sd %4.2f" % (k, means[k], sds[k]))

# agreement of the judge with each human, against the human-human baseline
jr = {a: su.pearson(jm, H[:, i]) for i, a in enumerate(ANNOTATORS)}
hh = [su.pearson(H[:, i], H[:, j]) for i in range(5) for j in range(i + 1, 5)]
print("\nPearson r with each annotator:")
for a in ANNOTATORS:
    print("   judge-%s   %.3f" % (a, jr[a]))
print("   %-9s %.3f   (mean judge-human)" % ("", float(np.mean(list(jr.values())))))
print("   %-9s %.3f   (mean human-human, for comparison)" % ("", float(np.mean(hh))))

a5 = su.krippendorff_ordinal(H)
a6 = su.krippendorff_ordinal(np.hstack([H, np.round(jm)[:, None]]))
print("\nordinal Krippendorff alpha:")
print("   5 humans           %.3f" % a5)
print("   5 humans + judge   %.3f" % a6)

R["panel"] = dict(mean_rating=means, sd_rating=sds, judge_vs_annotator=jr,
                  mean_judge_human_r=float(np.mean(list(jr.values()))),
                  mean_human_human_r=float(np.mean(hh)),
                  alpha_5=float(a5), alpha_6=float(a6))

# ====================================================== B. self-consistency ====
print("\n" + "=" * 78)
print("B. SELF-CONSISTENCY OF THE JUDGE")
print("=" * 78)

a_self = su.krippendorff_ordinal(M)
print("\nordinal alpha among the judge's own K=%d samples: %.3f" % (K, a_self))
print("   (the five humans, same coefficient, same items: %.3f)" % a5)
print("\nr against the human mean as samples are averaged:")
kc = []
for k in range(1, K + 1):
    with np.errstate(invalid="ignore"):
        v = np.nanmean(M[:, :k], axis=1)
    # a pair whose first k samples all failed has no score at this k; correlate
    # over the pairs that do, and report how many were dropped
    ok = np.isfinite(v)
    r = su.pearson(v[ok], y[ok])
    kc.append(float(r))
    print("   k=%d   r = %.3f%s" % (k, r, "" if ok.all() else "   (n=%d)" % ok.sum()))
R["self_consistency"] = dict(alpha_self=float(a_self), alpha_humans=float(a5),
                             r_by_k=kc)

# calibrated, per-seed, exactly as exp2 evaluates every other metric
from sklearn.isotonic import IsotonicRegression
raw_r, cal_r, cal_rmse = [], [], []
for sd in SEEDS:
    tr, va, te = make_split(len(P), sd)
    fit = np.concatenate([tr, va])
    raw_r.append(su.pearson(jm[te], y[te]))
    iso = IsotonicRegression(y_min=1, y_max=10, out_of_bounds="clip").fit(jm[fit], y[fit])
    pc = iso.predict(jm[te])
    cal_r.append(su.pearson(pc, y[te])); cal_rmse.append(su.rmse(pc, y[te]))
lo, hi = su.boot_ci(lambda a, b: su.pearson(a, b), jm, y)
print("\ncorpus correlation (test splits over %d seeds):" % len(SEEDS))
print("   r raw        %.3f +- %.3f" % (np.mean(raw_r), np.std(raw_r)))
print("   r calibrated %.3f +- %.3f" % (np.mean(cal_r), np.std(cal_r)))
print("   RMSE cal     %.2f +- %.2f" % (np.mean(cal_rmse), np.std(cal_rmse)))
print("   r full corpus %.3f  95%% CI [%.3f, %.3f]" % (su.pearson(jm, y), lo, hi))
R["corpus"] = dict(r_raw_mean=float(np.mean(raw_r)), r_raw_sd=float(np.std(raw_r)),
                   r_cal_mean=float(np.mean(cal_r)), r_cal_sd=float(np.std(cal_r)),
                   rmse_cal_mean=float(np.mean(cal_rmse)),
                   r_full=float(su.pearson(jm, y)), r_full_ci=[float(lo), float(hi)])

# ======================================================= C. the diagnostic =====
print("\n" + "=" * 78)
print("C. LEGALEDIT DIAGNOSTIC")
print("=" * 78)

# rebuild exp4's probe set exactly (same seeds, same construction)
ref = _sents("A-25.jsonl") + _sents("code_securite_routiere.jsonl")
rng = np.random.default_rng(7)
pool = [ref[i] for i in rng.choice(len(ref), size=400, replace=False)]
ED = legal_edits(pool, seed=7, per_sentence=1)
n = len(ED)
ident = [dict(original=o, simplification=o) for o, _, _ in ED]
edit = [dict(original=o, simplification=p) for o, p, _ in ED]
unrel = [dict(original=a, simplification=b) for a, b in unrelated_pairs(n, seed=77)]
ALL = ident + edit + unrel
RULES = [r for _, _, r in ED]
T1 = np.array([tier(r) == 1 for r in RULES])

ME = J.score_matrix(ALL, k=K, tag="exp4")
ve = np.nanmean(ME, axis=1)
i_, e_, u_ = ve[:n], ve[n:2 * n], ve[2 * n:]

print("\nmean raw judge score (1-10 scale, directly interpretable):")
print("   identical   %5.2f" % i_.mean())
print("   legaledit   %5.2f   (tier-1 only: %.2f)" % (e_.mean(), e_[T1].mean()))
print("   unrelated   %5.2f" % u_.mean())
margin = float((i_ - e_).mean() / max(1e-9, (i_ - u_).mean()))
discrim = float((i_ > e_).mean())
d, blo, bhi, _ = su.boot_diff_test(lambda a, b: float(np.mean(a)), i_, e_, np.zeros(n))
print("\n   margin fraction  %.3f   (NLI-bidirectional 0.670; best similarity 0.039)" % margin)
print("   discrimination   %.1f%%  (identical ranked strictly above the edit)" % (discrim * 100))
print("   identical - edit %.2f points, 95%% CI [%.2f, %.2f]" % (d, blo, bhi))

m1 = float((i_[T1] - e_[T1]).mean() / max(1e-9, (i_[T1] - u_[T1]).mean()))
m2 = float((i_[~T1] - e_[~T1]).mean() / max(1e-9, (i_[~T1] - u_[~T1]).mean()))
print("   margin fraction, tier-1 %.3f   tier-2 %.3f" % (m1, m2))

R["diagnostic"] = dict(n=n, identical=float(i_.mean()), legaledit=float(e_.mean()),
                       legaledit_t1=float(e_[T1].mean()), unrelated=float(u_.mean()),
                       margin_frac=margin, margin_frac_t1=m1, margin_frac_t2=m2,
                       discrim=discrim, ident_minus_edit=float(d),
                       ident_minus_edit_ci=[float(blo), float(bhi)])

print("\nper rule (mean judge score on the edited clause, n = count):")
by = collections.defaultdict(list)
for r, s in zip(RULES, e_):
    by[r].append(s)
rows = sorted(by.items(), key=lambda kv: np.mean(kv[1]))
R["by_rule"] = {}
for r, v in rows:
    print("   %-34s %5.2f  (n=%3d, tier %d)" % (r, np.mean(v), len(v), tier(r)))
    R["by_rule"][r] = dict(mean=float(np.mean(v)), n=len(v), tier=tier(r))

# ------- what the judge says is wrong, not just the number it returns ---------
def norm(t):
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().upper()
    for kk in ("INCOHERENCE", "OMISSION", "HALLUCINATION", "CONFUSION"):
        if kk in t:
            return kk
    return "OTHER"

recs = J.records(ALL, k=K, tag="exp4")
def err_stats(sl):
    tot = named = flagged = 0
    cnt = collections.Counter()
    for rr in recs[sl]:
        for one in rr:
            if not one or one.get("score") is None:
                continue
            tot += 1
            es = [norm(e.get("type", "") if isinstance(e, dict) else e)
                  for e in (one.get("errors") or [])]
            if es:
                flagged += 1
            if "INCOHERENCE" in es:
                named += 1
            cnt.update(es)
    return tot, flagged, named, cnt

print("\nerror types the judge reports (over all %d samples per probe set):" % K)
for label, sl in [("identical", slice(0, n)), ("legaledit", slice(n, 2 * n)),
                  ("unrelated", slice(2 * n, 3 * n))]:
    tot, flagged, named, cnt = err_stats(sl)
    print("   %-10s any error %5.1f%%   names INCOHERENCE %5.1f%%   %s"
          % (label, 100 * flagged / max(1, tot), 100 * named / max(1, tot),
             dict(cnt.most_common(4))))
    R.setdefault("errors", {})[label] = dict(
        samples=tot, any_error=flagged / max(1, tot), incoherence=named / max(1, tot),
        counts=dict(cnt))

# ============================================ D. does corpus r predict margin? ==
# The judge tops the corpus-correlation table and sits fourth on the diagnostic;
# bidirectional NLI does the reverse. With the judge in the comparison there are
# now enough scorers evaluated on both axes to ask the paper's central question
# quantitatively rather than by inspection: across metrics, does agreement with
# human ratings predict sensitivity to a legally decisive edit?
B = json.load(open(os.path.join(OUT, "exp2_baselines.json")))
C = json.load(open(os.path.join(OUT, "exp4_challenge.json")))["rows"]
both = []
for k in C:
    if k in B and not k.startswith("surface:"):
        rc = float(np.mean([d["r"] for d in B[k]["per_seed_cal"]]))
        both.append((k, rc, C[k]["margin_frac"]))
both.sort(key=lambda t: -t[1])
rr = np.array([t[1] for t in both]); mm = np.array([t[2] for t in both])
print("\n" + "=" * 78)
print("D. DOES CORPUS CORRELATION PREDICT DIAGNOSTIC SENSITIVITY?  (n=%d metrics)" % len(both))
print("=" * 78)
print("\n   %-34s %8s %8s %6s %6s" % ("metric", "r_cal", "margin", "rk_r", "rk_m"))
kr = {k: i + 1 for i, (k, _, _) in enumerate(both)}
km = {k: i + 1 for i, (k, _, _) in enumerate(sorted(both, key=lambda t: -t[2]))}
for k, a, b in both:
    print("   %-34s %8.3f %8.3f %6d %6d" % (k[:34], a, b, kr[k], km[k]))
sp = su.spearman(rr, mm); pe = su.pearson(rr, mm)
lo2, hi2 = su.boot_ci(lambda a, b: su.spearman(a, b), rr, mm)
print("\n   Spearman(corpus r_cal, margin fraction) = %.3f  95%% CI [%.3f, %.3f]"
      % (sp, lo2, hi2))
print("   Pearson  (same)                         = %.3f" % pe)
R["predictiveness"] = dict(
    metrics=[dict(name=k, r_cal=a, margin_frac=b, rank_r=kr[k], rank_margin=km[k])
             for k, a, b in both],
    spearman=float(sp), spearman_ci=[float(lo2), float(hi2)], pearson=float(pe))

# =============================================================== D. cost =======
u1, u2 = J.usage_summary(tag="frjudge"), J.usage_summary(tag="exp4")
tot = {k: u1.get(k, 0) + u2.get(k, 0) for k in set(u1) | set(u2)}
# deepseek-chat list price, USD per 1M tokens (cache miss / hit / output)
cost = (tot.get("prompt_tokens", 0) - tot.get("prompt_cache_hit_tokens", 0)) / 1e6 * 0.27 \
     + tot.get("prompt_cache_hit_tokens", 0) / 1e6 * 0.07 \
     + tot.get("completion_tokens", 0) / 1e6 * 1.10
print("\n" + "=" * 78)
print("D. COST:  %d calls (%d failed), %d prompt + %d completion tokens, ~$%.2f"
      % (tot.get("calls", 0), tot.get("failed", 0), tot.get("prompt_tokens", 0),
         tot.get("completion_tokens", 0), cost))
print("=" * 78)
R["usage"] = dict(total=tot, est_usd=round(cost, 2), frjudge=u1, exp4=u2)

json.dump(R, open(os.path.join(OUT, "exp5_llm_judge.json"), "w"), indent=2, default=float)
print("\nwrote results/exp5_llm_judge.json")
