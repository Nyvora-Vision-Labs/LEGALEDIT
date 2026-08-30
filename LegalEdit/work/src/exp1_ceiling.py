"""Experiment 1: annotator agreement, human ceiling, disagreement structure."""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_pairs, ANNOTATORS
import stats_utils as su

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)

P = load_pairs()
M = np.array([[p["ratings"][a] for a in ANNOTATORS] for p in P], float)  # 297 x 5
y = M.mean(1)
res = {"n_pairs": len(P)}

# ---- 1. agreement -----------------------------------------------------------
res["alpha_lmp_ordinal"] = su.krippendorff_ordinal(M)
res["alpha_lmp_interval"] = su.krippendorff_interval(M)
simp = np.array([[p["simplicity"][a] for a in ANNOTATORS] for p in P], object)
res["alpha_simplicity_nominal"] = su.krippendorff_nominal(simp)
ch = np.array([[p["characterization"][a] or "NA" for a in ANNOTATORS] for p in P], object)
res["alpha_characterization_nominal"] = su.krippendorff_nominal(ch)

# ---- 2. per-annotator descriptive -------------------------------------------
res["annotator"] = {a: {"mean": float(M[:, i].mean()), "sd": float(M[:, i].std()),
                        "pct_1": float((M[:, i] == 1).mean()), "pct_10": float((M[:, i] == 10).mean())}
                    for i, a in enumerate(ANNOTATORS)}

# ---- 3. pairwise inter-annotator correlation --------------------------------
res["pairwise_r"] = {}
best = (None, -1)
for i in range(5):
    for j in range(i + 1, 5):
        r = su.pearson(M[:, i], M[:, j])
        res["pairwise_r"][f"{ANNOTATORS[i]}-{ANNOTATORS[j]}"] = r
        if r > best[1]:
            best = (f"{ANNOTATORS[i]}-{ANNOTATORS[j]}", r)
res["best_human_pair"] = {"pair": best[0], "r": best[1]}
res["mean_pairwise_r"] = float(np.mean(list(res["pairwise_r"].values())))

# ---- 4. leave-one-annotator-out ceiling -------------------------------------
loo = {}
for i, a in enumerate(ANNOTATORS):
    other = np.delete(M, i, axis=1).mean(1)
    r = su.pearson(M[:, i], other)
    lo, hi = su.boot_ci(su.pearson, M[:, i], other, n=5000)
    loo[a] = {"r": r, "ci": [lo, hi], "rmse": su.rmse(M[:, i], other),
              "spearman": su.spearman(M[:, i], other)}
res["leave_one_annotator_out"] = loo
res["ceiling_r"] = float(np.mean([v["r"] for v in loo.values()]))
res["ceiling_rmse"] = float(np.mean([v["rmse"] for v in loo.values()]))
res["ceiling_spearman"] = float(np.mean([v["spearman"] for v in loo.values()]))

# ---- 5. disagreement structure ----------------------------------------------
rng = M.max(1) - M.min(1)
res["spread"] = {"mean_range": float(rng.mean()), "median_range": float(np.median(rng)),
                 "pct_range_ge6": float((rng >= 6).mean()),
                 "pct_range_ge9": float((rng >= 9).mean()),
                 "mean_within_pair_sd": float(M.std(1).mean())}
res["label_hist"] = {int(k): int(v) for k, v in sorted(collections.Counter(M.ravel().astype(int)).items())}

# two clusters: correlation of cluster means
lenient = [ANNOTATORS.index(a) for a in ANNOTATORS if M[:, ANNOTATORS.index(a)].mean() > 6]
strict = [i for i in range(5) if i not in lenient]
res["clusters"] = {"lenient": [ANNOTATORS[i] for i in lenient],
                   "strict": [ANNOTATORS[i] for i in strict],
                   "lenient_mean": float(M[:, lenient].mean()),
                   "strict_mean": float(M[:, strict].mean()),
                   "r_between_cluster_means": su.pearson(M[:, lenient].mean(1), M[:, strict].mean(1))}

# ---- 6. how much variance does the mean label retain? -----------------------
res["label_sd"] = float(y.std())
res["mean_rater_sd"] = float(M.std(1).mean())
# reliability of the mean of 5 raters (Spearman-Brown from mean pairwise r)
rbar = res["mean_pairwise_r"]
res["spearman_brown_reliability_k5"] = float(5 * rbar / (1 + 4 * rbar))

# ---- 7. power ---------------------------------------------------------------
n_test = int(round(0.30 * len(P)))
res["test_n"] = n_test
res["power_to_detect_r0.74_at_test_n"] = su.power_for_r(0.74, n_test)
res["min_detectable_r_vs_0.46_at_test_n"] = su.min_detectable_r_diff(0.46, n_test)
res["min_detectable_r_vs_0.60_at_test_n"] = su.min_detectable_r_diff(0.60, n_test)

json.dump(res, open(os.path.join(OUT, "exp1_ceiling.json"), "w"), indent=2)

print(f"pairs={res['n_pairs']}  test_n={n_test}")
print(f"alpha LMP ordinal={res['alpha_lmp_ordinal']:.3f} interval={res['alpha_lmp_interval']:.3f}")
print(f"alpha simplicity={res['alpha_simplicity_nominal']:.3f} charact={res['alpha_characterization_nominal']:.3f}")
print("\nleave-one-annotator-out:")
for a, v in loo.items():
    print(f"  {a}: r={v['r']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]  rho={v['spearman']:.3f}  RMSE={v['rmse']:.2f}")
print(f"  CEILING r={res['ceiling_r']:.3f} rho={res['ceiling_spearman']:.3f} RMSE={res['ceiling_rmse']:.2f}")
print(f"\nbest human pair {best[0]} r={best[1]:.3f}; mean pairwise r={rbar:.3f}")
print(f"Spearman-Brown reliability of 5-rater mean = {res['spearman_brown_reliability_k5']:.3f}")
print(f"spread: median range={res['spread']['median_range']:.0f}, pct>=6: {res['spread']['pct_range_ge6']:.1%}")
print(f"clusters lenient={res['clusters']['lenient']} ({res['clusters']['lenient_mean']:.2f}) "
      f"strict={res['clusters']['strict']} ({res['clusters']['strict_mean']:.2f}) "
      f"r={res['clusters']['r_between_cluster_means']:.3f}")
print(f"\npower to detect r=.74 at n={n_test}: {res['power_to_detect_r0.74_at_test_n']:.3f}")
print(f"min detectable r vs .46: {res['min_detectable_r_vs_0.46_at_test_n']:.3f}; vs .60: {res['min_detectable_r_vs_0.60_at_test_n']:.3f}")
