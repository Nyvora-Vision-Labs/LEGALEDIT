"""Experiment 2: metric zoo + trivial surface baselines, raw and calibrated.

Key methodological fix over the original paper: unsupervised metrics are never
fitted to the human scale, so their RMSE against a 1-10 Likert label is not
comparable to a supervised regressor's. We therefore report each metric twice:
raw (decimal-scaled, as in the original) and calibrated (isotonic map fitted on
the training split only).
"""
import json, os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_pairs
from splits import make_split, SEEDS
import features as F
import stats_utils as su

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "raw_metric_scores.pkl")
CDIR = os.path.join(OUT, "cache"); os.makedirs(CDIR, exist_ok=True)


def cached(name, fn):
    """Per-metric cache: a crash or kill never costs completed work."""
    f = os.path.join(CDIR, name + ".npy")
    if os.path.exists(f):
        print("  cached", name, flush=True); return np.load(f)
    v = np.asarray(fn(), float); np.save(f, v)
    print("  computed", name, flush=True); return v

P = load_pairs()
y = np.array([p["label"] for p in P])


def compute_raw():
    import metrics_zoo as M
    S = {}
    S["BERTScore-CamemBERTv2"] = cached("bs_cam", lambda: M.bertscore_f1(P, "almanach/camembertv2-base"))
    S["BERTScore-CamemBERTv2-Recall"] = cached("bsr_cam", lambda: M.bertscore_recall(P, "almanach/camembertv2-base"))
    S["BERTScore-FlauBERT"] = cached("bs_flau", lambda: M.bertscore_f1(P, "flaubert/flaubert_base_cased"))
    for nm, key, hf in [("LaBSE", "labse", "sentence-transformers/LaBSE"),
                        ("Sentence-CamemBERT", "scam", "dangvantuan/sentence-camembert-base"),
                        ("Para-mUSE-mpnet", "mpnet", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")]:
        S[nm] = cached(key, lambda hf=hf: M.sbert_cosine(P, hf))
    S["mE5-base"] = cached("e5", lambda: M.sbert_cosine(P, "intfloat/multilingual-e5-base", prefix="query: "))
    nli = cached("nli3", lambda: np.stack(M.nli_bidirectional(P, device="cpu")))
    S["NLI-fwd (no hallucination)"], S["NLI-bwd (no omission)"], S["NLI-bidirectional (min)"] = nli
    return S


if os.path.exists(CACHE):
    S = pickle.load(open(CACHE, "rb"))
    print("loaded cached scores:", list(S))
else:
    print("computing raw metric scores ...", flush=True)
    S = compute_raw()
    pickle.dump(S, open(CACHE, "wb"))

# --- prompted LLM judge -------------------------------------------------------
# Kept out of raw_metric_scores.pkl deliberately: it has its own cache, so the
# existing pickle stays valid and a run without an API key is unaffected.
import llm_judge as J
if J.available() and os.environ.get("SKIP_JUDGE") != "1":
    S["LLM-judge (deepseek-chat, k=%d)" % J.K] = cached(
        "llm_judge", lambda: J.deepseek_score(P, tag="frjudge"))
else:
    print("  skipping LLM judge (no DEEPSEEK_API_KEY or SKIP_JUDGE=1)")

# --- trivial surface features -------------------------------------------------
X = F.matrix(P)
for i, nm in enumerate(F.FEATURE_NAMES):
    S["surface: " + nm] = X[:, i] * (-1 if nm == "abs_len_diff" else 1)

# --- decimal scaling (the original paper's normalisation) ---------------------
def decimal_scale(v):
    v = np.asarray(v, float)
    lo, hi = v.min(), v.max()
    return 10 * (v - lo) / (hi - lo + 1e-12)

# --- per-seed evaluation ------------------------------------------------------
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

rows = {}
for name, raw in S.items():
    raw = np.asarray(raw, float)
    r_full = su.pearson(raw, y)
    rec = dict(pearson_full=r_full, spearman_full=su.spearman(raw, y),
               kendall_full=su.kendall(raw, y),
               per_seed_raw=[], per_seed_cal=[], per_seed_over=[], per_seed_over_cal=[])
    for sd in SEEDS:
        tr, va, te = make_split(len(P), sd)
        fit = np.concatenate([tr, va])
        sc = decimal_scale(raw)
        rec["per_seed_raw"].append(dict(r=su.pearson(raw[te], y[te]),
                                        rho=su.spearman(raw[te], y[te]),
                                        rmse=su.rmse(sc[te], y[te])))
        rec["per_seed_over"].append(float((sc[te] > y[te]).mean()))
        iso = IsotonicRegression(y_min=1, y_max=10, out_of_bounds="clip").fit(raw[fit], y[fit])
        pc = iso.predict(raw[te])
        rec["per_seed_cal"].append(dict(r=su.pearson(pc, y[te]) if pc.std() > 0 else 0.0,
                                        rho=su.spearman(pc, y[te]) if pc.std() > 0 else 0.0,
                                        rmse=su.rmse(pc, y[te])))
        rec["per_seed_over_cal"].append(float((pc > y[te]).mean()))
    rows[name] = rec

# --- supervised surface-feature model (ridge) --------------------------------
sup = dict(per_seed=[], preds=np.zeros(len(P)))
for sd in SEEDS:
    tr, va, te = make_split(len(P), sd)
    fit = np.concatenate([tr, va])
    ss = StandardScaler().fit(X[fit])
    m = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(ss.transform(X[fit]), y[fit])
    pr = np.clip(m.predict(ss.transform(X[te])), 1, 10)
    sup["per_seed"].append(dict(r=su.pearson(pr, y[te]), rho=su.spearman(pr, y[te]),
                                rmse=su.rmse(pr, y[te]), over=float((pr > y[te]).mean())))
rows["Surface-Ridge (supervised, 10 feats)"] = dict(
    pearson_full=float(np.mean([d["r"] for d in sup["per_seed"]])),
    spearman_full=float(np.mean([d["rho"] for d in sup["per_seed"]])),
    kendall_full=float("nan"),
    per_seed_raw=sup["per_seed"], per_seed_cal=sup["per_seed"],
    per_seed_over=[d["over"] for d in sup["per_seed"]],
    per_seed_over_cal=[d["over"] for d in sup["per_seed"]])

json.dump({k: {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist())
               for kk, vv in v.items()} for k, v in rows.items()},
          open(os.path.join(OUT, "exp2_baselines.json"), "w"), indent=2, default=float)

agg = lambda lst, k: (float(np.mean([d[k] for d in lst])), float(np.std([d[k] for d in lst])))
print(f"\n{'metric':38s} {'r_full':>7s} {'rho_full':>8s} | {'r_test':>13s} {'RMSE_raw':>9s} | "
      f"{'r_cal':>13s} {'RMSE_cal':>9s} {'over%':>6s}")
print("-" * 118)
for name in sorted(rows, key=lambda k: -rows[k]["pearson_full"]):
    v = rows[name]
    rr, rs = agg(v["per_seed_raw"], "r"); mr, _ = agg(v["per_seed_raw"], "rmse")
    cr, cs = agg(v["per_seed_cal"], "r"); mc, _ = agg(v["per_seed_cal"], "rmse")
    ov = float(np.mean(v["per_seed_over_cal"])) * 100
    print(f"{name:38s} {v['pearson_full']:7.3f} {v['spearman_full']:8.3f} | "
          f"{rr:6.3f}±{rs:5.3f} {mr:9.2f} | {cr:6.3f}±{cs:5.3f} {mc:9.2f} {ov:6.1f}")
