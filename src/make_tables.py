"""Generate every LaTeX table in the paper from the results JSONs."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
TAB = os.path.join(ROOT, "paper", "tables"); os.makedirs(TAB, exist_ok=True)
J = lambda n: json.load(open(os.path.join(RES, n)))
W = lambda n, s: open(os.path.join(TAB, n), "w").write(s)
esc = lambda s: s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def fmt(v, d=3):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


# One size policy for every table in the paper. acl.sty gives \small and
# \footnotesize the same 9pt body and \scriptsize 7pt, so there is no useful
# middle size; instead of scaling each table by whatever factor \resizebox
# needs -- which is what made the tables print at five different sizes, with
# five different booktabs rule weights -- every table is set at \small by its
# float and made to fit by tightening the column separation. The three that
# still do not fit a column are set as full-width table* floats in main.tex.
COLSEP = "\\setlength{\\tabcolsep}{4pt}"

# Names long enough to burst a single column, shortened consistently with the
# short forms used in the figures. The unabbreviated names are in the tables.
SHORT = {
    "BERTScore-CamemBERTv2-Recall": "BERTScore-CmBv2-R",
    "BERTScore-CamemBERTv2": "BERTScore-CmBv2",
    "Sentence-CamemBERT": "Sent-CamemBERT",
    "NLI-bidirectional (min)": "NLI-bidirectional",
    "NLI-fwd (no hallucination)": "NLI-fwd",
    "NLI-bwd (no omission)": "NLI-bwd",
    "LLM-judge (deepseek-chat, k=5)": "LLM-judge (k=5)",
    "surface: token Jaccard": "token Jaccard",
    "surface: -|len diff|": "$-|\\,|s|-|o|\\,|$",
}


# ---------------------------------------------------------------- Table: IAA --
def tab_agreement():
    e = J("exp1_ceiling.json")
    s = [COLSEP, "\\begin{tabular}{lrrr}", "\\toprule",
         "Dimension & \\multicolumn{2}{c}{Krippendorff's $\\alpha$} & Reported \\\\",
         "\\cmidrule(lr){2-3}",
         " & nominal & appropriate & prev. \\\\", "\\midrule",
         f"Simplicity level & {fmt(e['alpha_simplicity_nominal'])} & "
         f"{fmt(e['alpha_simplicity_nominal'])} & 0.18 \\\\",
         f"Characterization & {fmt(e['alpha_characterization_nominal'])} & "
         f"{fmt(e['alpha_characterization_nominal'])} & 0.55 \\\\",
         f"Legal meaning & 0.104 & \\textbf{{{fmt(e['alpha_lmp_ordinal'])}}} & 0.10 \\\\",
         "\\bottomrule", "\\end{tabular}"]
    W("agreement.tex", "\n".join(s))


# ------------------------------------------------------------ Table: ceiling --
def tab_ceiling():
    e = J("exp1_ceiling.json")
    s = [COLSEP, "\\begin{tabular}{lrrr}", "\\toprule",
         "Held-out & $r$ (95\\% CI) & $\\rho$ & RMSE \\\\", "\\midrule"]
    for a, v in e["leave_one_annotator_out"].items():
        s.append(f"{a} & {fmt(v['r'])} [{fmt(v['ci'][0],2)}, {fmt(v['ci'][1],2)}] & "
                 f"{fmt(v['spearman'])} & {fmt(v['rmse'],2)} \\\\")
    s += ["\\midrule",
          f"\\textbf{{Mean (ceiling)}} & \\textbf{{{fmt(e['ceiling_r'])}}} & "
          f"{fmt(e['ceiling_spearman'])} & {fmt(e['ceiling_rmse'],2)} \\\\",
          f"Best pair ({esc(e['best_human_pair']['pair'])}) & {fmt(e['best_human_pair']['r'])} & -- & -- \\\\",
          f"Mean pairwise & {fmt(e['mean_pairwise_r'])} & -- & -- \\\\",
          "\\bottomrule", "\\end{tabular}"]
    W("ceiling.tex", "\n".join(s))


# ---------------------------------------------------------- Table: baselines --
BASE_ORDER = ["BERTScore-CamemBERTv2", "BERTScore-CamemBERTv2-Recall", "BERTScore-FlauBERT",
              "LaBSE", "Sentence-CamemBERT", "Para-mUSE-mpnet", "mE5-base",
              "NLI-fwd (no hallucination)", "NLI-bwd (no omission)", "NLI-bidirectional (min)"]
SURF = ["surface: abs_len_diff", "surface: jaccard", "surface: len_ratio",
        "Surface-Ridge (supervised, 10 feats)"]


def tab_baselines():
    b = J("exp2_baselines.json"); e = J("exp1_ceiling.json")
    ag = lambda rec, key, f: (float(np.mean([d[f] for d in rec[key]])),
                              float(np.std([d[f] for d in rec[key]])))
    s = [COLSEP, "\\begin{tabular}{lrrrrr}", "\\toprule",
         "Metric & $r$ & $\\rho$ & RMSE$_{\\text{raw}}$ & RMSE$_{\\text{cal}}$ & over\\% \\\\",
         "\\midrule", "\\multicolumn{6}{l}{\\emph{Unsupervised, French-capable}}\\\\"]
    def row(name, disp=None):
        if name not in b: return None
        v = b[name]
        r, rs = ag(v, "per_seed_cal", "r"); rho, _ = ag(v, "per_seed_cal", "rho")
        mr, _ = ag(v, "per_seed_raw", "rmse"); mc, _ = ag(v, "per_seed_cal", "rmse")
        ov = float(np.mean(v["per_seed_over_cal"])) * 100
        return (f"{esc(disp or name)} & {fmt(r)}$_{{\\pm{rs:.2f}}}$ & {fmt(rho)} & "
                f"{fmt(mr,2)} & {fmt(mc,2)} & {ov:.1f} \\\\")
    for n in BASE_ORDER:
        r = row(n)
        if r: s.append(r)
    s += ["\\midrule", "\\multicolumn{6}{l}{\\emph{Trivial surface features}}\\\\"]
    for n, d in [("surface: abs_len_diff", "$-|\\,|s|-|o|\\,|$ (length difference)"),
                 ("surface: jaccard", "Token Jaccard"),
                 ("surface: len_ratio", "Length ratio")]:
        r = row(n, d)
        if r: s.append(r)
    r = row("Surface-Ridge (supervised, 10 feats)", "Surface-Ridge (10 features, supervised)")
    if r: s.append(r)
    # The prompted judge is unsupervised in the same sense as the metric zoo --
    # it never sees a training label -- but it is a different kind of system, so
    # it gets its own block rather than being folded into the encoder metrics.
    jn = [k for k in b if k.startswith("LLM-judge")]
    if jn:
        s += ["\\midrule", "\\multicolumn{6}{l}{\\emph{Prompted LLM judge}}\\\\"]
        for n in jn:
            r = row(n)
            if r: s.append(r)
    s += ["\\midrule",
          f"\\textbf{{Human ceiling}} (one annotator) & \\textbf{{{fmt(e['ceiling_r'])}}} & "
          f"{fmt(e['ceiling_spearman'])} & -- & {fmt(e['ceiling_rmse'],2)} & -- \\\\",
          "\\bottomrule", "\\end{tabular}"]
    W("baselines.tex", "\n".join(s))


# ------------------------------------------------------------- Table: models --
def nli_reference(tag):
    """The untrained NLI-bidirectional row, rebuilt on the same family of splits
    the trained models used. The trained-model table's version of this row is an isotonic
    calibration fitted on train+val and scored on test, averaged over the ten
    canonical seeds (exp2_baselines.py); recomputing it here rather than quoting
    it keeps the reference honest when the split changes, since a corpus-level
    correlation is not comparable to a document-grouped one."""
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from data import load_pairs
    from splits import make_split, grouped_split, SEEDS
    import stats_utils as su
    nli = np.load(os.path.join(RES, "cache", "nli3.npy"))[2]
    P = load_pairs(); y = np.array([p["label"] for p in P], float)
    groups = [p["source"] for p in P]
    acc = {"r": [], "rho": [], "rmse": [], "over": []}
    for sd in SEEDS:
        tr, va, te = (grouped_split(groups, sd) if tag == "_grouped"
                      else make_split(len(P), sd))
        fit = np.concatenate([tr, va])
        iso = IsotonicRegression(y_min=1, y_max=10, out_of_bounds="clip").fit(nli[fit], y[fit])
        pc = iso.predict(nli[te])
        acc["r"].append(su.pearson(pc, y[te])); acc["rho"].append(su.spearman(pc, y[te]))
        acc["rmse"].append(su.rmse(pc, y[te])); acc["over"].append(float((pc > y[te]).mean()))
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    return ("\\emph{NLI-bidirectional} (untrained) & "
            f"{fmt(m['r'])} & {fmt(m['rho'])} & {fmt(m['rmse'],2)} & {m['over']*100:.1f} & "
            "1 & -- & -- & \\emph{0.670} \\\\")


# Presentation order for the trained-model tables: the five training-target variants in the
# order Section 6 introduces them, then the two augmented mixtures, which the
# text discusses as a separate group and which a \\midrule sets apart.
MODEL_ORDER = ["JudgeBERT (repro)", "JudgeBERT-Dist (soft labels)",
               "JudgeBERT-Annot (5 heads)", "JudgeBERT-Quantile (tau=.25)",
               "JudgeBERT-MT (+charact.)",
               "JudgeBERT-DA (repro)", "JudgeBERT-DA+LegalEdit"]
AUGMENTED = {"JudgeBERT-DA (repro)", "JudgeBERT-DA+LegalEdit"}

# Config keys are the experiment's internal names; the paper presents this work
# as an extension of JUDGEBERT rather than an outside reproduction of it, so the
# scalar head is named for its objective and not for its provenance.
MODEL_DISPLAY = {"JudgeBERT (repro)": "JudgeBERT-Scalar (baseline)",
                 "JudgeBERT-DA (repro)": "JudgeBERT-DA"}

# Minimum identical-minus-unrelated gap, in points of the 1-10 scale, for the
# LegalEdit margin ratio to be reported at all. See tab_models.
MARGIN_MIN_RANGE = 1.0


def tab_models(tag=""):
    f = os.path.join(RES, f"exp3_train{tag}.json")
    if not os.path.exists(f):
        # Render a marked placeholder rather than leaving a stale file behind.
        # A draft compiled before the grid finishes should say plainly that the
        # numbers are not in yet, instead of printing a bare word in a table and
        # reading as a broken build. As soon as the first (config, seed) is
        # banked the real table replaces this, partially filled.
        W(f"models{tag}.tex",
          "\\begin{tabular}{@{}p{0.92\\columnwidth}@{}}\n\\toprule\n"
          "\\emph{Awaiting results.} This table is generated from "
          f"\\texttt{{results/exp3\\_train{esc(tag)}.json}}, which the training "
          "grid has not yet produced. \\\\\n\\bottomrule\n\\end{tabular}")
        return
    m = json.load(open(f)); e = J("exp1_ceiling.json")
    # nine columns do not fit \\textwidth at the default column separation
    s = [COLSEP,
         "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
         "Model & $r$ & $\\rho$ & RMSE & over\\% & ident.\\ & unrel.\\ & "
         "\\multicolumn{2}{c}{\\textsc{LegalEdit}} \\\\",
         "\\cmidrule(lr){8-9}",
         " &  &  &  &  & pass & pass & false-acc.\\ & margin \\\\", "\\midrule"]
    A = lambda rs, k: (float(np.mean([d[k] for d in rs])), float(np.std([d[k] for d in rs])))
    ordered = ([k for k in MODEL_ORDER if k in m] +
               [k for k in m if k not in MODEL_ORDER])
    seen_aug = False
    for name in ordered:
        rs = m[name]
        if name in AUGMENTED and not seen_aug:
            seen_aug = True
            s.append("\\midrule")
        r, rsd = A(rs, "r"); rho, _ = A(rs, "rho"); rm, _ = A(rs, "rmse")
        ov, _ = A(rs, "over"); ip, _ = A(rs, "identical_pass"); up, _ = A(rs, "unrelated_pass")
        fa, _ = A(rs, "legaledit_false_accept")
        # Margin as a fraction of the identical-to-unrelated range, defined
        # exactly as in the calibrated baseline table so the trained models are comparable
        # to the untrained baselines.
        #
        # The ratio only means something if the model has a range to spend. A
        # regressor that has collapsed onto a constant scores an identical pair
        # and an unrelated pair within a thousandth of a point of each other,
        # and the quotient of two such differences is floating-point noise: the
        # reproduction at seed 42 yields a spurious 0.287 that way. We require
        # the denominator to be at least MARGIN_MIN_RANGE points on the ten-point
        # scale and report "--" otherwise, which is the honest reading of a
        # model that cannot tell the two probes apart in the first place.
        mf = []
        for d in rs:
            den = d["identical_mean"] - d["unrelated_mean"]
            if den >= MARGIN_MIN_RANGE:
                mf.append((d["identical_mean"] - d["legaledit_mean"]) / den)
        mfv = float(np.mean(mf)) if mf else None
        s.append(f"{esc(MODEL_DISPLAY.get(name, name))} & {fmt(r)}$_{{\\pm{rsd:.2f}}}$ & {fmt(rho)} & {fmt(rm,2)} & "
                 f"{ov*100:.1f} & {ip*100:.0f} & {up*100:.0f} & {fa*100:.0f} & {fmt(mfv)} \\\\")
    try:
        nli_line = nli_reference(tag)
    except Exception as ex:
        print("  (nli reference row skipped:", type(ex).__name__, ex, ")")
        nli_line = None
    s += ["\\midrule"] + ([nli_line] if nli_line else [] ) + [
          f"\\textbf{{Human ceiling}} & \\textbf{{{fmt(e['ceiling_r'])}}} & {fmt(e['ceiling_spearman'])} & "
          f"{fmt(e['ceiling_rmse'],2)} & -- & -- & -- & -- & -- \\\\",
          "\\bottomrule", "\\end{tabular}"]
    W(f"models{tag}.tex", "\n".join(s))


# ---------------------------------------------------------- Table: challenge --
# Row names for the diagnostic table. The surface keys carry bare "|"
# characters, which Times sets as a thin bar that reads as an "l"; both are
# written in math instead, matching the notation used in the baseline table.
DISPLAY = {"surface: -|len diff|": "surface: $-|\\,|s|-|o|\\,|$",
           "surface: token Jaccard": "surface: token Jaccard"}


def tab_challenge():
    f = os.path.join(RES, "exp4_challenge.json")
    if not os.path.exists(f): return
    c = json.load(open(f))["rows"]
    s = [COLSEP, "\\begin{tabular}{lrrrrr}", "\\toprule",
         "Metric & identical & \\textsc{LegalEdit} & unrelated & discrim.\\% & margin \\\\",
         "\\midrule"]
    for k in sorted(c, key=lambda k: -c[k]["margin_frac"]):
        v = c[k]
        s.append(f"{DISPLAY.get(k) or esc(k)} & {fmt(v['identical'])} & {fmt(v['legaledit'])} & "
                 f"{fmt(v['unrelated'])} & {v['discrim']*100:.1f} & {fmt(v['margin_frac'])} \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    W("challenge.tex", "\n".join(s))


# ------------------------------------------------------- numbers.tex macros --
def numbers():
    e = J("exp1_ceiling.json")
    n = {}
    n["alphaOrd"] = f"{e['alpha_lmp_ordinal']:.3f}"
    n["ceilR"] = f"{e['ceiling_r']:.3f}"
    n["ceilRmse"] = f"{e['ceiling_rmse']:.2f}"
    n["bestPairR"] = f"{e['best_human_pair']['r']:.3f}"
    n["meanPairR"] = f"{e['mean_pairwise_r']:.3f}"
    n["sbRel"] = f"{e['spearman_brown_reliability_k5']:.3f}"
    n["sbSqrt"] = f"{np.sqrt(e['spearman_brown_reliability_k5']):.2f}"
    n["lenientMean"] = f"{e['clusters']['lenient_mean']:.2f}"
    n["strictMean"] = f"{e['clusters']['strict_mean']:.2f}"
    n["medianRange"] = f"{e['spread']['median_range']:.0f}"
    n["pctSpreadSix"] = f"{e['spread']['pct_range_ge6']*100:.0f}\\%"
    n["labelSd"] = f"{e['label_sd']:.2f}"
    n["testN"] = str(e["test_n"])
    n["mindetect"] = f"{e['min_detectable_r_vs_0.46_at_test_n']:.2f}"
    n["annotMeans"] = ", ".join(f"{v['mean']:.2f}" for v in e["annotator"].values())

    # challenge-set size / minimality, computed the same way the paper describes
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    from data import load_pairs
    from augment import legal_edits, _sents
    import re
    ref = _sents("A-25.jsonl") + _sents("code_securite_routiere.jsonl")
    rng = np.random.default_rng(7)
    pool = [ref[i] for i in rng.choice(len(ref), size=400, replace=False)]
    ED = legal_edits(pool, seed=7, per_sentence=1)
    tok = lambda t: set(re.findall(r"\w+", t.lower()))
    jac = [len(tok(o) & tok(p)) / len(tok(o) | tok(p)) for o, p, _ in ED]
    n["numedits"] = str(len(ED))
    n["numeditsTierOne"] = str(sum(1 for _, _, r in ED if r not in
                                   __import__("augment").TIER2))
    n["meanjaccard"] = f"{np.mean(jac):.2f}"
    n["minjaccard"] = f"{np.min(jac):.2f}"

    try:
        b = J("exp2_baselines.json")
        k = "Surface-Ridge (supervised, 10 feats)"
        n["surfR"] = f"{np.mean([d['r'] for d in b[k]['per_seed_raw']]):.2f}"
    except Exception:
        n["surfR"] = "0.61"
    try:
        c = J("exp4_challenge.json")["rows"]
        mf = [v["margin_frac"] for v in c.values()]
        n["maxMargin"] = f"{max(mf):.3f}"
        n["meanMargin"] = f"{np.mean(mf):.3f}"
        n["nMetricsChallenged"] = str(len(mf))
        n["minMargin"] = f"{min(mf):.3f}"
        # the similarity family: everything that is not an NLI variant, not one
        # of the two raw surface probes, and not the prompted judge. The judge
        # exclusion matters: \bertMargin is quoted in the abstract as the TOP of
        # the similarity range, and the judge's 0.370 would silently widen the
        # claimed "0.022--0.039" to "0.022--0.370" and contradict the sentence
        # it appears in.
        sim = [v["margin_frac"] for k, v in c.items()
               if "NLI" not in k and not k.startswith("surface")
               and not k.startswith("LLM-judge")]
        n["bertMargin"] = f"{max(sim):.3f}"
        n["simMarginLo"] = f"{min(sim):.3f}"
        # party-swap items pooled over the similarity family (the per-rule cells
        # are too thin to quote separately: 11 and 3 items)
        import pickle as _pk
        _m = json.load(open(os.path.join(RES, "exp4_challenge.json")))
        _r = np.array(_m["rules"]); _n = _m["n"]
        _S = _pk.load(open(os.path.join(RES, "exp4_raw.pkl"), "rb"))
        _p = np.isin(_r, ["party_insured_to_insurer", "party_insurer_to_insured"])
        _v = []
        for _k, _s in _S.items():
            if "NLI" in _k:
                continue
            _a = np.asarray(_s, float)
            _i, _e, _u = _a[:_n], _a[_n:2 * _n], _a[2 * _n:]
            _v.append((_i[_p] - _e[_p]).mean() / (_i.mean() - _u.mean()))
        n["partyMargin"] = f"{np.mean(_v):.3f}"
        n["partyMarginLo"] = f"{min(_v):.3f}"
        n["partyMarginHi"] = f"{max(_v):.3f}"
        n["nParty"] = str(int(_p.sum()))
        # The same quantity over *every* metric in the pickle. ruleMarginMean
        # below averages over all metrics, so a sentence comparing the party
        # cell against it must use this figure and not the similarity-only
        # partyMargin above, or it compares two different metric sets.
        _va = []
        for _k, _s in _S.items():
            _a = np.asarray(_s, float)
            _i, _e, _u = _a[:_n], _a[_n:2 * _n], _a[2 * _n:]
            _va.append((_i[_p] - _e[_p]).mean() / (_i.mean() - _u.mean()))
        n["partyMarginAll"] = f"{np.mean(_va):.3f}"
        # all-rules mean restricted to the similarity family, so the appendix
        # can compare the similarity-only party cell against its own baseline.
        _pr = {}
        for _rule in set(_r.tolist()):
            _mk = _r == _rule
            _cell = []
            for _k, _s in _S.items():
                if "NLI" in _k:
                    continue
                _a = np.asarray(_s, float)
                _i, _e, _u = _a[:_n], _a[_n:2 * _n], _a[2 * _n:]
                _cell.append((_i[_mk] - _e[_mk]).mean() / (_i.mean() - _u.mean()))
            _pr[_rule] = np.mean(_cell)
        n["ruleMarginMeanSim"] = f"{np.mean(list(_pr.values())):.3f}"
    except Exception as ex:
        print("skip party numbers:", type(ex).__name__, ex)

    # ---- numbers the prose quotes that no script used to produce -------------
    # Every one of these was computed by hand for an earlier draft. They are
    # regenerated here so the paper cannot drift from the result files.
    try:
        n.update(_dissociation_numbers())
    except Exception as ex:
        print("skip dissociation numbers:", type(ex).__name__, ex)
    try:
        n.update(_calibration_numbers())
    except Exception as ex:
        print("skip calibration numbers:", type(ex).__name__, ex)
    try:
        n.update(_nli_saturation_numbers())
    except Exception as ex:
        print("skip NLI saturation numbers:", type(ex).__name__, ex)
    try:
        n.update(_rule_numbers())
    except Exception as ex:
        print("skip rule numbers:", type(ex).__name__, ex)
    try:
        n.update(_judge_numbers())
    except Exception as ex:
        print("skip judge numbers:", type(ex).__name__, ex)
    out = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in n.items())
    open(os.path.join(ROOT, "paper", "numbers.tex"), "w").write(out + "\n")
    print("numbers.tex:", n)


# ------------------------------------------------- Table: length-bucket split --
def tab_buckets():
    """Correlation within terciles of |len(simp) - len(orig)|.

    A metric whose signal is concentrated in the buckets where the
    simplification is much shorter is detecting deletion, not legal change.
    """
    import pickle
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    from data import load_pairs
    from splits import make_split, SEEDS
    import features as F
    import stats_utils as su
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    P = load_pairs()
    y = np.array([p["label"] for p in P])
    X = F.matrix(P)
    d = X[:, F.FEATURE_NAMES.index("abs_len_diff")]
    q = np.quantile(d, [1 / 3, 2 / 3])
    bucket = np.digitize(d, q)                      # 0 = smallest length change
    BN = ["small", "medium", "large"]

    def by_bucket(idx, pred):
        """idx/pred are aligned arrays over (possibly repeated) pair indices."""
        out = []
        for b in range(3):
            m = bucket[idx] == b
            out.append(su.pearson(pred[m], y[idx][m]) if m.sum() > 3 else float("nan"))
        return out, su.pearson(pred, y[idx])

    rows = []

    # (a) the length feature itself, then one representative of each neural
    # family. The section's claim that the neural metrics keep their
    # correlation inside a length stratum is a plural one, so it needs more
    # than the single best-correlating metric to rest on.
    raw_f = os.path.join(RES, "raw_metric_scores.pkl")
    all_idx = np.arange(len(P))
    rows.append(("$-|\\,|s|-|o|\\,|$",) + by_bucket(all_idx, -d))
    if os.path.exists(raw_f):
        S = pickle.load(open(raw_f, "rb"))
        for k in ("BERTScore-FlauBERT", "LaBSE", "NLI-bidirectional (min)"):
            if k in S:
                rows.append((esc(SHORT.get(k, k)),) + by_bucket(all_idx, np.asarray(S[k], float)))

    # (b) the supervised surface baseline, pooled out-of-fold over the seeds
    pi, pp = [], []
    for sd in SEEDS:
        tr, va, te = make_split(len(P), sd)
        fit = np.concatenate([tr, va])
        ss = StandardScaler().fit(X[fit])
        m = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(ss.transform(X[fit]), y[fit])
        pi.append(te); pp.append(np.clip(m.predict(ss.transform(X[te])), 1, 10))
    rows.append(("Surface-Ridge",) + by_bucket(np.concatenate(pi),
                                               np.concatenate(pp)))

    # (c) the trained models, pooled over seeds
    # Only the reproduction. This table is about the length confound in the
    # metrics of the diagnostic, not about the training targets; carrying
    # all seven variants here would add six rows the surrounding argument never
    # refers to.
    f = os.path.join(RES, "exp3_train.json")
    if os.path.exists(f):
        m = json.load(open(f))
        for name in [k for k in ("JudgeBERT (repro)",) if k in m]:
            rs = m[name]
            idx = np.concatenate([r["test_idx"] for r in rs])
            pr = np.concatenate([r["preds"] for r in rs])
            disp = MODEL_DISPLAY.get(name, name).replace(" (baseline)", "")
            rows.append((esc(disp),) + by_bucket(idx, pr))

    s = [COLSEP, "\\begin{tabular}{lrrrr}", "\\toprule",
         "& \\multicolumn{3}{c}{$r$ within tercile} & \\\\",
         "\\cmidrule(lr){2-4}",
         "Metric & " + " & ".join(BN) + " & all \\\\", "\\midrule"]
    for name, per, overall in rows:
        s.append(f"{name} & " + " & ".join(fmt(v, 3) for v in per) +
                 f" & {fmt(overall)} \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    W("buckets.tex", "\n".join(s))


# ------------------------------------------------- Table: significance tests --
def tab_significance():
    """Williams tests: is the length feature really better than the neural metrics?

    All correlations are with the same human label on the same 297 items, so the
    comparisons are dependent and Williams (1959) is the correct test; a naive
    two-sample z on Fisher-transformed r would be anticonservative here.
    """
    import pickle
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    from data import load_pairs
    import features as F
    import stats_utils as su

    P = load_pairs()
    y = np.array([p["label"] for p in P])
    X = F.matrix(P)
    ref = -X[:, F.FEATURE_NAMES.index("abs_len_diff")]     # the trivial baseline
    r_ref = su.pearson(ref, y)
    S = pickle.load(open(os.path.join(RES, "raw_metric_scores.pkl"), "rb"))

    s = [COLSEP, "\\begin{tabular}{lrrrr}", "\\toprule",
         "Neural metric $m$ & $r(m,h)$ & $r(m,\\ell)$ & $t$ & $p$ \\\\",
         "\\midrule"]
    for k in BASE_ORDER:
        if k not in S:
            continue
        v = np.asarray(S[k], float)
        r_m = su.pearson(v, y)
        r_ml = su.pearson(v, ref)
        t, df, p = su.williams_test(r_ref, r_m, r_ml, len(y))
        star = "$<$0.001" if p < 1e-3 else f"{p:.3f}"
        s.append(f"{esc(k)} & {fmt(r_m)} & {fmt(r_ml)} & {t:.2f} & {star} \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    W("significance.tex", "\n".join(s))
    return r_ref, len(y)


# ------------------------------------------ Table: LegalEdit per-rule margins --
RULE_DISPLAY = {
    "modal_obligation_to_permission": "\\emph{doit} $\\rightarrow$ \\emph{peut}",
    "modal_permission_to_obligation": "\\emph{peut} $\\rightarrow$ \\emph{doit}",
    "party_insured_to_insurer": "\\emph{assur\\'e} $\\rightarrow$ \\emph{assureur}",
    "party_insurer_to_insured": "\\emph{assureur} $\\rightarrow$ \\emph{assur\\'e}",
    "coverage_include_to_exclude": "\\emph{couvre} $\\rightarrow$ \\emph{ne couvre pas}",
    "coverage_exclude_to_include": "\\emph{exclut} $\\rightarrow$ \\emph{inclut}",
    "negation_drop": "drop \\emph{ne \\dots pas}",
    "amount_x10": "amount $\\times$ 10",
    "scope_all_to_some": "\\emph{tous les} $\\rightarrow$ \\emph{certains}",
    "temporal_flip": "\\emph{avant} $\\leftrightarrow$ \\emph{apr\\`es}",
    "condition_unless_to_if": "\\emph{sauf si} $\\rightarrow$ \\emph{si}",
    "obligation_of_result": "\\emph{est tenu} $\\rightarrow$ \\emph{n'est pas tenu}",
    "connective_and_to_or": "\\emph{et} $\\rightarrow$ \\emph{ou}",
    "connective_or_to_and": "\\emph{ou} $\\rightarrow$ \\emph{et}",
    "vehicle_class": "\\emph{automobile} $\\leftrightarrow$ \\emph{v\\'ehicule}",
}


def tab_rules():
    """Margin fraction broken down by perturbation rule, averaged over metrics.

    Answers the obvious follow-up: is the failure uniform, or are some legal
    operations detectable?
    """
    import pickle
    f = os.path.join(RES, "exp4_challenge.json")
    raw = os.path.join(RES, "exp4_raw.pkl")
    if not (os.path.exists(f) and os.path.exists(raw)):
        return
    meta = json.load(open(f))
    rules = np.array(meta["rules"]); n = meta["n"]
    S = pickle.load(open(raw, "rb"))
    import collections
    per_rule = collections.defaultdict(list)
    counts = collections.Counter(rules.tolist())
    for name, v in S.items():
        v = np.asarray(v, float)
        i_, e_, u_ = v[:n], v[n:2 * n], v[2 * n:]
        denom = i_.mean() - u_.mean()
        if abs(denom) < 1e-9:
            continue
        for r in counts:
            m = rules == r
            per_rule[r].append((i_[m] - e_[m]).mean() / denom)
    order = sorted(per_rule, key=lambda r: np.mean(per_rule[r]), reverse=True)
    s = [COLSEP, "\\begin{tabular}{lrrr}", "\\toprule",
         "Perturbation & $n$ & mean margin & worst \\\\", "\\midrule"]
    for r in order:
        vals = np.array(per_rule[r])
        s.append(f"{RULE_DISPLAY.get(r, esc(r))} & {counts[r]} & "
                 f"{fmt(float(vals.mean()))} & {fmt(float(vals.min()))} \\\\")
    s += ["\\midrule",
          f"\\textbf{{All}} & {n} & \\textbf{{"
          f"{fmt(float(np.mean([np.mean(v) for v in per_rule.values()])))}}} & -- \\\\",
          "\\bottomrule", "\\end{tabular}"]
    W("rules.tex", "\n".join(s))


# --------------------------------------------- Table: LegalEdit examples --
def tab_examples(k=5):
    """Show the actual perturbations, with the changed span marked."""
    import sys as _s, os as _o, re, difflib
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    from augment import legal_edits, _sents, tier
    ref = _sents("A-25.jsonl") + _sents("code_securite_routiere.jsonl")
    rng = np.random.default_rng(7)
    pool = [ref[i] for i in rng.choice(len(ref), size=400, replace=False)]
    ED = legal_edits(pool, seed=7, per_sentence=1)
    seen, chosen = set(), []
    for o, p, r in ED:
        if r in seen or len(o) > 165:
            continue
        seen.add(r); chosen.append((o, p, r))
        if len(chosen) >= k:
            break

    def latex(s):
        # T1/ptm has no U+2019; French typographic apostrophes and quotes are
        # normalised to their ASCII equivalents before escaping.
        s = (s.replace("\u2019", "'").replace("\u2018", "'")
              .replace("\u201c", "``").replace("\u201d", "''")
              .replace("\u2013", "--").replace("\u2014", "--")
              .replace("\u00a0", " ").replace("\u202f", " ")
              .replace("\u00b0", "\x00deg").replace("\u00ba", "\x00deg"))
        s = (s.replace("\\", "").replace("&", "\\&").replace("%", "\\%")
              .replace("_", "\\_").replace("#", "\\#").replace("$", "\\$"))
        return s.replace("\x00deg", "\\textdegree{}")

    def mark(o, p):
        """Bold the tokens that differ, on the perturbed side."""
        ot, pt = o.split(), p.split()
        sm = difflib.SequenceMatcher(None, ot, pt)
        out = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            seg = pt[j1:j2]
            if not seg:
                continue
            txt = latex(" ".join(seg))
            out.append(txt if tag == "equal" else "\\textbf{%s}" % txt)
        return " ".join(out)

    s = [COLSEP, "\\begin{tabular}{@{}p{0.97\\columnwidth}@{}}", "\\toprule"]
    for o, p, r in chosen:
        s.append(f"{RULE_DISPLAY.get(r, esc(r))} \\hfill \\emph{{tier {tier(r)}}} \\\\[1pt]")
        s.append(f"\\footnotesize $o$: {latex(o)} \\\\[1pt]")
        s.append(f"\\footnotesize $s'$: {mark(o, p)} \\\\")
        s.append("\\midrule")
    s[-1] = "\\bottomrule"
    s.append("\\end{tabular}")
    W("examples.tex", "\n".join(s))


# ---------------------------------------------- numbers quoted in the prose --
def _neural_margin_and_r():
    """(calibrated r, LegalEdit margin) for the ten neural baselines."""
    C = J("exp4_challenge.json")["rows"]; B = J("exp2_baselines.json")
    out = {}
    for k in BASE_ORDER:
        if k in C and k in B:
            out[k] = (float(np.mean([d["r"] for d in B[k]["per_seed_cal"]])),
                      C[k]["margin_frac"])
    return out


def _all_margin_and_r(): 
    """(calibrated r, LegalEdit margin) for every scorer in Table challenge.

    This is the set Figure dissociation plots, so it is the set the headline
    coefficient should be computed over. The two keys that differ between the
    challenge and baseline files are aliased.
    """
    C = J("exp4_challenge.json")["rows"]; B = J("exp2_baselines.json")
    alias = {"surface: token Jaccard": "surface: jaccard",
             "surface: -|len diff|": "surface: abs_len_diff"}
    out = {}
    for k, v in C.items():
        bk = alias.get(k, k)
        if bk in B:
            out[k] = (float(np.mean([d["r"] for d in B[bk]["per_seed_cal"]])),
                      v["margin_frac"])
    return out


def _rho_ci(d, boot=10000, seed=0):
    from scipy.stats import spearmanr
    x = np.array([v[0] for v in d.values()]); m = np.array([v[1] for v in d.values()])
    s = spearmanr(x, m)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i].tolist())) < 3:
            continue
        bs.append(spearmanr(x[i], m[i]).statistic)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(s.statistic), float(s.pvalue), lo, hi, len(x)


def _dissociation_numbers(boot=10000, seed=0):
    """Rank correlation between corpus correlation and LegalEdit margin.

    Reported over two metric sets, because they say different things and the
    paper quotes both. The ten neural baselines exclude the length feature and
    the prompted judge, which are exactly the two extremes the prose argues
    from, and over that subset the coefficient is a moderate *positive* 0.58.
    Over the full set of Table challenge, the set Figure dissociation plots, it
    is 0.24. Neither is distinguishable from zero at this many metrics, and the
    honest reading is that the coefficient is unresolvable rather than null.
    """
    rho, p, lo, hi, n = _rho_ci(_neural_margin_and_r(), boot, seed)
    arho, ap, alo, ahi, an = _rho_ci(_all_margin_and_r(), boot, seed)
    return {"dissocRho": f"{rho:.2f}", "dissocRhoLo": f"{lo:.2f}",
            "dissocRhoHi": f"{hi:.2f}", "nDissoc": str(n),
            "dissocRhoP": f"{p:.2f}",
            "dissocRhoAll": f"{arho:.2f}", "dissocRhoAllLo": f"{alo:.2f}",
            "dissocRhoAllHi": f"{ahi:.2f}", "nDissocAll": str(an),
            "dissocRhoAllP": f"{ap:.2f}"}


def _calibration_numbers():
    """How far the isotonic map actually moves r and rho.

    Section "Requirement 2" used to assert that both were unaffected. Only the
    ordering is: isotonic is monotone, so Spearman moves only by the ties its
    plateaus introduce, but the map is nonlinear and Pearson is not invariant
    under it. The two figures below are the largest absolute shift of each
    coefficient over the ten metrics of the comparison set, so the prose can
    state the asymmetry rather than deny the effect.
    """
    B = J("exp2_baselines.json")
    dr, drho = [], []
    for k in BASE_ORDER:
        if k not in B:
            continue
        f = lambda tag, stat: float(np.mean([d[stat] for d in B[k][tag]]))
        dr.append(abs(f("per_seed_cal", "r") - f("per_seed_raw", "r")))
        drho.append(abs(f("per_seed_cal", "rho") - f("per_seed_raw", "rho")))
    return {"isoDeltaR": f"{max(dr):.3f}", "isoDeltaRho": f"{max(drho):.3f}"}


def _nli_saturation_numbers():
    """What bidirectional NLI actually scores on a sentence paired with itself.

    The argument of the discussion turns on this: the metric that best tracks
    legal meaning is the one the identical-pair filter would reject.
    """
    import pickle
    meta = J("exp4_challenge.json"); n = meta["n"]
    S = pickle.load(open(os.path.join(RES, "exp4_raw.pkl"), "rb"))
    v = np.asarray(S["NLI-bidirectional (min)"], float)[:n]
    return {"nliIdentRaw": f"{v.mean():.3f}",
            "nliIdentPass": f"{(v >= 0.99).mean() * 100:.1f}"}


def _rule_numbers():
    """Is the per-rule ordering explained by how many characters the edit moves?

    Reported descriptively in the paper; computed here so the claim that it is
    not explained by edit size is checkable.
    """
    import pickle, collections, re
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    from augment import legal_edits, _sents
    meta = J("exp4_challenge.json")
    rules = np.array(meta["rules"]); n = meta["n"]
    S = pickle.load(open(os.path.join(RES, "exp4_raw.pkl"), "rb"))

    per_rule = collections.defaultdict(list)
    for name, s in S.items():
        a = np.asarray(s, float)
        i_, e_, u_ = a[:n], a[n:2 * n], a[2 * n:]
        den = i_.mean() - u_.mean()
        if abs(den) < 1e-9:
            continue
        for r in set(rules.tolist()):
            m = rules == r
            per_rule[r].append((i_[m] - e_[m]).mean() / den)

    ref = _sents("A-25.jsonl") + _sents("code_securite_routiere.jsonl")
    rng = np.random.default_rng(7)
    pool = [ref[i] for i in rng.choice(len(ref), size=400, replace=False)]
    ED = legal_edits(pool, seed=7, per_sentence=1)
    size = collections.defaultdict(list)
    for o, p, r in ED:
        size[r].append(abs(len(p) - len(o)))

    order = sorted(per_rule)
    mm = np.array([np.mean(per_rule[r]) for r in order])
    sz = np.array([np.mean(size[r]) for r in order])
    import stats_utils as su
    r_size = su.pearson(sz, mm)
    b = np.random.default_rng(0)
    bs = [su.pearson(sz[i], mm[i]) for i in
          (b.integers(0, len(mm), len(mm)) for _ in range(10000))]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    best = max(order, key=lambda r: np.mean(per_rule[r]))
    return {"nRules": str(len(order)),
            "ruleMarginMean": f"{mm.mean():.3f}",
            "bestRuleMargin": f"{np.mean(per_rule[best]):.3f}",
            "ruleSizeR": f"{r_size:.2f}",
            "ruleSizeLo": f"{lo:.2f}", "ruleSizeHi": f"{hi:.2f}"}


def _judge_numbers():
    """Every judge figure the prose quotes, read straight from the result files.

    Nothing here is transcribed by hand: if an experiment is rerun the macros
    move with it, which is the only way a paper with this many numbers stays
    honest.
    """
    n = {}
    A = J("exp5_llm_judge.json")
    n["judgeR"] = f"{A['corpus']['r_cal_mean']:.3f}"
    n["judgeRsd"] = f"{A['corpus']['r_cal_sd']:.3f}"
    n["judgeRraw"] = f"{A['corpus']['r_raw_mean']:.3f}"
    n["judgeRmse"] = f"{A['corpus']['rmse_cal_mean']:.2f}"
    n["judgeMargin"] = f"{A['diagnostic']['margin_frac']:.3f}"
    n["judgeMarginTierOne"] = f"{A['diagnostic']['margin_frac_t1']:.3f}"
    n["judgeDiscrim"] = f"{A['diagnostic']['discrim']*100:.1f}"
    n["judgeIdent"] = f"{A['diagnostic']['identical']:.2f}"
    n["judgeEdit"] = f"{A['diagnostic']['legaledit']:.2f}"
    n["judgeEditTierOne"] = f"{A['diagnostic']['legaledit_t1']:.2f}"
    n["judgeUnrel"] = f"{A['diagnostic']['unrelated']:.2f}"
    n["judgeAlphaSelf"] = f"{A['self_consistency']['alpha_self']:.3f}"
    n["judgeKone"] = f"{A['self_consistency']['r_by_k'][0]:.3f}"
    n["judgeKfive"] = f"{A['self_consistency']['r_by_k'][-1]:.3f}"
    n["judgeMeanRating"] = f"{A['panel']['mean_rating']['JUDGE']:.2f}"
    n["judgeHumanR"] = f"{A['panel']['mean_judge_human_r']:.3f}"
    n["humanHumanR"] = f"{A['panel']['mean_human_human_r']:.3f}"
    n["judgeAlphaPanel"] = f"{A['panel']['alpha_6']:.3f}"
    n["judgeErrEdit"] = f"{A['errors']['legaledit']['any_error']*100:.1f}"
    n["judgeErrIncoh"] = f"{A['errors']['legaledit']['incoherence']*100:.1f}"
    n["judgeErrIdent"] = f"{A['errors']['identical']['any_error']*100:.1f}"
    n["judgeCalls"] = f"{A['usage']['total']['calls']:,}".replace(",", "{,}")
    n["judgeCost"] = f"{A['usage']['est_usd']:.2f}"
    rr = A["by_rule"]
    n["asymWeaken"] = f"{rr['modal_obligation_to_permission']['mean']:.2f}"
    n["asymStrengthen"] = f"{rr['modal_permission_to_obligation']['mean']:.2f}"
    n["asymGap"] = f"{rr['modal_permission_to_obligation']['mean'] - rr['modal_obligation_to_permission']['mean']:.2f}"
    n["asymWeakenN"] = str(rr["modal_obligation_to_permission"]["n"])
    n["asymStrengthenN"] = str(rr["modal_permission_to_obligation"]["n"])

    try:
        B = J("exp5b_replication.json")
        n["repRcalOne"] = f"{B['r_cal_mean']['run1']:.3f}"
        n["repRcalTwo"] = f"{B['r_cal_mean']['run2']:.3f}"
        n["repMarginOne"] = f"{B['margin_frac']['run1']:.3f}"
        n["repMarginTwo"] = f"{B['margin_frac']['run2']:.3f}"
        n["repItemR"] = f"{B['item_stability']['frjudge_r']:.3f}"
        n["repProbeR"] = f"{B['item_stability']['probe_r']:.3f}"
        n["repAsymOne"] = f"{B['asymmetry_gap']['run1']:.2f}"
        n["repAsymTwo"] = f"{B['asymmetry_gap']['run2']:.2f}"
        n["repNstats"] = str(sum(1 for k, v in B.items()
                                 if isinstance(v, dict) and v.get("ok") is not None))
    except Exception as ex:
        print("skip replication numbers:", type(ex).__name__, ex)

    try:
        C = J("exp7_panel.json")["configs"]
        pv = {k: v for k, v in C.items() if k.startswith("chat v")}
        ms = [v["margin_frac"] for v in pv.values()]
        rs = [v["r_cal_mean"] for v in pv.values()]
        n["promptMarginLo"] = f"{min(ms):.3f}"
        n["promptMarginHi"] = f"{max(ms):.3f}"
        n["promptMarginRange"] = f"{max(ms) - min(ms):.3f}"
        n["promptRLo"] = f"{min(rs):.3f}"
        n["promptRHi"] = f"{max(rs):.3f}"
        n["nPrompts"] = str(len(pv))
        for key, lab in [("Bare", "chat v3 (no taxonomy)"), ("Rubric", "chat v1 (rubric, FR)")]:
            if lab in C:
                n["prompt%sMargin" % key] = f"{C[lab]['margin_frac']:.3f}"
                n["prompt%sR" % key] = f"{C[lab]['r_cal_mean']:.3f}"
                n["prompt%sEdit" % key] = f"{C[lab]['legaledit']:.2f}"
        if "reasoner (K=1)" in C and "chat (K=1)" in C:
            n["reasonerR"] = f"{C['reasoner (K=1)']['r_cal_mean']:.3f}"
            n["reasonerMargin"] = f"{C['reasoner (K=1)']['margin_frac']:.3f}"
            n["chatKoneR"] = f"{C['chat (K=1)']['r_cal_mean']:.3f}"
            n["chatKoneMargin"] = f"{C['chat (K=1)']['margin_frac']:.3f}"
        gaps = [v["asym"] for v in C.values()]
        n["asymGapLo"] = f"{min(gaps):.2f}"
        n["asymGapHi"] = f"{max(gaps):.2f}"
        n["nConfigs"] = str(len(C))
    except Exception as ex:
        print("skip panel numbers:", type(ex).__name__, ex)

    try:
        D = J("exp6_hybrid.json")
        w = "weighted w*judge+(1-w)*NLI"
        n["hybridR"] = f"{D['run1'][w]['r_cal_mean']:.3f}"
        n["hybridMargin"] = f"{D['run1'][w]['margin_frac']:.3f}"
        n["hybridW"] = f"{D['w_mean']:.2f}"
        n["judgeCalMargin"] = f"{D['run1']['judge (calibrated)']['margin_frac']:.3f}"
        n["nliCalMargin"] = f"{D['run1']['NLI (calibrated)']['margin_frac']:.3f}"
    except Exception as ex:
        print("skip hybrid numbers:", type(ex).__name__, ex)
    return n


def tab_judge():
    """Model and prompt variants of the prompted judge (Experiment 7)."""
    f = os.path.join(RES, "exp7_panel.json")
    if not os.path.exists(f):
        return
    C = json.load(open(f))["configs"]
    ORDER = ["chat v1 (rubric, FR)", "chat v2 (rubric, EN)", "chat v3 (no taxonomy)",
             "chat (K=1)", "reasoner (K=1)"]
    DISP = {"chat v1 (rubric, FR)": "\\texttt{chat}, rubric (FR) \\emph{[primary]}",
            "chat v2 (rubric, EN)": "\\texttt{chat}, rubric (EN)",
            "chat v3 (no taxonomy)": "\\texttt{chat}, bare instruction",
            "chat (K=1)": "\\texttt{chat}, rubric, $K{=}1$",
            "reasoner (K=1)": "\\texttt{reasoner}, rubric, $K{=}1$"}
    s = [COLSEP, "\\begin{tabular}{lrrrrr}", "\\toprule",
         "Configuration & $r$ & edit & margin & discrim.\\% & asym. \\\\", "\\midrule"]
    for k in [k for k in ORDER if k in C] + [k for k in C if k not in ORDER]:
        v = C[k]
        s.append(f"{DISP.get(k, esc(k))} & {fmt(v['r_cal_mean'])} & {fmt(v['legaledit'],2)} & "
                 f"{fmt(v['margin_frac'])} & {v['discrim']*100:.1f} & {fmt(v['asym'],2)} \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    W("judge.tex", "\n".join(s))


# ------------------------------------------------------------------- driver --
if __name__ == "__main__":
    tab_agreement(); tab_ceiling()
    for fn in (tab_baselines, tab_models, tab_challenge, tab_buckets,
               tab_significance, tab_rules, tab_examples, tab_judge, numbers):
        try:
            fn()
        except Exception as ex:
            print("skip", fn.__name__, type(ex).__name__, ex)
    try:
        tab_models("_grouped")
    except Exception as ex:
        print("skip models_grouped", type(ex).__name__, ex)
    print("tables written to", TAB, sorted(os.listdir(TAB)))
