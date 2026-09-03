"""Figures for the paper.

Every figure is drawn at the size it is printed at, so that \\includegraphics
never rescales it. A figure authored at 6in and dropped into a 3in column has
its type shrunk to half the requested point size, which is why the panel text
used to come out smaller than the caption underneath it. The two widths below
are the ACL a4 geometry (\\columnwidth and \\textwidth) in inches; keeping the
figsize equal to one of them makes width=\\columnwidth a no-op scale of 1.0.
"""
import json, os, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_pairs, ANNOTATORS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "paper", "figs"); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(ROOT, "results")

# \columnwidth = 219.086pt, \textwidth = 455.244pt under acl.sty on a4.
COL_W, FULL_W = 219.086 / 72, 455.244 / 72

# The body font is Times at 10pt and captions at 10pt, so panel text sits at
# 6.5--8pt: small enough to read as figure furniture, large enough to read.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "axes.labelpad": 2.5,
    "grid.linewidth": 0.5,
    "grid.color": "#dddddd",
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "legend.borderpad": 0.2,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
})

C_STRICT, C_LEN, C_NLI = "#B4432A", "#2A6FB4", "#1F7A5A"
C_GREY, C_FAINT = "#555555", "#999999"


def save(fig, name):
    """Write at exactly the requested figsize; no tight-bbox trim, so the
    LaTeX-side scale factor stays 1.0 and type sizes survive to the page."""
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


# Long metric names are what makes a dot plot outgrow a column, and the tables
# carry the unabbreviated ones, so the figures use short forms throughout.
SHORT = {
    "NLI-bidirectional (min)": "NLI-bidirectional",
    "NLI-fwd (no hallucination)": "NLI-fwd",
    "NLI-bwd (no omission)": "NLI-bwd",
    "LLM-judge (deepseek-chat, k=5)": "LLM-judge",
    "surface: token Jaccard": "token Jaccard",
    "surface: -|len diff|": "$-|\\,|s|-|o|\\,|$",
    "BERTScore-CamemBERTv2-Recall": "BERTScore-CmBv2-R",
    "BERTScore-CamemBERTv2": "BERTScore-CmBv2",
    "Sentence-CamemBERT": "Sent-CamemBERT",
}


def fig_disagreement():
    P = load_pairs()
    M = np.array([[p["ratings"][a] for a in ANNOTATORS] for p in P], float)
    means = M.mean(0)
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 1.95), layout="constrained")
    # (a) per-annotator rating histograms
    for i, a in enumerate(ANNOTATORS):
        c = C_LEN if means[i] > 6 else C_STRICT
        h = np.bincount(M[:, i].astype(int), minlength=11)[1:]
        ax[0].plot(range(1, 11), h, marker="o", ms=2.5, lw=1.0, color=c, alpha=.85,
                   label=f"{a} ({means[i]:.1f})")
    ax[0].set_xlabel("legal meaning rating"); ax[0].set_ylabel("items")
    ax[0].set_title("(a) two annotator populations")
    ax[0].set_xticks([2, 4, 6, 8, 10])
    ax[0].legend(fontsize=6, frameon=False, ncol=2, loc="upper center",
                 handlelength=1.2, labelspacing=.25, borderaxespad=.1)
    ax[0].margins(y=.22)
    # (b) spread per item
    rng = np.sort(M.max(1) - M.min(1))
    ax[1].fill_between(np.arange(len(rng)) / len(rng) * 100, rng, step="pre",
                       color="#bbbbbb", lw=0)
    ax[1].axhline(6, ls="--", lw=.9, color=C_STRICT)
    ax[1].set_xlabel("percentile of items"); ax[1].set_ylabel("max $-$ min rating")
    ax[1].set_title("(b) within-item disagreement")
    ax[1].set_xlim(0, 100); ax[1].set_ylim(0, rng.max() + 1.4)
    ax[1].text(2, 6.35, f"{(rng>=6).mean()*100:.0f}% span $\\geq$6 pts",
               fontsize=6.2, color=C_STRICT, va="bottom")
    # (c) ceiling vs reported
    e = json.load(open(os.path.join(RES, "exp1_ceiling.json")))
    loo = [v["r"] for v in e["leave_one_annotator_out"].values()]
    sb = np.sqrt(e["spearman_brown_reliability_k5"])
    vals = [e["ceiling_r"], e["best_human_pair"]["r"]]
    ax[2].barh(["single annotator\n(ceiling)", "best annotator\npair"], vals,
               color="#bbbbbb", height=.45)
    ax[2].errorbar([e["ceiling_r"]], [0],
                   xerr=[[e["ceiling_r"] - min(loo)], [max(loo) - e["ceiling_r"]]],
                   fmt="none", ecolor="#333333", capsize=2, lw=.8)
    # the two reference lines the section argues should both be reported
    ax[2].axvline(0.74, color=C_STRICT, lw=1.2)
    ax[2].axvline(sb, color=C_GREY, lw=1.0, ls=(0, (3, 2)))
    ax[2].set_ylim(-0.55, 2.15)
    ax[2].text(0.725, 1.5, "JudgeBERT\nreported 0.74", fontsize=6, color=C_STRICT,
               ha="right", va="center", linespacing=1.3)
    ax[2].text(sb + 0.02, 1.5, "aggregate\nallows %.2f" % sb, fontsize=6, color=C_GREY,
               ha="left", va="center", linespacing=1.3)
    for j, v in enumerate(vals):
        ax[2].text(0.02, j, f"{v:.3f}", fontsize=6.2, color="#222222", va="center",
                   ha="left")
    ax[2].set_xlim(0, 1.0); ax[2].set_xlabel("Pearson $r$ with consensus")
    ax[2].set_title("(c) the human ceiling")
    ax[2].tick_params(axis="y", length=0)
    for a in ax:
        a.spines["left"].set_color("#666666"); a.spines["bottom"].set_color("#666666")
    save(fig, "disagreement.pdf")


def fig_challenge():
    f = os.path.join(RES, "exp4_challenge.json")
    if not os.path.exists(f):
        print("skip challenge fig"); return
    c = json.load(open(f))["rows"]
    order = sorted(c, key=lambda k: -c[k]["margin_frac"])
    # one row per metric plus fixed furniture: legend, axis label, margin header
    fig, ax = plt.subplots(figsize=(COL_W, 0.175 * len(order) + 0.90),
                           layout="constrained")
    yy = np.arange(len(order))
    for j, k in enumerate(order):
        v = c[k]
        ax.plot([v["unrelated"], v["identical"]], [j, j], color="#d5d5d5", lw=2.4,
                zorder=1, solid_capstyle="round")
        ax.scatter(v["unrelated"], j, s=11, color="#444444", zorder=3)
        ax.scatter(v["identical"], j, s=11, color=C_LEN, zorder=3)
        ax.scatter(v["legaledit"], j, s=24, marker="D", color=C_STRICT, zorder=4,
                   edgecolor="white", linewidth=.5)
    # margin value in a gutter to the right of the plotting range
    for j, k in enumerate(order):
        ax.text(1.10, j, f"{c[k]['margin_frac']:.2f}", fontsize=6, va="center",
                ha="left", color="#222222",
                fontweight="bold" if c[k]["margin_frac"] > 0.2 else "normal")
    ax.text(1.10, -0.95, "margin", fontsize=6, va="center", ha="left", style="italic",
            color=C_GREY)

    # separate the family that reacts to the edit from the family that does not
    cut = sum(1 for k in order if c[k]["margin_frac"] > 0.2)
    if 0 < cut < len(order):
        # The rule alone carries the split. An inline caption for it has to sit
        # in a 12pt gap between two rows whose markers run the width of the
        # axes, and at any size that fits it lands on one of them; the caption
        # names the two families instead.
        ax.axhline(cut - 0.5, color=C_FAINT, lw=.7, ls=(0, (4, 3)), zorder=2, xmax=.79)

    ax.set_yticks(yy)
    ax.set_yticklabels([SHORT.get(k, k) for k in order], fontsize=6.2)
    ax.invert_yaxis(); ax.set_xlabel("normalised score")
    ax.set_xlim(-0.03, 1.30)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines["bottom"].set_bounds(0, 1.0)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=11, color=C_LEN, label="identical")
    ax.scatter([], [], s=24, marker="D", color=C_STRICT, label="legal edit")
    ax.scatter([], [], s=11, color="#444444", label="unrelated")
    ax.legend(fontsize=6.2, frameon=False, loc="lower center", bbox_to_anchor=(.42, 1.0),
              ncol=3, handletextpad=.2, columnspacing=.9, borderaxespad=.1)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#666666")
    save(fig, "challenge.pdf")


def fig_dissociation():
    """Corpus correlation against LegalEdit margin.

    The paper's central new claim is that these two orderings come apart, so it
    deserves a picture: every metric is a point, and the axes are the two things
    a practitioner might select a metric on.
    """
    cf = os.path.join(RES, "exp4_challenge.json")
    bf = os.path.join(RES, "exp2_baselines.json")
    if not (os.path.exists(cf) and os.path.exists(bf)):
        print("skip dissociation fig"); return
    C = json.load(open(cf))["rows"]; B = json.load(open(bf))
    ALIAS = {"surface: token Jaccard": "surface: jaccard",
             "surface: -|len diff|": "surface: abs_len_diff"}
    LABEL = {"NLI-bidirectional (min)": "NLI-bidirectional",
             "NLI-fwd (no hallucination)": "NLI-fwd",
             "NLI-bwd (no omission)": "NLI-bwd",
             "LLM-judge (deepseek-chat, k=5)": "LLM-judge",
             "surface: token Jaccard": "token Jaccard",
             "surface: -|len diff|": "$-|\\,|s|-|o|\\,|$"}
    # where to put each label relative to its point, so none lands on another
    # and none lands in the upper-right quadrant the caption points at
    NUDGE = {"NLI-bidirectional": ("left", "center", (7, 0)),
             "NLI-fwd": ("left", "center", (7, 0)),
             "NLI-bwd": ("right", "center", (-7, 0)),
             "LLM-judge": ("right", "center", (-7, 0)),
             "token Jaccard": ("center", "bottom", (0, 6)),
             "$-|\\,|s|-|o|\\,|$": ("center", "bottom", (0, 7))}

    def family(k):
        if "LLM-judge" in k:
            return "judge"
        if "NLI" in k:
            return "nli"
        return "surface" if k.startswith("surface") else "sim"

    pts = []
    for k, v in C.items():
        b = B.get(ALIAS.get(k, k))
        if not b:
            continue
        r = float(np.mean([d["r"] for d in b["per_seed_cal"]]))
        pts.append((k, r, v["margin_frac"], family(k)))

    COL = {"nli": C_NLI, "sim": C_LEN, "surface": C_STRICT, "judge": "#7A4FA3"}
    MARK = {"nli": "o", "sim": "o", "surface": "o", "judge": "D"}
    LBL = {"sim": "embedding / BERTScore", "surface": "surface feature",
           "nli": "entailment", "judge": "prompted LLM judge"}
    fig, ax = plt.subplots(figsize=(COL_W, 2.65), layout="constrained")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for fam in ("sim", "surface", "nli", "judge"):
        xs = [p[1] for p in pts if p[3] == fam]; ys = [p[2] for p in pts if p[3] == fam]
        ax.scatter(xs, ys, s=22, color=COL[fam], label=LBL[fam], zorder=3,
                   marker=MARK[fam], edgecolor="white", linewidth=.6)

    # Label the points that carry the argument individually. The seven
    # embedding metrics sit on top of each other along the floor, so they are
    # bracketed as a group rather than labelled into an unreadable pile.
    for k, r, m, fam in pts:
        if k not in LABEL:
            continue
        t = LABEL[k]
        ha, va, off = NUDGE[t]
        ax.annotate(t, (r, m), fontsize=6.2, color="#222222", ha=ha, va=va,
                    xytext=off, textcoords="offset points")
    sim = [(r, m) for _, r, m, f in pts if f == "sim"]
    x0, x1 = min(p[0] for p in sim), max(p[0] for p in sim)
    ax.annotate("", xy=(x0 - .012, -.055), xytext=(x1 + .012, -.055),
                arrowprops=dict(arrowstyle="-", color=C_FAINT, lw=.8,
                                shrinkA=0, shrinkB=0))
    for xx in (x0 - .012, x1 + .012):
        ax.plot([xx, xx], [-.055, -.040], color=C_FAINT, lw=.8, zorder=1)
    ax.text((x0 + x1) / 2, -.072,
            f"{len(sim)} embedding / BERTScore metrics\n"
            f"margin {min(p[1] for p in sim):.2f}–{max(p[1] for p in sim):.2f}",
            fontsize=6, color=C_GREY, ha="center", va="top", style="italic",
            linespacing=1.25)

    ax.set_xlabel("Pearson $r$ with the human label (calibrated)")
    ax.set_ylabel("LegalEdit margin")
    ax.set_ylim(-0.21, 0.76); ax.set_xlim(0.18, 0.82)
    ax.set_yticks([0, .2, .4, .6])
    ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.spines["bottom"].set_bounds(0.18, 0.82)
    ax.spines["left"].set_bounds(0.0, 0.70)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#666666")
    # above the axes, so the empty upper-right quadrant stays empty
    ax.legend(fontsize=6.2, frameon=False, loc="lower left", ncol=2,
              bbox_to_anchor=(-0.02, 1.0), handletextpad=.2, labelspacing=.3,
              columnspacing=.8, borderaxespad=0)
    save(fig, "dissociation.pdf")


# ------------------------------------------------------------------- driver --
if __name__ == "__main__":
    for fn in (fig_disagreement, fig_challenge, fig_dissociation):
        try:
            fn()
        except Exception as ex:
            print("skip", fn.__name__, type(ex).__name__, ex)
