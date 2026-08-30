"""Figures for the paper."""
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
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})

C_STRICT, C_LEN = "#B4432A", "#2A6FB4"


def fig_disagreement():
    P = load_pairs()
    M = np.array([[p["ratings"][a] for a in ANNOTATORS] for p in P], float)
    means = M.mean(0)
    fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.5))
    # (a) per-annotator rating histograms
    for i, a in enumerate(ANNOTATORS):
        c = C_LEN if means[i] > 6 else C_STRICT
        h = np.bincount(M[:, i].astype(int), minlength=11)[1:]
        ax[0].plot(range(1, 11), h, marker="o", ms=3, lw=1.2, color=c, alpha=.85,
                   label=f"{a} ({means[i]:.1f})")
    ax[0].set_xlabel("legal meaning rating"); ax[0].set_ylabel("items")
    ax[0].set_title("(a) two annotator populations", fontsize=9)
    ax[0].legend(fontsize=6, frameon=False, ncol=2)
    # (b) spread per item
    rng = np.sort(M.max(1) - M.min(1))
    ax[1].fill_between(np.arange(len(rng)) / len(rng) * 100, rng, step="pre",
                       color="#888", alpha=.5)
    ax[1].axhline(6, ls="--", lw=1, color=C_STRICT)
    ax[1].set_xlabel("percentile of items"); ax[1].set_ylabel("max $-$ min rating")
    ax[1].set_title("(b) within-item disagreement", fontsize=9)
    ax[1].text(3, 6.4, f"{(rng>=6).mean()*100:.0f}% span $\\geq$6 pts", fontsize=7, color=C_STRICT)
    # (c) ceiling vs reported
    e = json.load(open(os.path.join(RES, "exp1_ceiling.json")))
    loo = [v["r"] for v in e["leave_one_annotator_out"].values()]
    sb = np.sqrt(e["spearman_brown_reliability_k5"])
    ax[2].barh(["single annotator\n(ceiling)", "best annotator\npair"],
               [e["ceiling_r"], e["best_human_pair"]["r"]], color="#888", height=.5)
    ax[2].errorbar([e["ceiling_r"]], [0], xerr=[[e["ceiling_r"]-min(loo)], [max(loo)-e["ceiling_r"]]],
                   fmt="none", ecolor="k", capsize=2, lw=.8)
    # the two reference lines the section argues should both be reported
    ax[2].axvline(0.74, color=C_STRICT, lw=1.5)
    ax[2].axvline(sb, color="#555", lw=1.1, ls=(0, (3, 2)))
    ax[2].set_ylim(-0.6, 1.9)
    ax[2].annotate("JUDGEBERT\nreported\n0.74", xy=(0.74, 1.45), xytext=(0.715, 1.45),
                   fontsize=6.5, color=C_STRICT, ha="right", va="center", linespacing=1.25)
    ax[2].annotate("aggregate\nallows %.2f" % sb, xy=(sb, 1.45), xytext=(sb + 0.025, 1.45),
                   fontsize=6.5, color="#555", ha="left", va="center", linespacing=1.25)
    for j, v in enumerate([e["ceiling_r"], e["best_human_pair"]["r"]]):
        ax[2].text(0.02, j, f"{v:.3f}", fontsize=7, color="white", va="center", ha="left",
                   fontweight="bold")
    ax[2].set_xlim(0, 1.0); ax[2].set_xlabel("Pearson $r$ with consensus")
    ax[2].set_title("(c) the human ceiling", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "disagreement.pdf"), bbox_inches="tight")
    print("wrote disagreement.pdf")


def fig_challenge():
    f = os.path.join(RES, "exp4_challenge.json")
    if not os.path.exists(f):
        print("skip challenge fig"); return
    c = json.load(open(f))["rows"]
    order = sorted(c, key=lambda k: -c[k]["margin_frac"])
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(order) + 1.0))
    yy = np.arange(len(order))
    for j, k in enumerate(order):
        v = c[k]
        ax.plot([v["unrelated"], v["identical"]], [j, j], color="#ccc", lw=3, zorder=1,
                solid_capstyle="round")
        ax.scatter(v["unrelated"], j, s=22, color="#444", zorder=3)
        ax.scatter(v["identical"], j, s=22, color=C_LEN, zorder=3)
        ax.scatter(v["legaledit"], j, s=52, marker="D", color=C_STRICT, zorder=4)
    # margin value at the right edge -- the number the caption talks about
    for j, k in enumerate(order):
        ax.text(1.045, j, f"{c[k]['margin_frac']:.2f}", fontsize=7, va="center",
                ha="left", color="#222",
                fontweight="bold" if c[k]["margin_frac"] > 0.2 else "normal")
    ax.text(1.045, -0.85, "margin", fontsize=7, va="center", ha="left", style="italic")

    # separate the family that reacts to the edit from the family that does not
    cut = sum(1 for k in order if c[k]["margin_frac"] > 0.2)
    if 0 < cut < len(order):
        ax.axhline(cut - 0.5, color="#999", lw=.8, ls=(0, (4, 3)), zorder=2)
        ax.text(0.56, cut - 0.5, "entailment-based  $\\uparrow$   /   "
                                 "$\\downarrow$  similarity-based",
                fontsize=6.5, color="#555", style="italic", ha="center", va="center",
                zorder=5, bbox=dict(fc="white", ec="none", pad=1.5))

    ax.set_yticks(yy); ax.set_yticklabels(order, fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel("normalised score")
    ax.set_xlim(-0.03, 1.04)
    ax.scatter([], [], s=22, color=C_LEN, label="identical pair")
    ax.scatter([], [], s=52, marker="D", color=C_STRICT, label="minimal legal edit")
    ax.scatter([], [], s=22, color="#444", label="unrelated pair")
    ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "challenge.pdf"), bbox_inches="tight")
    print("wrote challenge.pdf")


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
    SHORT = {"NLI-bidirectional (min)": "NLI-bidirectional",
             "NLI-fwd (no hallucination)": "NLI-fwd",
             "NLI-bwd (no omission)": "NLI-bwd",
             "surface: token Jaccard": "token Jaccard",
             "surface: -|len diff|": "$-|\\,|s|-|o|\\,|$"}
    pts = []
    for k, v in C.items():
        b = B.get(ALIAS.get(k, k))
        if not b:
            continue
        r = float(np.mean([d["r"] for d in b["per_seed_cal"]]))
        fam = "nli" if "NLI" in k else ("surface" if k.startswith("surface") else "sim")
        pts.append((k, r, v["margin_frac"], fam))

    COL = {"nli": "#1F7A5A", "sim": C_LEN, "surface": C_STRICT}
    LBL = {"nli": "entailment", "sim": "embedding / BERTScore", "surface": "surface feature"}
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for fam in ("sim", "surface", "nli"):
        xs = [p[1] for p in pts if p[3] == fam]; ys = [p[2] for p in pts if p[3] == fam]
        ax.scatter(xs, ys, s=44, color=COL[fam], label=LBL[fam], zorder=3,
                   edgecolor="white", linewidth=.8)

    # Label the points that carry the argument individually. The seven
    # embedding metrics sit on top of each other along the floor, so they are
    # bracketed as a group rather than labelled into an unreadable pile.
    for k, r, m, fam in pts:
        if k in SHORT:
            ax.annotate(SHORT[k], (r, m), fontsize=7, color="#222", ha="center",
                        va="bottom", xytext=(0, 8), textcoords="offset points")
    sim = [(r, m) for _, r, m, f in pts if f == "sim"]
    x0, x1 = min(p[0] for p in sim), max(p[0] for p in sim)
    ax.annotate("", xy=(x0 - .012, -.045), xytext=(x1 + .012, -.045),
                arrowprops=dict(arrowstyle="-", color="#888", lw=.9,
                                shrinkA=0, shrinkB=0))
    for xx in (x0 - .012, x1 + .012):
        ax.plot([xx, xx], [-.045, -.032], color="#888", lw=.9, zorder=1)
    ax.text((x0 + x1) / 2, -.062, f"{len(sim)} embedding / BERTScore metrics: "
            f"margin {min(p[1] for p in sim):.2f}\u2013{max(p[1] for p in sim):.2f}",
            fontsize=6.8, color="#555", ha="center", va="top", style="italic")

    ax.set_xlabel("Pearson $r$ with the human label on FrJudge (calibrated)")
    ax.set_ylabel("LegalEdit margin")
    ax.set_ylim(-0.13, 0.82); ax.set_xlim(0.18, 0.70)
    ax.set_yticks([0, .2, .4, .6, .8])
    ax.spines["bottom"].set_bounds(0.18, 0.70)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "dissociation.pdf"), bbox_inches="tight")
    print("wrote dissociation.pdf")


# ------------------------------------------------------------------- driver --
if __name__ == "__main__":
    for fn in (fig_disagreement, fig_challenge, fig_dissociation):
        try:
            fn()
        except Exception as ex:
            print("skip", fn.__name__, type(ex).__name__, ex)
