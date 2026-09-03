"""Experiment 3: JUDGEBERT reproduction + variants, with honest evaluation.

Evaluation protocol (differs from the original paper on purpose):
  * correlation/RMSE are ALWAYS measured on the human-annotated test split only,
    never on a split containing augmentation. Augmentation affects training only.
  * the three diagnostic regimes (identical / unrelated / legaledit) are reported
    separately as pass rates, never mixed into the correlation.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_pairs, ANNOTATORS
from splits import make_split, grouped_split, SEEDS
from augment import unrelated_pairs, legal_edits, tier
from train import train_one, predict
import stats_utils as su

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)

P = load_pairs()
CHARS = sorted({p["char_major"] for p in P if p["char_major"]})
CIDX = {c: i for i, c in enumerate(CHARS)}


def base_items():
    return [dict(text_a=p["original"], text_b=p["simplification"], mean=p["label"],
                 ratings=[p["ratings"][a] for a in ANNOTATORS],
                 aux=CIDX.get(p["char_major"], -1)) for p in P]


def aug_items(kind, seed, n):
    """Augmentation pools. `mean`/`ratings` are the definitional gold for each."""
    out = []
    if kind in ("identical", "da", "da+edit"):
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(P), size=min(n, len(P)), replace=False)
        for i in pick:
            s = P[i]["original"]
            out.append(dict(text_a=s, text_b=s, mean=10.0, ratings=[10] * 5, aux=-1))
    if kind in ("unrelated", "da", "da+edit"):
        for a, b in unrelated_pairs(n, seed=seed):
            out.append(dict(text_a=a, text_b=b, mean=1.0, ratings=[1] * 5, aux=-1))
    if kind == "da+edit":
        ed = legal_edits([p["original"] for p in P], seed=seed, per_sentence=1)
        rng = np.random.default_rng(seed + 1)
        idx = rng.choice(len(ed), size=min(n, len(ed)), replace=False)
        for j in idx:
            o, pert, rule = ed[j]
            out.append(dict(text_a=o, text_b=pert, mean=1.0, ratings=[1] * 5, aux=-1))
    return out


def diagnostics(m, btc, seed):
    """Held-out identical / unrelated / legaledit probes (never trained on)."""
    rng = np.random.default_rng(1000 + seed)
    # identical: held-out reference sentences, not FrJUDGE originals
    from augment import _sents
    ref = _sents("A-25.jsonl") + _sents("code_securite_routiere.jsonl")
    ridx = rng.choice(len(ref), size=min(297, len(ref)), replace=False)
    ident = [(ref[i], ref[i]) for i in ridx]
    unrel = unrelated_pairs(297, seed=5000 + seed)
    ed = legal_edits([ref[i] for i in ridx], seed=seed, per_sentence=1)
    d = {}
    s = predict(m, btc, ident)
    d["identical_mean"] = float(s.mean()); d["identical_pass"] = float((s >= 9.9).mean())
    s = predict(m, btc, [(a, b) for a, b in unrel])
    d["unrelated_mean"] = float(s.mean()); d["unrelated_pass"] = float((s <= 1.1).mean())
    if ed:
        se = predict(m, btc, [(o, p) for o, p, _ in ed])
        si = predict(m, btc, [(o, o) for o, _, _ in ed])
        t1 = np.array([tier(r) == 1 for _, _, r in ed])
        d["legaledit_mean"] = float(se.mean())
        d["legaledit_mean_t1"] = float(se[t1].mean()) if t1.any() else float("nan")
        d["legaledit_false_accept"] = float((se >= 7).mean())      # scored "accurate"
        d["legaledit_margin"] = float((si - se).mean())            # identical - perturbed
        d["legaledit_discrim"] = float((si > se + 1.0).mean())     # ranks correctly
    return d


CONFIGS = {
    "JudgeBERT (repro)":            dict(head="scalar", aug=None),
    "JudgeBERT-DA (repro)":         dict(head="scalar", aug="da"),
    "JudgeBERT-DA+LegalEdit":       dict(head="scalar", aug="da+edit"),
    "JudgeBERT-Dist (soft labels)": dict(head="dist",   aug=None),
    "JudgeBERT-Annot (5 heads)":    dict(head="annot",  aug=None),
    "JudgeBERT-MT (+charact.)":     dict(head="scalar", aug=None, aux=True),
    "JudgeBERT-Quantile (tau=.25)": dict(head="quant",  aug=None),
}


def run(names, seeds, grouped=False, tag=""):
    items = base_items()
    y_all = np.array([it["mean"] for it in items])
    groups = [p["source"] for p in P]
    res = {}
    for name in names:
        cfg = CONFIGS[name]
        per_seed = []
        for sd in seeds:
            t0 = time.time()
            if grouped:
                tr, va, te = grouped_split(groups, sd)
            else:
                tr, va, te = make_split(len(P), sd)
            train_items = [items[i] for i in tr]
            if cfg["aug"]:
                train_items = train_items + aug_items(cfg["aug"], sd, len(tr))
            m, btc, eps = train_one(train_items, [items[i] for i in va],
                                    head=cfg["head"],
                                    aux_classes=len(CHARS) if cfg.get("aux") else 0,
                                    seed=sd, verbose=os.environ.get("EXP3_VERBOSE") == "1")
            pr = predict(m, btc, [(items[i]["text_a"], items[i]["text_b"]) for i in te])
            yt = y_all[te]
            rec = dict(seed=sd, epochs=eps, r=su.pearson(pr, yt), rho=su.spearman(pr, yt),
                       rmse=su.rmse(pr, yt), over=float((pr > yt).mean()),
                       secs=time.time() - t0)
            rec.update(diagnostics(m, btc, sd))
            rec["preds"] = pr.tolist(); rec["test_idx"] = te.tolist()
            per_seed.append(rec)
            print(f"  [{name}] seed={sd} ep={eps} r={rec['r']:.3f} rho={rec['rho']:.3f} "
                  f"RMSE={rec['rmse']:.2f} over={rec['over']:.0%} | id={rec['identical_pass']:.2f} "
                  f"un={rec['unrelated_pass']:.2f} edit_FA={rec.get('legaledit_false_accept',float('nan')):.2f} "
                  f"({rec['secs']:.0f}s)", flush=True)
            del m
            import torch, gc; gc.collect(); torch.mps.empty_cache()
        res[name] = per_seed
        f = os.path.join(OUT, f"exp3_train{tag}.json")
        prev = json.load(open(f)) if os.path.exists(f) else {}
        # merge by seed so the grid can be driven one process per (config, seed)
        # -- MPS does not reliably release memory between runs inside one process.
        for k, v in res.items():
            fresh = {d["seed"] for d in v}
            merged = [d for d in prev.get(k, []) if d["seed"] not in fresh] + v
            prev[k] = sorted(merged, key=lambda d: d["seed"])
        json.dump(prev, open(f, "w"), indent=2, default=float)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="all")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--grouped", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    names = list(CONFIGS) if a.configs == "all" else a.configs.split("|")
    run(names, [int(s) for s in a.seeds.split(",")], grouped=a.grouped, tag=a.tag)
