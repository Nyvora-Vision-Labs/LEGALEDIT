"""Canonical splits. Mirrors the paper: 60-10-30 train/val/test, seeds 42..51."""
import numpy as np

SEEDS = list(range(42, 52))


def make_split(n, seed, frac=(0.6, 0.1, 0.3)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a = int(round(frac[0] * n)); b = a + int(round(frac[1] * n))
    return idx[:a], idx[a:b], idx[b:]


def grouped_split(groups, seed, frac=(0.6, 0.1, 0.3), restarts=64):
    """Split so that no source document straddles train, val and test.

    FrJUDGE's 297 pairs come from only 21 source forms, and the three largest
    hold 71% of them (109 / 59 / 44 pairs). Sweeping the forms in random order
    and cutting at the cumulative 60/70% marks -- the obvious construction --
    lets one large form overshoot a boundary and starve the split behind it: on
    seeds 42-46 it produced test splits of 16, 87, 30, 0 and 88 pairs and val
    splits of 0, 4, 0, 117 and 22. A zero-sized or 16-pair test split cannot
    carry a correlation, so we instead treat the assignment as a small
    partitioning problem: random restarts, then local moves of one form at a
    time, scored by total deviation from the target split sizes. Thousands of
    assignments hit 178/30/89 exactly, so the seed still picks a genuinely
    different partition instead of re-deriving a single optimum.

    One consequence is structural and worth stating: the largest form is 37% of
    the corpus, larger than the 30% test quota, so it always lands in train and
    test is always drawn from the remaining 188 pairs.
    """
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    sizes = np.array([(groups == g).sum() for g in uniq], float)
    target = np.array(frac) * sizes.sum()
    rng = np.random.default_rng(seed)

    def cost(who):
        got = np.array([sizes[who == j].sum() for j in range(3)])
        return np.abs(got - target).sum()

    best, best_c = None, np.inf
    for _ in range(restarts):
        who = rng.integers(0, 3, len(uniq))
        c = cost(who)
        moved = True
        while moved:
            moved = False
            for k in range(len(uniq)):
                cur = who[k]
                for j in range(3):
                    if j == cur:
                        continue
                    who[k] = j
                    cj = cost(who)
                    if cj < c - 1e-9:
                        c, cur, moved = cj, j, True
                    else:
                        who[k] = cur
        got = np.array([sizes[who == j].sum() for j in range(3)])
        if c < best_c and got.min() >= 1:
            best, best_c = who.copy(), c

    sel = lambda j: np.where(np.isin(groups, uniq[best == j]))[0]
    return sel(0), sel(1), sel(2)
