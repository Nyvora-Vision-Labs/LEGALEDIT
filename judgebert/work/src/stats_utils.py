"""Shared statistics: bootstrap CIs, correlation comparisons, ordinal agreement."""
import numpy as np
from scipy import stats as sps

RNG = np.random.default_rng(20250829)


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return float(sps.spearmanr(a, b).statistic)


def kendall(a, b):
    return float(sps.kendalltau(a, b).statistic)


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def boot_ci(fn, *arrays, n=10000, alpha=0.05, rng=None):
    """Percentile bootstrap CI of fn(*arrays) resampling item indices."""
    rng = rng or RNG
    arrays = [np.asarray(a) for a in arrays]
    N = len(arrays[0])
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        vals[i] = fn(*[a[idx] for a in arrays])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def boot_diff_test(fn, x1, x2, y, n=10000, rng=None):
    """Paired bootstrap: P(fn(x2,y) >= fn(x1,y)) under resampling.

    Returns (delta, ci_lo, ci_hi, p_two_sided) for fn(x1,y) - fn(x2,y).
    """
    rng = rng or RNG
    x1, x2, y = map(lambda a: np.asarray(a, float), (x1, x2, y))
    N = len(y)
    obs = fn(x1, y) - fn(x2, y)
    d = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        d[i] = fn(x1[idx], y[idx]) - fn(x2[idx], y[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    # two-sided p: fraction of centred bootstrap at least as extreme as 0
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(obs), float(lo), float(hi), float(min(1.0, p))


def williams_test(r12, r13, r23, n):
    """Williams test for dependent correlations sharing variable 1.

    H0: rho12 == rho13. Returns (t, df, p_two_sided).
    """
    R = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
    df = n - 3
    num = (r12 - r13) * np.sqrt(df * (1 + r23))
    den = np.sqrt(2 * R * df / (n - 1) + ((r12 + r13) ** 2 / 4) * (1 - r23) ** 3)
    t = num / den
    p = 2 * (1 - sps.t.cdf(abs(t), df))
    return float(t), int(df), float(p)


def krippendorff_ordinal(matrix):
    """Krippendorff's alpha, ordinal metric. matrix: items x coders, NaN allowed."""
    import krippendorff
    return float(krippendorff.alpha(reliability_data=np.asarray(matrix, float).T,
                                    level_of_measurement="ordinal"))


def krippendorff_interval(matrix):
    import krippendorff
    return float(krippendorff.alpha(reliability_data=np.asarray(matrix, float).T,
                                    level_of_measurement="interval"))


def krippendorff_nominal(matrix):
    """Nominal alpha; string categories are mapped to integer codes."""
    import krippendorff
    m = np.asarray(matrix, object)
    vocab = {v: i for i, v in enumerate(sorted({x for x in m.ravel() if x is not None}))}
    coded = np.array([[vocab.get(x, np.nan) for x in row] for row in m], float)
    return float(krippendorff.alpha(reliability_data=coded.T,
                                    level_of_measurement="nominal"))


def power_for_r(r, n, alpha=0.05):
    """Post-hoc power to detect correlation r at sample size n (Fisher z)."""
    if n < 4:
        return float("nan")
    z = np.arctanh(r) * np.sqrt(n - 3)
    crit = sps.norm.ppf(1 - alpha / 2)
    return float(sps.norm.cdf(z - crit) + sps.norm.cdf(-z - crit))


def min_detectable_r_diff(r_base, n, alpha=0.05, power=0.8):
    """Smallest r2 > r_base detectable vs r_base for independent samples of size n."""
    from scipy.optimize import brentq
    def f(r2):
        se = np.sqrt(2.0 / (n - 3))
        z = (np.arctanh(r2) - np.arctanh(r_base)) / se
        crit = sps.norm.ppf(1 - alpha / 2)
        return sps.norm.cdf(z - crit) - power
    try:
        return float(brentq(f, r_base + 1e-6, 0.999))
    except Exception:
        return float("nan")
