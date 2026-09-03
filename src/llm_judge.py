"""Prompted LLM-as-judge scorer, exposed with the metric-zoo signature.

The Limitations section names the absence of an LLM judge as the largest gap in
our comparison: a prompted frontier model given the annotators' own rubric is
the natural competitor to a fine-tuned 112M encoder. This module supplies one,
as a plain `pairs -> np.ndarray` scoring function, so it drops into
`exp2_baselines.py` (corpus correlation) and `exp4_challenge.py` (the LEGALEDIT
diagnostic) beside every other metric with no change to either harness.

Three properties matter for the experiment to be sound:

* **Blinding.** Each pair is scored in its own request, so the judge never sees
  the probe condition, the rule name, or a neighbouring item. This is stronger
  than shuffling a batch: a single-item call carries no condition information at
  all.
* **Self-consistency.** A judge has sampling variance the way training has seed
  variance, so we draw K samples per pair and report their mean. The per-sample
  matrix is kept, which lets us compute the judge's agreement *with itself*
  under the same ordinal Krippendorff coefficient we apply to the five humans.
* **Reproducibility.** Sampling is not deterministic and an API is not a
  fixture, so every response is cached to an append-only JSONL keyed by a hash
  of (model, prompt version, sample index, pair). A rerun re-reads the cache and
  spends nothing; deleting the file forces a fresh draw.
"""
import os, sys, json, time, hashlib, threading, urllib.request, urllib.error
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "results", "llm_cache")

MODEL = os.environ.get("JUDGE_MODEL", "deepseek-chat")
K = int(os.environ.get("JUDGE_K", "5"))
TEMP = float(os.environ.get("JUDGE_TEMP", "1.0"))
WORKERS = int(os.environ.get("JUDGE_WORKERS", "12"))
ENDPOINT = "https://api.deepseek.com/chat/completions"

# Spend ceiling for a whole run. DeepSeek reports token counts rather than a
# dollar figure, so cost is derived from PRICES below and the guard is
# deliberately conservative: it stops issuing calls once the running estimate
# crosses the cap, keeping everything already cached.
BUDGET_USD = float(os.environ.get("JUDGE_BUDGET_USD", "5"))

# USD per 1M tokens (miss / cache-hit input, output). List prices; the reasoner
# entry is rounded up on purpose so the guard errs toward stopping early.
PRICES = {
    "deepseek-chat":     (0.27, 0.07, 1.10),
    "deepseek-reasoner": (0.55, 0.14, 2.19),
}
LEDGER = os.path.join(CACHE_DIR, "spend_ledger.json")


class BudgetExceeded(RuntimeError):
    """Raised when the panel has spent its allowance."""

# Bumping this invalidates the cache. Keep it in the hash so a prompt edit can
# never be silently mixed with responses drawn under the previous wording.
PROMPT_VERSION = "v1"

# The FrJudge rubric as the paper describes it (\S5.2, \S6.1): a ten-point
# ordinal scale for legal meaning preservation, scored by starting at the top
# and deducting once per identified legal error, with the four error types that
# drive the annotation. Simplicity and readability are annotated separately in
# the corpus and are explicitly excluded here, because the label we are
# predicting is the legal-meaning column alone.
SYSTEM = (
    "Vous êtes juriste, spécialiste du droit québécois des assurances et de la "
    "rédaction législative. Vous évaluez si une reformulation simplifiée d'une "
    "clause préserve exactement sa portée juridique."
)

TEMPLATE = """Voici une clause originale et sa version simplifiée.

CLAUSE ORIGINALE:
{original}

VERSION SIMPLIFIÉE:
{simplification}

Évaluez UNIQUEMENT la préservation du sens juridique: la version simplifiée \
dit-elle exactement ce que dit l'originale sur le plan du droit? Ignorez \
complètement la lisibilité, le style et la simplicité — ils sont évalués \
séparément.

Barème (échelle ordinale de 1 à 10):
- Partez de 10 (portée juridique identique).
- Retirez un point par erreur juridique identifiée, parmi:
  * OMISSION: un élément juridiquement opérant de l'originale a disparu
    (condition, exclusion, délai, montant, partie liée, exception).
  * HALLUCINATION: la version simplifiée ajoute un contenu juridique absent
    de l'originale.
  * INCOHÉRENCE: la version simplifiée contredit l'originale (modalité,
    polarité, partie, quantum, portée).
  * CONFUSION: la version simplifiée rend la règle juridiquement ambiguë ou
    inintelligible alors que l'originale ne l'était pas.
- 1 = le sens juridique n'est pas préservé du tout (règle inversée, autre règle,
  ou contenu sans rapport).

Répondez uniquement par un objet JSON, sans texte autour:
{{"erreurs": [{{"type": "...", "detail": "..."}}], "score": <entier de 1 à 10>}}"""


# Prompt variants for the sensitivity experiment. "v1" is the wording every
# earlier result was produced under, and it is registered under exactly the
# string PROMPT_VERSION already carries, so existing cache entries stay valid.
#
#   v1  French instructions, full four-type rubric  (the original)
#   v2  English instructions over the same French text, same rubric semantics
#       -- isolates the language the rubric is written in
#   v3  French, no error taxonomy at all -- isolates how much of the judge's
#       behaviour the rubric detail is actually doing
SYSTEM_EN = (
    "You are a lawyer specialising in Quebec insurance law and legislative "
    "drafting. You assess whether a simplified rewrite of a clause preserves "
    "its legal effect exactly."
)

TEMPLATE_EN = """Here is an original clause and its simplified version.

ORIGINAL CLAUSE:
{original}

SIMPLIFIED VERSION:
{simplification}

Judge ONLY preservation of legal meaning: does the simplified version say \
exactly what the original says as a matter of law? Ignore readability, style \
and simplicity entirely -- they are assessed separately.

Scale (ordinal, 1 to 10):
- Start at 10 (legal effect identical).
- Deduct one point per legal error identified, from:
  * OMISSION: a legally operative element of the original is gone
    (condition, exclusion, deadline, amount, bound party, exception).
  * HALLUCINATION: the simplified version adds legal content absent from
    the original.
  * INCOHERENCE: the simplified version contradicts the original (modality,
    polarity, party, quantum, scope).
  * CONFUSION: the simplified version makes the rule legally ambiguous or
    unintelligible where the original was not.
- 1 = the legal meaning is not preserved at all (rule reversed, a different
  rule, or unrelated content).

Reply only with a JSON object, no surrounding text:
{{"erreurs": [{{"type": "...", "detail": "..."}}], "score": <integer 1 to 10>}}"""

TEMPLATE_BARE = """Voici une clause originale et sa version simplifiée.

CLAUSE ORIGINALE:
{original}

VERSION SIMPLIFIÉE:
{simplification}

Sur une échelle de 1 à 10, dans quelle mesure la version simplifiée préserve-t-elle \
le sens juridique de la clause originale? 10 = portée juridique identique, \
1 = sens juridique pas préservé du tout. Ignorez la lisibilité et le style.

Répondez uniquement par un objet JSON, sans texte autour:
{{"erreurs": [], "score": <entier de 1 à 10>}}"""

PROMPTS = {
    "v1": (SYSTEM, TEMPLATE),
    "v2": (SYSTEM_EN, TEMPLATE_EN),
    "v3": (SYSTEM, TEMPLATE_BARE),
}


# ------------------------------------------------------------------ plumbing --
def _api_key(model=MODEL):
    var = "DEEPSEEK_API_KEY"
    k = os.environ.get(var)
    if not k:  # fall back to a .env above src/ so scripts run without sourcing
        d = HERE
        for _ in range(4):
            d = os.path.dirname(d)
            env = os.path.join(d, ".env")
            if os.path.exists(env):
                for line in open(env, encoding="utf-8"):
                    if line.startswith(var + "="):
                        k = line.split("=", 1)[1].strip()
                break
    if not k:
        raise RuntimeError("%s not set and not found in .env" % var)
    return k


def available(model=MODEL):
    try:
        _api_key(model); return True
    except RuntimeError:
        return False


_ledger_lock = threading.Lock()


def call_cost(model, usage):
    """Estimated USD for one call from its reported token counts."""
    miss, hit, out = PRICES.get(model, PRICES["deepseek-reasoner"])
    ph = usage.get("prompt_cache_hit_tokens", 0) or 0
    pm = (usage.get("prompt_tokens", 0) or 0) - ph
    return (pm * miss + ph * hit + (usage.get("completion_tokens", 0) or 0) * out) / 1e6


def spend(add=0.0, model=None):
    """Read, or add to, the cumulative estimated spend. Returns the total."""
    with _ledger_lock:
        d = {}
        if os.path.exists(LEDGER):
            try:
                d = json.load(open(LEDGER))
            except json.JSONDecodeError:
                d = {}
        if add:
            os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
            d[model] = round(d.get(model, 0.0) + add, 6)
            d["_total"] = round(d.get("_total", 0.0) + add, 6)
            json.dump(d, open(LEDGER, "w"), indent=1)
        return d.get("_total", 0.0)





def _hash(model, prompt_version, sample, original, simplification):
    h = hashlib.sha256()
    for part in (model, prompt_version, str(sample), original, simplification):
        h.update(part.encode("utf-8")); h.update(b"\x00")
    return h.hexdigest()


class _Cache:
    """Append-only JSONL. Survives a kill mid-run; a rerun costs nothing."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.d = {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line); self.d[r["h"]] = r
                except json.JSONDecodeError:
                    continue          # truncated final line from a hard kill

    def get(self, h):
        return self.d.get(h)

    def put(self, rec):
        with self.lock:
            self.d[rec["h"]] = rec
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _post(key, body, timeout=180):
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _parse(txt):
    """Pull (score, errors) out of the reply. Returns (None, []) if unusable."""
    t = txt.strip()
    if t.startswith("```"):                      # strip a ```json fence
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    try:
        o = json.loads(t[t.index("{"):t.rindex("}") + 1])
        s = int(round(float(o["score"])))
        if 1 <= s <= 10:
            errs = o.get("erreurs") or o.get("errors") or []
            return s, (errs if isinstance(errs, list) else [])
    except (ValueError, KeyError, TypeError):
        pass
    return None, []


def _one(key, cache, sample, pair, model, temp, retries=5, prompt="v1"):
    h = _hash(model, prompt, sample, pair["original"], pair["simplification"])
    hit = cache.get(h)
    if hit is not None:
        return hit
    # The reasoner spends completion tokens on its chain of thought before it
    # emits any content, so a 400-token cap yields an empty message. It needs
    # room for the reasoning plus the JSON; those tokens are billed as output,
    # which is why the reasoner runs on a smaller item set.
    body = dict(model=model, temperature=temp,
                max_tokens=3000 if "reason" in model else 400,
                messages=[{"role": "system", "content": PROMPTS[prompt][0]},
                          {"role": "user",
                           "content": PROMPTS[prompt][1].format(**pair)}])
    if model == "deepseek-chat":
        # The chat model supports structured output. The reasoner does not take
        # the flag the same way, so it relies on the prompt's JSON instruction
        # plus the tolerant parser instead of failing outright.
        body["response_format"] = {"type": "json_object"}
    if spend() >= BUDGET_USD:
        raise BudgetExceeded("spent $%.2f of $%.2f" % (spend(), BUDGET_USD))
    delay = 2.0
    for attempt in range(retries):
        try:
            r = _post(key, body)
            txt = r["choices"][0]["message"]["content"]
            score, errs = _parse(txt)
            if score is None:
                raise ValueError("unparseable: " + txt[:120])
            u = r.get("usage", {}) or {}
            spend(call_cost(model, u), model)
            rec = dict(h=h, score=score, errors=errs, sample=sample, model=model,
                       usage=u)
            cache.put(rec)
            return rec
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError,
                KeyError, TimeoutError, OSError) as e:
            if attempt == retries - 1:
                rec = dict(h=h, score=None, errors=[], sample=sample, model=model,
                           usage={}, fail=type(e).__name__ + ": " + str(e)[:200])
                cache.put(rec)                   # cache the failure, not the retry storm
                return rec
            time.sleep(delay); delay *= 2


def score_matrix(pairs, k=None, model=None, temp=None, tag="judge", verbose=True,
                 prompt="v1"):
    """Return an (n_pairs, k) array of raw 1-10 judgments. NaN where a call failed."""
    from concurrent.futures import ThreadPoolExecutor
    k = K if k is None else k
    model = MODEL if model is None else model
    temp = TEMP if temp is None else temp
    key = _api_key(model)
    safe = model.replace("/", "__").replace(":", "-")
    cache = _Cache(os.path.join(CACHE_DIR, "%s_%s_%s.jsonl" % (tag, safe, prompt)))

    jobs = [(s, i) for s in range(k) for i in range(len(pairs))]
    todo = sum(1 for s, i in jobs
               if cache.get(_hash(model, prompt, s, pairs[i]["original"],
                                  pairs[i]["simplification"])) is None)
    if verbose:
        print("  [judge] %s: %d pairs x k=%d, %d cached, %d to fetch"
              % (tag, len(pairs), k, len(jobs) - todo, todo), flush=True)

    out = np.full((len(pairs), k), np.nan)
    done = [0]
    prog = threading.Lock()

    stopped = [False]

    def work(job):
        s, i = job
        try:
            rec = _one(key, cache, s, pairs[i], model, temp, prompt=prompt)
        except BudgetExceeded:
            stopped[0] = True
            rec = dict(score=None)          # not cached: it never reached the API
        with prog:
            done[0] += 1
            if verbose and done[0] % 250 == 0:
                print("    %d/%d  ($%.2f)" % (done[0], len(jobs), spend()), flush=True)
        return i, s, rec

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, s, rec in ex.map(work, jobs):
            if rec.get("score") is not None:
                out[i, s] = rec["score"]
    if verbose:
        nf = int(np.isnan(out).sum())
        print("  [judge] %s done; %d/%d calls unusable" % (tag, nf, out.size), flush=True)
    if stopped[0]:
        raise BudgetExceeded(
            "budget ceiling $%.2f reached during '%s'; %d of %d calls were not made. "
            "Cached work is kept -- raise JUDGE_BUDGET_USD to continue."
            % (BUDGET_USD, tag, int(np.isnan(out).sum()), out.size))
    return out


def deepseek_score(pairs, k=None, model=None, temp=None, tag="judge", verbose=True):
    """Metric-zoo signature: one score per pair, higher = more meaning preserved.

    The reported score is the mean over K samples (self-consistency). A pair
    whose samples all failed falls back to the corpus mean so it cannot silently
    become an outlier; failures are reported by `score_matrix`.
    """
    M = score_matrix(pairs, k=k, model=model, temp=temp, tag=tag, verbose=verbose)
    with np.errstate(invalid="ignore"):
        v = np.nanmean(M, axis=1)
    if np.isnan(v).any():
        v = np.where(np.isnan(v), np.nanmean(v), v)
    return v


def records(pairs, k=None, model=None, tag="judge"):
    """Cached response records per pair: list (len n) of lists (len k).

    Used for the error-type analysis -- what the judge *said* was wrong, not
    just the number it produced. Only reads the cache; never calls the API.
    """
    k = K if k is None else k
    model = MODEL if model is None else model
    cache = _Cache(os.path.join(CACHE_DIR, "%s_%s_%s.jsonl" % (tag, model.replace("/", "__").replace(":", "-"), PROMPT_VERSION)))
    out = []
    for p in pairs:
        out.append([cache.get(_hash(model, PROMPT_VERSION, s, p["original"],
                                    p["simplification"])) for s in range(k)])
    return out


def usage_summary(tag="judge", model=None):
    """Token/҇call totals for the cost note in the paper."""
    model = MODEL if model is None else model
    p = os.path.join(CACHE_DIR, "%s_%s_%s.jsonl" % (tag, model.replace("/", "__").replace(":", "-"), PROMPT_VERSION))
    if not os.path.exists(p):
        return {}
    n = pin = pout = cached = fail = 0
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        if r.get("score") is None:
            fail += 1
        u = r.get("usage") or {}
        pin += u.get("prompt_tokens", 0); pout += u.get("completion_tokens", 0)
        cached += u.get("prompt_cache_hit_tokens", 0)
    return dict(calls=n, failed=fail, prompt_tokens=pin, completion_tokens=pout,
                prompt_cache_hit_tokens=cached)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    from data import load_pairs
    P = load_pairs()[:4]
    M = score_matrix(P, k=2, tag="smoke")
    for p, row in zip(P, M):
        print(round(p["label"], 1), row, p["original"][:70])
