"""Reference-free similarity/adequacy metrics, all French-capable.

Every metric returns one score per pair, higher = more meaning preserved.
Raw scales differ; calibration is handled downstream.
"""
import os, numpy as np, torch

# METRICS_DEVICE lets the metric zoo be pinned to the CPU so a re-computation
# can run alongside a training job without contending for the same MPS queue.
DEV = os.environ.get("METRICS_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
_cache = {}


def _enc(name):
    if name not in _cache:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModel.from_pretrained(name).to(DEV).eval()
        _cache[name] = (tok, mod)
    return _cache[name]


@torch.no_grad()
def _token_embeds(name, texts, bs=16):
    tok, mod = _enc(name)
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                truncation=True, max_length=256).to(DEV)
        h = mod(**b).last_hidden_state
        for j in range(h.shape[0]):
            m = b["attention_mask"][j].bool()
            out.append(torch.nn.functional.normalize(h[j][m][1:-1], dim=-1).float().cpu())
    return out


def bertscore_f1(pairs, name):
    """Greedy-matching BERTScore F1 with the given encoder (no idf, no rescale)."""
    O = _token_embeds(name, [p["original"] for p in pairs])
    S = _token_embeds(name, [p["simplification"] for p in pairs])
    sc = []
    for o, s in zip(O, S):
        if len(o) == 0 or len(s) == 0:
            sc.append(0.0); continue
        M = o @ s.T
        R = M.max(dim=1).values.mean().item()   # recall  over original tokens
        P = M.max(dim=0).values.mean().item()   # precision over simplified tokens
        sc.append(0.0 if P + R == 0 else 2 * P * R / (P + R))
    return np.array(sc)


def bertscore_recall(pairs, name):
    """Recall only -- directly targets omission, the dominant legal error."""
    O = _token_embeds(name, [p["original"] for p in pairs])
    S = _token_embeds(name, [p["simplification"] for p in pairs])
    return np.array([(o @ s.T).max(dim=1).values.mean().item() if len(o) and len(s) else 0.0
                     for o, s in zip(O, S)])


def sbert_cosine(pairs, name, prefix=""):
    from sentence_transformers import SentenceTransformer
    key = "st:" + name
    if key not in _cache:
        _cache[key] = SentenceTransformer(name, device=DEV)
    m = _cache[key]
    a = m.encode([prefix + p["original"] for p in pairs], convert_to_numpy=True,
                 normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    b = m.encode([prefix + p["simplification"] for p in pairs], convert_to_numpy=True,
                 normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    return (a * b).sum(1)


@torch.no_grad()
def nli_bidirectional(pairs, name="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                      device="cpu"):
    """Return (fwd, bwd, min) entailment probabilities.

    fwd = P(original entails simplification)   -> low means hallucination
    bwd = P(simplification entails original)   -> low means omission
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    key = "nli:" + name
    if key not in _cache:
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        _cache[key] = (tok, mod)
    tok, mod = _cache[key]
    lab = {v.lower(): k for k, v in mod.config.id2label.items()}
    ent = lab.get("entailment", 0)

    def run(prem, hyp):
        out = []
        for i in range(0, len(prem), 16):
            b = tok(prem[i:i + 16], hyp[i:i + 16], return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to(device)
            p = mod(**b).logits.softmax(-1)[:, ent]
            out.extend(p.float().cpu().tolist())
        return np.array(out)

    o = [p["original"] for p in pairs]
    s = [p["simplification"] for p in pairs]
    fwd, bwd = run(o, s), run(s, o)
    return fwd, bwd, np.minimum(fwd, bwd)
