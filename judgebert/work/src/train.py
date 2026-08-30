"""Training / evaluation loop shared by every JUDGEBERT variant."""
import time
import numpy as np, torch, torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model import JudgeModel, Batcher, pinball, soft_hist, DEV

_MPS = torch.backends.mps.is_available()


def _forward_loss(m, out, aux_out, batch_y, aux_y, head, tau, aux_w):
    if head == "scalar":
        loss = nn.functional.mse_loss(out.squeeze(-1), batch_y["mean"])
    elif head == "quant":
        loss = pinball(out.squeeze(-1), batch_y["mean"], tau)
    elif head == "annot":
        loss = nn.functional.mse_loss(out, batch_y["annot"])
    elif head == "dist":
        loss = nn.functional.kl_div(out.log_softmax(-1), batch_y["dist"], reduction="batchmean")
    if aux_out is not None and aux_y is not None:
        mask = aux_y >= 0
        if mask.any():
            loss = loss + aux_w * nn.functional.cross_entropy(aux_out[mask], aux_y[mask])
    return loss


def make_targets(items, device):
    """items: list of dicts with keys text_a, text_b, mean, ratings, aux."""
    y = {"mean": torch.tensor([it["mean"] for it in items], dtype=torch.float, device=device)}
    y["annot"] = torch.tensor(np.array([it["ratings"] for it in items]), dtype=torch.float, device=device)
    y["dist"] = torch.tensor(np.array([soft_hist(it["ratings"]) for it in items]),
                             dtype=torch.float, device=device)
    aux = torch.tensor([it.get("aux", -1) for it in items], dtype=torch.long, device=device)
    return y, aux


def _bucketed_order(items, rng, bs, window=8):
    """Shuffle, then length-sort inside windows of `window` batches.

    Cuts padding waste (pair lengths range 30-417 tokens) without making the
    batch composition deterministic across epochs.
    """
    idx = rng.permutation(len(items))
    W = bs * window
    out = []
    for i in range(0, len(idx), W):
        chunk = sorted(idx[i:i + W], key=lambda j: len(items[j]["text_a"]) + len(items[j]["text_b"]))
        out.extend(chunk)
    blocks = [out[i:i + bs] for i in range(0, len(out), bs)]
    rng.shuffle(blocks)
    return [j for b in blocks for j in b]


def train_one(train_items, val_items, head="scalar", aux_classes=0, tau=0.25,
              aux_w=0.3, seed=42, lr=5e-5, bs=16, max_epochs=60, patience=5,
              verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    m = JudgeModel(head=head, aux_classes=aux_classes).to(DEV)
    btc = Batcher()
    opt = AdamW(m.parameters(), lr=lr)
    steps = max(1, (len(train_items) + bs - 1) // bs) * max_epochs
    sch = get_linear_schedule_with_warmup(opt, 0, steps)
    best, best_state, bad = float("inf"), None, 0
    t_start = time.time()
    rng = np.random.default_rng(seed)
    for ep in range(max_epochs):
        m.train()
        order = _bucketed_order(train_items, rng, bs)
        for i in range(0, len(order), bs):
            items = [train_items[j] for j in order[i:i + bs]]
            b = btc([(it["text_a"], it["text_b"]) for it in items]).to(DEV)
            y, aux = make_targets(items, DEV)
            out, aux_out = m(**b)
            loss = _forward_loss(m, out, aux_out, y, aux, head, tau, aux_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
        vl = evaluate_loss(m, btc, val_items, head, tau, aux_w)
        if _MPS:
            torch.mps.empty_cache()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 5 == 0:
            print(f"    ep{ep:3d} val={vl:.4f} best={best:.4f} "
                  f"({time.time() - t_start:.0f}s elapsed)", flush=True)
    if best_state:
        m.load_state_dict(best_state)
    # Release the optimiser moments (2x model size) and the CPU checkpoint before
    # the caller runs inference; MPS is slow to reclaim these on its own.
    del opt, sch, best_state
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return m, btc, ep + 1


@torch.no_grad()
def evaluate_loss(m, btc, items, head, tau, aux_w, bs=32):
    m.eval(); tot, n = 0.0, 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        b = btc([(it["text_a"], it["text_b"]) for it in chunk]).to(DEV)
        y, aux = make_targets(chunk, DEV)
        out, aux_out = m(**b)
        tot += float(_forward_loss(m, out, aux_out, y, aux, head, tau, aux_w)) * len(chunk)
        n += len(chunk)
    return tot / max(1, n)


@torch.no_grad()
def predict(m, btc, texts, bs=32):
    """texts: list of (a, b). Returns 1-10 scores."""
    m.eval(); out = []
    for i in range(0, len(texts), bs):
        b = btc(texts[i:i + bs]).to(DEV)
        o, _ = m(**b)
        out.extend(m.score(o).float().cpu().tolist())
    return np.clip(np.array(out), 1, 10)
