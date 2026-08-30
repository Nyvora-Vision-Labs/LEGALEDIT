"""JUDGEBERT reproduction plus the variants introduced in this work."""
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
BASE = "almanach/camembertv2-base"


class JudgeModel(nn.Module):
    """Shared encoder + task-specific head.

    head:
      'scalar' -> 1 output, MSE on the mean label            (JUDGEBERT repro)
      'dist'   -> 10 logits, KL against the empirical rating histogram
      'annot'  -> 5 outputs, MSE against each annotator      (annotator-aware)
      'quant'  -> 1 output, pinball loss at `tau`            (conservative)
    aux_classes > 0 adds a characterization classification head.
    """

    def __init__(self, head="scalar", aux_classes=0, base=BASE, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.head_type = head
        n_out = {"scalar": 1, "dist": 10, "annot": 5, "quant": 1}[head]
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h, n_out)
        self.aux = nn.Linear(h, aux_classes) if aux_classes else None

    def forward(self, **b):
        z = self.drop(self.enc(**b).last_hidden_state[:, 0])
        return self.head(z), (self.aux(z) if self.aux is not None else None)

    @torch.no_grad()
    def score(self, out):
        """Map head output to a single 1-10 legal-meaning score."""
        if self.head_type == "dist":
            p = out.softmax(-1)
            return (p * torch.arange(1, 11, device=p.device).float()).sum(-1)
        if self.head_type == "annot":
            return out.mean(-1)
        return out.squeeze(-1)


def pinball(pred, target, tau):
    d = target - pred
    return torch.maximum(tau * d, (tau - 1) * d).mean()


def soft_hist(ratings):
    """5 integer ratings -> length-10 probability vector."""
    v = np.zeros(10, float)
    for r in ratings:
        v[int(r) - 1] += 1
    return v / v.sum()


# Pad every batch up to one of a few fixed lengths rather than to the longest
# member. Pair lengths span 30-417 tokens, so per-batch padding produced a new
# tensor shape almost every step; MPS compiles a kernel per distinct shape, and
# the accumulating graph cache is what made epochs slow from ~10s to ~100s over
# a single run. Seven buckets cap the number of distinct shapes. Padding is
# masked out by the attention mask, so this does not change any result.
_BUCKETS = (64, 96, 128, 160, 192, 224, 256)


class Batcher:
    def __init__(self, base=BASE, max_len=256, buckets=_BUCKETS):
        self.tok = AutoTokenizer.from_pretrained(base)
        self.max_len = max_len
        self.buckets = tuple(b for b in buckets if b <= max_len)

    def __call__(self, pairs):
        enc = self.tok([p[0] for p in pairs], [p[1] for p in pairs],
                       truncation=True, max_length=self.max_len)
        longest = max(len(x) for x in enc["input_ids"])
        target = next((b for b in self.buckets if b >= longest), self.max_len)
        return self.tok.pad(enc, padding="max_length", max_length=target,
                            return_tensors="pt")
