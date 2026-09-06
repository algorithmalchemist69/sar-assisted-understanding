"""Training of the classification heads. The encoders never train.

Two regimes, because the assignment's literal baseline and the fair B-vs-C test
are different experiments:

* **R1 "clean"** -- train at 0% masking, evaluate degraded. This is the PDF's
  minimum baseline: "evaluate *the same model* after artificially masking".
  It measures robustness of a model that never saw a cloud.
* **R2 "degraded"** -- train with masking augmentation, sampling a level per
  sample per epoch. Both the optical and the fusion arm get the identical
  augmentation, so the only difference between them is the SAR branch. This is
  the comparison that isolates the contribution of SAR.

Reporting only R1 would confound "SAR helps" with "training on masked data
helps"; reporting only R2 would ignore what the PDF asked for. So we run both.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .config import Config, set_seed
from .evaluate import compute_metrics
from .models import ClassifierHead, arm_input_dim


@dataclass
class FeatureBank:
    """Cached features for one split: optical at every level, SAR once."""

    optical: dict[float, np.ndarray]
    sar: np.ndarray
    targets: np.ndarray
    levels: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.levels = sorted(self.optical)
        n = len(self.targets)
        for lvl, f in self.optical.items():
            assert len(f) == n, f"optical level {lvl}: {len(f)} rows vs {n} targets"
        assert len(self.sar) == n, "SAR/target row mismatch"


def _assemble(arm: str, opt: np.ndarray, sar: np.ndarray) -> np.ndarray:
    if arm == "optical":
        return opt
    if arm == "sar":
        return sar
    if arm in ("fusion", "fusion_shuf"):
        return np.hstack([opt, sar])
    raise ValueError(arm)


def shuffled_sar(sar: np.ndarray, seed: int) -> np.ndarray:
    """Permute SAR rows across samples: same features, same head capacity, but
    the SAR vector no longer belongs to the patch it is concatenated with."""
    rng = np.random.default_rng(seed)
    return sar[rng.permutation(len(sar))]


def train_head(
    cfg: Config,
    arm: str,
    regime: str,
    train_bank: FeatureBank,
    val_bank: FeatureBank,
    n_classes: int,
    seed: int,
    device: torch.device,
    verbose: bool = False,
):
    """Return (trained head, history). Model selection is on validation macro F1."""
    set_seed(seed)
    tr_levels = [float(v) for v in cfg.train.train_levels] if regime == "degraded" else [0.0]

    sar_tr = shuffled_sar(train_bank.sar, seed) if arm == "fusion_shuf" else train_bank.sar
    sar_va = shuffled_sar(val_bank.sar, seed + 500) if arm == "fusion_shuf" else val_bank.sar

    # (n_levels, N, D) so a per-sample level can be gathered cheaply each epoch.
    Xtr = torch.stack([
        torch.from_numpy(_assemble(arm, train_bank.optical[l], sar_tr)) for l in tr_levels
    ]).to(device)
    Ytr = torch.from_numpy(train_bank.targets).to(device)
    Xva = {l: torch.from_numpy(_assemble(arm, val_bank.optical[l], sar_va)).to(device)
           for l in tr_levels}
    Yva = val_bank.targets

    head = ClassifierHead(arm_input_dim(arm, cfg), n_classes,
                          int(cfg.head.hidden_dim), float(cfg.head.dropout)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg.train.lr),
                            weight_decay=float(cfg.train.weight_decay))
    lossf = nn.BCEWithLogitsLoss()
    n = Xtr.shape[1]
    bs = int(cfg.train.batch_size)
    gen = torch.Generator().manual_seed(seed)

    best = {"macro_f1": -1.0, "epoch": -1, "state": None}
    history = []
    patience = int(cfg.train.early_stop_patience)

    for epoch in range(int(cfg.train.epochs)):
        head.train()
        perm = torch.randperm(n, generator=gen).to(device)
        # One masking level drawn per sample per epoch (a single level in R1).
        lvl_idx = torch.randint(len(tr_levels), (n,), generator=gen).to(device)
        total = 0.0
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            x = Xtr[lvl_idx[idx], idx]
            loss = lossf(head(x), Ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        head.eval()
        with torch.no_grad():
            scores = [
                compute_metrics(Yva, torch.sigmoid(head(Xva[l])).cpu().numpy(),
                                float(cfg.eval.threshold))["macro_f1"]
                for l in tr_levels
            ]
        val_f1 = float(np.mean(scores))   # averaged over the levels seen in training
        history.append({"epoch": epoch, "train_loss": total / n, "val_macro_f1": val_f1})
        if val_f1 > best["macro_f1"]:
            best = {"macro_f1": val_f1, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in head.state_dict().items()}}
        if epoch - best["epoch"] >= patience:
            break
        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch:3d} loss {total/n:.4f} val_macroF1 {val_f1:.4f}")

    head.load_state_dict(best["state"])
    head.eval()
    return head, {"history": history, "best_epoch": best["epoch"],
                  "best_val_macro_f1": best["macro_f1"]}


@torch.no_grad()
def predict(head, arm: str, bank: FeatureBank, level: float, device, seed: int = 0) -> np.ndarray:
    sar = shuffled_sar(bank.sar, seed + 900) if arm == "fusion_shuf" else bank.sar
    x = torch.from_numpy(_assemble(arm, bank.optical[level], sar)).to(device)
    return torch.sigmoid(head(x)).cpu().numpy()
