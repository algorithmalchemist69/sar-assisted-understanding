"""Training for the SAR -> optical reconstruction experiment (arm ED).

This is an *additional* arm alongside the existing B (degraded optical) and
C (degraded optical + raw SAR fusion). It does not modify either.

Two stages, deliberately kept separate
--------------------------------------
1. **Decoder.** ``z_sar -> z_hat_clean``, trained purely on a regression loss
   against the clean ViT feature. It never sees a class label. Keeping this
   stage label-free means the reconstruction metrics measure reconstruction,
   not "how well a feature was shaped for this particular classifier".
2. **Classifier.** The decoder is then frozen and the head is trained on
   ``[z_degraded ; z_hat_clean]`` (768-d) with exactly the recipe arms B and C
   use -- same optimiser, same schedule, same early stopping, same seeds.

Joint end-to-end training of both stages is the obvious alternative and would
probably score better, but it would destroy the thing the experiment is for: a
jointly-trained decoder is just a reparameterised fusion head, and the
reconstruction metric would no longer be independent evidence.

Regimes
-------
``ed_r1``  head trained at 0% masking only  (mirrors R1 "clean")
``ed_r2``  head trained with the R2 masking augmentation (mirrors R2 "degraded")

The *decoder* for variant ``ed`` is regime-independent by construction: neither
its input (SAR, never masked) nor its target (the clean optical feature) depends
on the masking level, so one decoder per seed serves both regimes. The
``ed_residual`` decoder does depend on the level distribution and is therefore
R2-only -- under R1 the residual target is identically zero and the variant is
degenerate.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import Config, set_seed
from .evaluate import compute_metrics
from .models.fusion_model import ClassifierHead
from .models.sar_optical_decoder import (
    SARToOpticalDecoder,
    ed_input_dim,
    reconstruction_loss,
)

ED_REGIMES = ("ed_r1", "ed_r2")
CLEAN_LEVEL = 0.0

# Variants whose decoder maps SAR -> clean optical directly (as opposed to
# predicting a residual). ``ed_shuf`` is the control: identical in every way to
# ``ed`` except the SAR rows are permuted against their targets during decoder
# training, so the decoder cannot learn any patch-specific SAR->optical mapping
# and can only fall back on the dataset mean. It mirrors the ``fusion_shuf``
# control the B/C experiment already uses.
DIRECT_VARIANTS = ("ed", "ed_shuf")


def _ed_cfg(cfg: Config):
    return cfg.encoder_decoder


def head_hidden_dim(cfg: Config) -> int:
    """Width of the ED classifier's hidden layer.

    Defaults to the project's ``head.hidden_dim`` (512) rather than a bespoke
    value, so ED differs from arms B and C only in *what is fed to the head* --
    which is the whole logic of the existing arm comparison. Override with
    ``encoder_decoder.head_hidden_dim`` if you want the narrower head.
    """
    v = _ed_cfg(cfg).get("head_hidden_dim")
    return int(v) if v else int(cfg.head.hidden_dim)


# --------------------------------------------------------------------------
# Stage 1: the decoder
# --------------------------------------------------------------------------

def train_decoder(
    cfg: Config,
    variant: str,
    train_bank,
    val_bank,
    seed: int,
    device: torch.device,
    verbose: bool = False,
):
    """Fit ``z_sar -> z_clean`` (``ed``) or ``z_sar -> z_clean - z_degraded``
    (``ed_residual``). Returns (decoder, history).

    Model selection is on *validation reconstruction loss*. No labels are used
    anywhere in this function.
    """
    set_seed(seed)
    ed = _ed_cfg(cfg)
    d_sar = int(cfg.models.sar.feature_dim)
    d_opt = int(cfg.models.optical.feature_dim)

    dec = SARToOpticalDecoder(
        in_dim=d_sar,
        hidden_dim=int(ed.hidden_dim),
        out_dim=int(ed.output_dim),
        dropout=float(ed.dropout),
        input_norm=bool(ed.get("input_norm", True)),
    ).to(device)
    assert int(ed.output_dim) == d_opt, "decoder output must match the ViT feature dim"

    Xtr = torch.from_numpy(train_bank.sar).to(device)
    Xva = torch.from_numpy(val_bank.sar).to(device)
    Ztr_clean = torch.from_numpy(train_bank.optical[CLEAN_LEVEL]).to(device)
    Zva_clean = torch.from_numpy(val_bank.optical[CLEAN_LEVEL]).to(device)

    if variant in DIRECT_VARIANTS:
        if variant == "ed_shuf":
            # Break the patch correspondence: same features, same capacity, but
            # this SAR vector no longer belongs to the patch it must predict.
            rng = np.random.default_rng(seed + 7000)
            Xtr = Xtr[torch.from_numpy(rng.permutation(Xtr.shape[0])).to(device)]
            rng2 = np.random.default_rng(seed + 7500)
            Xva = Xva[torch.from_numpy(rng2.permutation(Xva.shape[0])).to(device)]
        # Target is fixed: the clean optical feature.
        tr_levels = [CLEAN_LEVEL]
        Ttr = {CLEAN_LEVEL: Ztr_clean}
        Tva = {CLEAN_LEVEL: Zva_clean}
    elif variant == "ed_residual":
        # Target is the level-dependent residual the decoder must fill in.
        tr_levels = [float(v) for v in cfg.train.train_levels]
        Ttr = {l: Ztr_clean - torch.from_numpy(train_bank.optical[l]).to(device)
               for l in tr_levels}
        Tva = {l: Zva_clean - torch.from_numpy(val_bank.optical[l]).to(device)
               for l in tr_levels}
    else:
        raise ValueError(f"unknown ED variant: {variant}")

    opt = torch.optim.AdamW(
        dec.parameters(), lr=float(ed.lr), weight_decay=float(ed.weight_decay)
    )
    kind = str(ed.reconstruction_loss)
    cw = float(ed.get("cosine_weight", 1.0))
    n, bs = Xtr.shape[0], int(ed.batch_size)
    gen = torch.Generator().manual_seed(seed)

    best = {"loss": float("inf"), "epoch": -1, "state": None}
    history = []
    patience = int(ed.early_stop_patience)

    for epoch in range(int(ed.epochs)):
        dec.train()
        perm = torch.randperm(n, generator=gen).to(device)
        lvl_idx = torch.randint(len(tr_levels), (n,), generator=gen).to(device)
        total = 0.0
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            # Gather the per-sample target level in one pass.
            tgt = torch.empty(len(idx), int(ed.output_dim), device=device)
            for li, lvl in enumerate(tr_levels):
                sel = lvl_idx[idx] == li
                if sel.any():
                    tgt[sel] = Ttr[lvl][idx[sel]]
            loss = reconstruction_loss(dec(Xtr[idx]), tgt, kind, cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        dec.eval()
        with torch.no_grad():
            pred = dec(Xva)
            val = float(np.mean([
                reconstruction_loss(pred, Tva[l], kind, cw).item() for l in tr_levels
            ]))
        history.append({"epoch": epoch, "train_loss": total / n, "val_recon_loss": val})
        if val < best["loss"]:
            best = {"loss": val, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in dec.state_dict().items()}}
        if epoch - best["epoch"] >= patience:
            break
        if verbose and epoch % 10 == 0:
            print(f"      dec epoch {epoch:3d} train {total/n:.5f} val {val:.5f}")

    dec.load_state_dict(best["state"])
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec, {"history": history, "best_epoch": best["epoch"],
                 "best_val_recon_loss": best["loss"]}


@torch.no_grad()
def reconstruct(
    dec: SARToOpticalDecoder, variant: str, sar: np.ndarray,
    z_degraded: np.ndarray, device: torch.device,
) -> np.ndarray:
    """Produce ``z_hat_clean`` at inference. Never touches the clean feature.

    ``ed``           z_hat = decoder(z_sar)
    ``ed_residual``  z_hat = z_degraded + decoder(z_sar)
    """
    dec.eval()
    out = dec(torch.from_numpy(sar).to(device))
    if variant == "ed_residual":
        out = out + torch.from_numpy(z_degraded).to(device)
    return out.cpu().numpy()


# --------------------------------------------------------------------------
# Stage 2: the classifier on [z_degraded ; z_hat_clean]
# --------------------------------------------------------------------------

def _ed_features(variant: str, opt_feat: np.ndarray, z_hat_direct: np.ndarray,
                 delta_hat: np.ndarray) -> np.ndarray:
    """Assemble the 768-d classifier input for one masking level."""
    if variant in DIRECT_VARIANTS:
        z_hat = z_hat_direct
    else:
        z_hat = opt_feat + delta_hat
    return np.hstack([opt_feat, z_hat])


def train_ed_head(
    cfg: Config,
    variant: str,
    regime: str,
    dec: SARToOpticalDecoder,
    train_bank,
    val_bank,
    n_classes: int,
    seed: int,
    device: torch.device,
    verbose: bool = False,
):
    """Train the ED classifier. Mirrors ``src.train.train_head`` exactly except
    for the input assembly, so B / C / ED remain comparable."""
    set_seed(seed)
    tr_levels = (
        [float(v) for v in cfg.train.train_levels] if regime == "ed_r2" else [CLEAN_LEVEL]
    )

    # Decoder output is a fixed function of SAR, so compute it once per split.
    raw_tr = reconstruct(dec, "ed", train_bank.sar, train_bank.optical[CLEAN_LEVEL], device)
    raw_va = reconstruct(dec, "ed", val_bank.sar, val_bank.optical[CLEAN_LEVEL], device)

    Xtr = torch.stack([
        torch.from_numpy(_ed_features(variant, train_bank.optical[l], raw_tr, raw_tr))
        for l in tr_levels
    ]).to(device)
    Ytr = torch.from_numpy(train_bank.targets).to(device)
    Xva = {
        l: torch.from_numpy(
            _ed_features(variant, val_bank.optical[l], raw_va, raw_va)
        ).to(device)
        for l in tr_levels
    }
    Yva = val_bank.targets

    head = ClassifierHead(
        ed_input_dim(cfg), n_classes, head_hidden_dim(cfg), float(cfg.head.dropout)
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg.train.lr),
                            weight_decay=float(cfg.train.weight_decay))
    lossf = nn.BCEWithLogitsLoss()
    n, bs = Xtr.shape[1], int(cfg.train.batch_size)
    gen = torch.Generator().manual_seed(seed)

    best = {"macro_f1": -1.0, "epoch": -1, "state": None}
    history = []
    patience = int(cfg.train.early_stop_patience)

    for epoch in range(int(cfg.train.epochs)):
        head.train()
        perm = torch.randperm(n, generator=gen).to(device)
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
        val_f1 = float(np.mean(scores))
        history.append({"epoch": epoch, "train_loss": total / n, "val_macro_f1": val_f1})
        if val_f1 > best["macro_f1"]:
            best = {"macro_f1": val_f1, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in head.state_dict().items()}}
        if epoch - best["epoch"] >= patience:
            break
        if verbose and epoch % 10 == 0:
            print(f"      head epoch {epoch:3d} loss {total/n:.4f} val_macroF1 {val_f1:.4f}")

    head.load_state_dict(best["state"])
    head.eval()
    return head, {"history": history, "best_epoch": best["epoch"],
                  "best_val_macro_f1": best["macro_f1"]}


@torch.no_grad()
def predict_ed(head, variant: str, dec, bank, level: float, device) -> np.ndarray:
    """Test-time prediction. The clean optical feature is never read here."""
    raw = reconstruct(dec, "ed", bank.sar, bank.optical[level], device)
    x = torch.from_numpy(_ed_features(variant, bank.optical[level], raw, raw)).to(device)
    return torch.sigmoid(head(x)).cpu().numpy()


# --------------------------------------------------------------------------
# Reconstruction quality
# --------------------------------------------------------------------------

def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return num / den


def reconstruction_metrics(
    variant: str, dec, bank, level: float, device, train_mean: np.ndarray | None = None
) -> dict:
    """Is ``z_hat_clean`` actually closer to ``z_clean`` than ``z_degraded`` is?

    Raw MSE and cosine are reported, but they must not be read on their own.
    The SSL4EO ViT feature space is strongly anisotropic: on this test split
    ~93% of the feature energy lies in the dataset mean vector, two *unrelated*
    patches already have cosine 0.958, and a constant predictor that always
    outputs the training-set mean scores cosine 0.976. A decoder can therefore
    post a cosine of 0.99 while having learned almost nothing patch-specific.

    So the honest headline numbers here are the mean-relative ones:

    ``r2``            1 - MSE(z_hat, z_clean) / MSE(mu, z_clean). This is the
                      fraction of the *patch-to-patch variance* explained.
                      0 = no better than the constant mean predictor,
                      1 = perfect. Negative = worse than predicting the mean.
    ``centered_cos``  cosine after subtracting the training mean from both
                      vectors, i.e. direction agreement of the part that
                      actually distinguishes one patch from another.

    ``train_mean`` must be the mean of the *training* split's clean features;
    computing it on test would leak test statistics into the baseline.
    """
    z_clean = bank.optical[CLEAN_LEVEL]
    z_deg = bank.optical[level]
    raw = reconstruct(dec, "ed", bank.sar, z_deg, device)
    z_hat = raw if variant in DIRECT_VARIANTS else z_deg + raw

    mse_hat = float(((z_hat - z_clean) ** 2).mean())
    mse_deg = float(((z_deg - z_clean) ** 2).mean())
    out = {
        "mse_hat_vs_clean": mse_hat,
        "mse_degraded_vs_clean": mse_deg,
        "cos_hat_vs_clean": float(_cos(z_hat, z_clean).mean()),
        "cos_degraded_vs_clean": float(_cos(z_deg, z_clean).mean()),
        "cos_hat_vs_degraded": float(_cos(z_hat, z_deg).mean()),
        # >0 means the reconstruction is closer to clean than the degraded
        # feature is, i.e. the decoder moved the representation the right way.
        "mse_improvement": mse_deg - mse_hat,
        "recovers": bool(mse_hat < mse_deg),
    }
    if train_mean is not None:
        mu = np.asarray(train_mean).reshape(1, -1)
        mse_mean = float(((mu - z_clean) ** 2).mean())
        out.update({
            "mse_mean_predictor": mse_mean,
            "cos_mean_predictor": float(_cos(np.repeat(mu, len(z_clean), 0), z_clean).mean()),
            # Variance explained relative to the constant-mean baseline.
            "r2_hat": 1.0 - mse_hat / mse_mean,
            "r2_degraded": 1.0 - mse_deg / mse_mean,
            "centered_cos_hat": float(_cos(z_hat - mu, z_clean - mu).mean()),
            "centered_cos_degraded": float(_cos(z_deg - mu, z_clean - mu).mean()),
            # The claim "the decoder learned something patch-specific" requires
            # this to be clearly positive, not merely a high raw cosine.
            "beats_mean_predictor": bool(mse_hat < mse_mean),
        })
    return out
