"""
Training, evaluation, and results aggregation.

v2 changes vs v1:
  - Differential learning rates: unfrozen RoBERTa layers use config.LR_ROBERTA (2e-5),
    all other params use config.LR (3e-4). This prevents the unfrozen layers from
    diverging while the smaller head/fusion components learn at their normal rate.
  - OneCycleLR max_lr passed as a list [LR_ROBERTA, LR] when RoBERTa params are
    present, so each param group cycles independently.
  - When UNFREEZE_TOP_LAYERS=0 (fully frozen RoBERTa), behaviour is identical to v1.
"""

import os
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

import config


# ─────────────────────────────────────────────────────────────────────────────
# Loss + batch helpers
# ─────────────────────────────────────────────────────────────────────────────

def smooth_bce(logits: torch.Tensor, labels: torch.Tensor,
               epsilon: float = config.LABEL_SMOOTH) -> torch.Tensor:
    labels_s = labels * (1.0 - epsilon) + 0.5 * epsilon
    return F.binary_cross_entropy_with_logits(logits, labels_s)


def run_batch(model: nn.Module, batch: dict, device: str) -> torch.Tensor:
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return model(
        b['joint_ids'], b['joint_mask'], b['passage_wids'],
        b['q_ids'],     b['q_mask'],
        b['word_gaze'], b['word_mask'],
        b['gaze_stats'],
    ), b['label']


def _base(model: nn.Module) -> nn.Module:
    return getattr(model, '_orig_mod', model)


# ─────────────────────────────────────────────────────────────────────────────
# Epoch-level training and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total, skipped = 0.0, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, labels = run_batch(model, batch, device)
            loss = smooth_bce(logits, labels)
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        scheduler.step()
        total += loss.item()
    if skipped:
        print(f'    WARNING: {skipped} batches skipped (non-finite loss/grad)')
    return total / max(len(loader) - skipped, 1)


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, labels = run_batch(model, batch, device)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(int)
    logits_np = np.nan_to_num(logits_np, nan=0.0, posinf=10.0, neginf=-10.0)
    return logits_np, labels_np


def evaluate(logits_np: np.ndarray, labels_np: np.ndarray,
             threshold: float = 0.5) -> tuple[float, float]:
    probs   = torch.sigmoid(torch.tensor(logits_np)).numpy()
    preds   = (probs > threshold).astype(int)
    auroc   = roc_auc_score(labels_np, probs) * 100 if len(np.unique(labels_np)) > 1 else 50.0
    bal_acc = balanced_accuracy_score(labels_np, preds) * 100
    return auroc, bal_acc


def optimize_threshold(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    probs  = torch.sigmoid(torch.tensor(val_logits.astype(np.float32))).numpy()
    labels = val_labels.astype(int)
    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.02):
        ba = balanced_accuracy_score(labels, (probs > t).astype(int)) * 100
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_t


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer builder — differential LR
# ─────────────────────────────────────────────────────────────────────────────

def _build_optimizer_and_scheduler(model: nn.Module, n_steps: int):
    """
    Build AdamW + OneCycleLR with differential learning rates.

    - Unfrozen RoBERTa params → config.LR_ROBERTA (default 2e-5)
    - All other trainable params → config.LR (default 3e-4)

    When there are no unfrozen RoBERTa params (UNFREEZE_TOP_LAYERS=0),
    this reduces to a single-group optimizer identical to v1.
    """
    base = _base(model)

    roberta_params = [p for n, p in base.named_parameters()
                      if 'roberta' in n and p.requires_grad]
    other_params   = [p for n, p in base.named_parameters()
                      if 'roberta' not in n and p.requires_grad]

    if roberta_params:
        optimizer = AdamW(
            [
                {'params': roberta_params, 'lr': config.LR_ROBERTA},
                {'params': other_params,   'lr': config.LR},
            ],
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[config.LR_ROBERTA, config.LR],  # one per param group
            total_steps=n_steps,
            pct_start=0.1,
            anneal_strategy='cos',
        )
        print(f'    Optimizer: AdamW  |  RoBERTa LR={config.LR_ROBERTA}  '
              f'|  other LR={config.LR}  '
              f'|  roberta_params={sum(p.numel() for p in roberta_params):,}')
    else:
        optimizer = AdamW(other_params, lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.LR,
            total_steps=n_steps,
            pct_start=0.1,
            anneal_strategy='cos',
        )
        print(f'    Optimizer: AdamW  |  single LR={config.LR}  (RoBERTa fully frozen)')

    return optimizer, scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Full fold training
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    model:        nn.Module,
    train_loader,
    val_loader,
    fold_k:       int,
    variant_name: str   = 'wgq',
    device:       str   = 'cuda',
    num_epochs:   int   = config.NUM_EPOCHS,
    patience:     int   = config.PATIENCE,
) -> dict | None:
    """
    Train model for one fold with early stopping on val AUROC.
    Uses differential LR for unfrozen RoBERTa params.

    Returns best-checkpoint dict {model, val_logits, val_labels}
    or None if no checkpoint was saved (model never improved above 0.0).
    """
    best_ckpt  = f'{config.CKPT_DIR}/{variant_name}_fold{fold_k}_best.pt'
    epoch_ckpt = f'{config.CKPT_DIR}/{variant_name}_fold{fold_k}_epoch.pt'
    os.makedirs(config.CKPT_DIR, exist_ok=True)

    if os.path.exists(best_ckpt) and not os.path.exists(epoch_ckpt):
        state = torch.load(best_ckpt, map_location='cpu', weights_only=False)
        _base(model).load_state_dict(state['model'])
        print(f'  [fold {fold_k}] Best checkpoint loaded — training already done.')
        return state

    n_steps = num_epochs * len(train_loader)
    optimizer, scheduler = _build_optimizer_and_scheduler(model, n_steps)

    start_epoch, best_auroc, no_improve = 1, 0.0, 0

    if os.path.exists(epoch_ckpt):
        try:
            resume = torch.load(epoch_ckpt, map_location=device, weights_only=False)
            _base(model).load_state_dict(resume['model'])
            optimizer.load_state_dict(resume['optimizer'])
            scheduler.load_state_dict(resume['scheduler'])
            start_epoch = resume['epoch'] + 1
            best_auroc  = resume['best_auroc']
            no_improve  = resume['no_improve']
            print(f'  [fold {fold_k}] Resumed from epoch {resume["epoch"]} '
                  f'(best_auroc={best_auroc:.1f})')
        except Exception as e:
            print(f'  [fold {fold_k}] Epoch ckpt unreadable ({e}) — starting fresh.')
            os.remove(epoch_ckpt)

    for epoch in range(start_epoch, num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_logits, val_labels = collect_preds(model, val_loader, device)
        val_auroc, val_ba = evaluate(val_logits, val_labels)

        improved = val_auroc > best_auroc
        marker   = ' *' if improved else ''
        print(f'  [fold {fold_k}] ep {epoch:2d}/{num_epochs}  '
              f'loss={loss:.4f}  val_AUROC={val_auroc:.1f}  val_BalAcc={val_ba:.1f}{marker}',
              flush=True)

        if improved:
            best_auroc = val_auroc
            no_improve = 0
            torch.save({
                'model':      {k: v.cpu().clone() for k, v in _base(model).state_dict().items()},
                'val_logits': val_logits,
                'val_labels': val_labels,
            }, best_ckpt)
        else:
            no_improve += 1

        torch.save({
            'epoch':      epoch,
            'best_auroc': best_auroc,
            'no_improve': no_improve,
            'model':      {k: v.cpu().clone() for k, v in _base(model).state_dict().items()},
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
        }, epoch_ckpt)

        if no_improve >= patience:
            print(f'  [fold {fold_k}] Early stopping at epoch {epoch}')
            break

    if os.path.exists(epoch_ckpt):
        os.remove(epoch_ckpt)

    if not os.path.exists(best_ckpt):
        print(f'  [fold {fold_k}] WARNING: no best checkpoint saved (model never improved).')
        return None

    best = torch.load(best_ckpt, map_location='cpu', weights_only=False)
    _base(model).load_state_dict(best['model'])
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Results aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_fold_results(fold_results: dict) -> dict:
    data = {r: {'auroc': [], 'bal_acc': []} for r in config.REGIME_NAMES}
    data['All'] = {'auroc': [], 'bal_acc': []}

    for fk, regimes in fold_results.items():
        fa, fb = [], []
        for r, m in regimes.items():
            if r in data and isinstance(m, dict):
                data[r]['auroc'].append(m['auroc'])
                data[r]['bal_acc'].append(m['bal_acc'])
                fa.append(m['auroc'])
                fb.append(m['bal_acc'])
        if fa:
            data['All']['auroc'].append(np.mean(fa))
            data['All']['bal_acc'].append(np.mean(fb))

    out = {}
    for r, d in data.items():
        for m, vals in d.items():
            if vals:
                mn = np.mean(vals)
                se = np.std(vals) / math.sqrt(len(vals))
                out[f'{r}|{m}'] = {
                    'mean':      round(mn, 1),
                    'sem':       round(se, 1),
                    'formatted': f'{mn:.1f} ± {se:.1f}',
                    'n_folds':   len(vals),
                }
    return out


def print_results_table(summary: dict) -> None:
    print(f'\n{"Regime":40s} {"AUROC":15s} {"Bal.Acc":15s}')
    print('-' * 72)
    for regime in config.REGIME_NAMES + ['All']:
        a = summary.get(f'{regime}|auroc',   {}).get('formatted', 'N/A')
        b = summary.get(f'{regime}|bal_acc', {}).get('formatted', 'N/A')
        print(f'{regime:40s} {a:15s} {b:15s}')


def print_comparison_table(summary: dict) -> None:
    our_auroc = summary.get('All|auroc',   {}).get('formatted', 'N/A')
    our_bal   = summary.get('All|bal_acc', {}).get('formatted', 'N/A')
    print(f'\n{"Model":<42} {"AUROC (All)":>14} {"Bal.Acc (All)":>15}')
    print('-' * 74)
    baselines = [
        ('MAG-Eye (EyeBench)',           '62.9 ± ?', '54.3 ± ?'),
        ('Text-Only RoBERTa (EyeBench)', '61.1 ± ?', '55.0 ± ?'),
        ('PLM-AS-RM (EyeBench)',         '58.4 ± ?', '55.2 ± ?'),
        ('Random Forest (EyeBench)',     '58.0 ± ?', '55.1 ± ?'),
        ('Majority Class',               '50.0 ± 0', '50.0 ± 0'),
    ]
    for name, auroc, bal in baselines:
        print(f'{name:<42} {auroc:>14} {bal:>15}')
    print('-' * 74)
    print(f'{"WGQModel v2 (large, unfreeze2)":<42} {our_auroc:>14} {our_bal:>15}')
