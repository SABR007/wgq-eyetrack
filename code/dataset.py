"""
PyTorch Dataset and DataLoader factory for the WGQModel.

Key design:
- OneStopWGQDataset indexes trials_df and looks up pre-computed tensors
  by DataFrame position (trials_df must be passed with the same ordering
  as the tokenized tensors dict).
- Normalizers (IA mean/std, gaze stats scaler) are fit on training data ONLY
  and applied to val/test — preventing data leakage.
- collate_fn pads word-gaze sequences to max length within each batch.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd

import config


class OneStopWGQDataset(Dataset):
    def __init__(
        self,
        trials_df:           pd.DataFrame,
        trial_id_to_pos:     dict,                # unique_trial_id → position in tokenized tensors
        tokenized:           dict,                # from tokenizer_utils.build_tokenized_tensors
        word_gaze:           dict,                # unique_trial_id → np.array(n_words, n_ia)
        gaze_stat_scaler:    StandardScaler,
        ia_mean:             np.ndarray,          # (num_ia_features,)
        ia_std:              np.ndarray,          # (num_ia_features,)
    ):
        self.df              = trials_df.reset_index(drop=True)
        self.id_to_pos       = trial_id_to_pos
        self.tok             = tokenized
        self.word_gaze       = word_gaze
        self.stat_scaler     = gaze_stat_scaler
        self.ia_mean         = ia_mean
        self.ia_std          = ia_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        tid = row[config.UNIQUE_TRIAL_COL]

        # ── Tokenized tensors (pre-computed, indexed by original position) ────
        pos = self.id_to_pos.get(tid, -1)
        if pos == -1:
            # Fallback: blank tensors (should not happen if splits are correct)
            j_ids  = torch.zeros(512, dtype=torch.long)
            j_mask = torch.zeros(512, dtype=torch.long)
            p_wids = torch.full((512,), -1, dtype=torch.long)
            q_ids  = torch.zeros(64,  dtype=torch.long)
            q_mask = torch.zeros(64,  dtype=torch.long)
        else:
            j_ids  = self.tok['joint_input_ids'][pos]
            j_mask = self.tok['joint_attn_mask'][pos]
            p_wids = self.tok['passage_wids'][pos]
            q_ids  = self.tok['question_input_ids'][pos]
            q_mask = self.tok['question_attn_mask'][pos]

        # ── Word-level IA gaze features ───────────────────────────────────────
        if tid in self.word_gaze:
            wg = self.word_gaze[tid].copy()
            wg = np.nan_to_num(wg, nan=0.0, posinf=0.0, neginf=0.0)
            wg = np.clip((wg - self.ia_mean) / self.ia_std, -20.0, 20.0)
        else:
            wg = np.zeros((1, len(self.ia_mean)), dtype=np.float32)

        # ── Global gaze statistics ────────────────────────────────────────────
        raw_stats = row[config.GAZE_STAT_COLS].fillna(0).values.astype(np.float32)
        gs = self.stat_scaler.transform(raw_stats.reshape(1, -1))[0].astype(np.float32)

        return {
            'j_ids'      : j_ids,
            'j_mask'     : j_mask,
            'p_wids'     : p_wids,
            'q_ids'      : q_ids,
            'q_mask'     : q_mask,
            'word_gaze'  : torch.tensor(wg, dtype=torch.float32),
            'gaze_stats' : torch.tensor(gs, dtype=torch.float32),
            'label'      : torch.tensor(float(row[config.IS_CORRECT_COL]), dtype=torch.float32),
        }


def collate_fn(batch: list) -> dict:
    wg_seqs = [b['word_gaze'] for b in batch]
    W_lens  = torch.tensor([s.size(0) for s in wg_seqs])
    W_max   = W_lens.max().item()
    F       = wg_seqs[0].size(-1)

    wg_padded = torch.zeros(len(batch), W_max, F)
    for i, wg in enumerate(wg_seqs):
        wg_padded[i, :wg.size(0)] = wg

    word_mask = torch.arange(W_max).unsqueeze(0) < W_lens.unsqueeze(1)

    # Clamp passage_wids to current batch W_max
    p_wids = torch.stack([b['p_wids'] for b in batch])
    p_wids = p_wids.clamp(min=-1, max=W_max - 1)

    return {
        'joint_ids'   : torch.stack([b['j_ids']       for b in batch]),
        'joint_mask'  : torch.stack([b['j_mask']       for b in batch]),
        'passage_wids': p_wids,
        'q_ids'       : torch.stack([b['q_ids']        for b in batch]),
        'q_mask'      : torch.stack([b['q_mask']       for b in batch]),
        'word_gaze'   : wg_padded,
        'word_mask'   : word_mask,
        'gaze_stats'  : torch.stack([b['gaze_stats']   for b in batch]),
        'label'       : torch.stack([b['label']        for b in batch]),
    }


def _fit_normalizers(train_df: pd.DataFrame, word_gaze: dict):
    """Fit IA and gaze-stats normalizers on training data only."""
    train_ids = train_df[config.UNIQUE_TRIAL_COL].values
    matched   = [tid for tid in train_ids if tid in word_gaze]

    if not matched:
        raise RuntimeError('No training trials found in word_gaze dict. '
                           'Check unique_trial_id construction.')

    all_ia  = np.concatenate([word_gaze[tid] for tid in matched], axis=0)
    ia_mean = np.nanmean(all_ia, axis=0).astype(np.float32)
    ia_std  = (np.nanstd(all_ia, axis=0) + 1e-8).astype(np.float32)

    stat_scaler = StandardScaler()
    stat_scaler.fit(train_df[config.GAZE_STAT_COLS].fillna(0).values.astype(np.float32))

    return ia_mean, ia_std, stat_scaler


def make_loaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_dfs: list[pd.DataFrame],
    tokenized: dict,
    word_gaze: dict,
    master_df: pd.DataFrame,          # full trials_df used during tokenization
) -> tuple:
    """
    Build DataLoaders for one fold.

    master_df is the full trials_df in the same order as tokenized tensors.
    train/val/test_dfs are subsets (rows of master_df).

    Returns: (train_loader, val_loader, [test_loader_r1, test_loader_r2, test_loader_r3])
    """
    # Build lookup: unique_trial_id → position in tokenized tensors
    id_to_pos = {tid: pos for pos, tid in enumerate(master_df[config.UNIQUE_TRIAL_COL])}

    ia_mean, ia_std, stat_scaler = _fit_normalizers(train_df, word_gaze)

    common = dict(
        trial_id_to_pos  = id_to_pos,
        tokenized        = tokenized,
        word_gaze        = word_gaze,
        gaze_stat_scaler = stat_scaler,
        ia_mean          = ia_mean,
        ia_std           = ia_std,
    )

    DL_KW = dict(
        collate_fn       = collate_fn,
        num_workers      = 4,
        pin_memory       = True,
        persistent_workers = True,
        prefetch_factor  = 2,
    )

    train_loader = DataLoader(
        OneStopWGQDataset(train_df, **common), config.BATCH_SIZE, shuffle=True,  **DL_KW
    )
    val_loader = DataLoader(
        OneStopWGQDataset(val_df,   **common), config.BATCH_SIZE, shuffle=False, **DL_KW
    )
    test_loaders = [
        DataLoader(OneStopWGQDataset(df, **common), config.BATCH_SIZE, shuffle=False, **DL_KW)
        for df in test_dfs
    ]

    return train_loader, val_loader, test_loaders
