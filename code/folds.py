"""
Loads EyeBench's pre-built fold CSVs and joins them with trials_df.

EyeBench stores exactly 9,718 trial IDs split across 7 regimes per fold:
  train_train
  val_{seen_subject_unseen_item, unseen_subject_seen_item, unseen_subject_unseen_item}
  test_{seen_subject_unseen_item, unseen_subject_seen_item, unseen_subject_unseen_item}

Fold assignment (from EyeBench's FoldSplitter.get_fold_indices):
  fold_k    → VAL
  (k+1)%10  → TEST
  rest       → TRAIN

We load the pre-built CSVs directly — no need to recompute splits.
"""

import os
import pandas as pd
import config


def _load_fold_csv(fold_k: int, fold_meta_dir: str) -> pd.DataFrame:
    path = os.path.join(fold_meta_dir, f'fold_{fold_k}_trial_ids_by_regime.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'EyeBench fold CSV not found: {path}\n'
            f'Make sure you uploaded the eyebench repo to {config.EYEBENCH_CODEBASE}'
        )
    return pd.read_csv(path)


def _join_with_trials(fold_csv_subset: pd.DataFrame, trials_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join fold_csv_subset with trials_df on unique_trial_id.
    Fills is_correct from fold CSV (authoritative) if missing in trials_df.
    """
    # Fold CSV columns we want: unique_trial_id + is_correct (ground truth label)
    fold_keys = fold_csv_subset[['unique_trial_id', 'is_correct']].rename(
        columns={'is_correct': '_fold_is_correct'}
    )

    merged = trials_df.merge(fold_keys, on='unique_trial_id', how='inner')

    # Use fold CSV is_correct as authoritative label
    merged['is_correct'] = merged['_fold_is_correct'].astype(int)
    merged = merged.drop(columns=['_fold_is_correct'])
    return merged.reset_index(drop=True)


def load_fold_splits(
    fold_k: int,
    trials_df: pd.DataFrame,
    fold_meta_dir: str = config.FOLD_META_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load EyeBench fold_k splits and join with trials_df.

    Returns:
        train_df : training trials
        val_df   : validation trials (all 3 val regimes combined, for model selection)
        r1_df    : test — Seen reader, Unseen text
        r2_df    : test — Unseen reader, Seen text
        r3_df    : test — Unseen reader, Unseen text
    """
    fold_csv = _load_fold_csv(fold_k, fold_meta_dir)
    trials_indexed = trials_df.set_index('unique_trial_id')

    def _get(regime_key: str) -> pd.DataFrame:
        ids = fold_csv.loc[fold_csv['regime'] == regime_key, 'unique_trial_id'].values
        valid = [tid for tid in ids if tid in trials_indexed.index]
        if len(valid) < len(ids):
            print(f'    fold {fold_k} / {regime_key}: {len(ids) - len(valid)} trial IDs '
                  f'not found in trials_df (word-gaze coverage gap)')
        return trials_indexed.loc[valid].reset_index()

    train_df = _get('train_train')
    val_df   = pd.concat([_get(k) for k in config.REGIME_VAL_KEYS],
                         ignore_index=True)
    r1_df    = _get(config.REGIME_TEST_KEYS[0])
    r2_df    = _get(config.REGIME_TEST_KEYS[1])
    r3_df    = _get(config.REGIME_TEST_KEYS[2])

    print(f'  Fold {fold_k:2d}  train={len(train_df):5d}  val={len(val_df):4d}  '
          f'R1(Sr/Ut)={len(r1_df):5d}  R2(Ur/St)={len(r2_df):4d}  R3(both)={len(r3_df):3d}')

    return train_df, val_df, r1_df, r2_df, r3_df


def print_fold_table(trials_df: pd.DataFrame, fold_meta_dir: str = config.FOLD_META_DIR) -> None:
    """Print split sizes for all 10 folds (diagnostic)."""
    print(f'{"Fold":>4}  {"Train":>7}  {"Val":>5}  {"Sr/Ut":>6}  {"Ur/St":>6}  {"Both":>5}')
    print('-' * 44)
    for k in range(config.N_FOLDS):
        try:
            tr, va, r1, r2, r3 = load_fold_splits(k, trials_df, fold_meta_dir)
            print(f'{k:>4}  {len(tr):>7}  {len(va):>5}  {len(r1):>6}  {len(r2):>6}  {len(r3):>5}')
        except FileNotFoundError as e:
            print(f'{k:>4}  ERROR: {e}')
