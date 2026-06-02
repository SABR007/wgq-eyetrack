"""
Loads the OneStop IA CSV, applies EyeBench exclusion filters, builds:
  - trials_df   : one row per trial with metadata + passage text + global gaze stats
  - word_gaze   : dict[unique_trial_id -> np.array(n_words, NUM_IA_FEATURES)]

All trial IDs produced here match EyeBench's fold CSVs exactly.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import config

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_raw_ia(ia_path: str) -> pd.DataFrame:
    """Load IA CSV and apply EyeBench's three exclusion filters."""
    print(f'Loading IA CSV from {ia_path}  (this may take ~1-2 min) ...')
    ia_df = pd.read_csv(ia_path, low_memory=False, na_values=['.'])
    print(f'  Raw rows: {len(ia_df):,}')

    # EyeBench ia_query: practice_trial==False & question_preview==False & repeated_reading_trial==False
    for col_cands in [
        ['practice_trial', 'practice'],
        ['repeated_reading_trial', 'repeated_reading', 'repeated'],
        ['question_preview', 'preview', 'pre_question'],
    ]:
        col = _find_col(ia_df, col_cands)
        if col:
            ia_df[col] = pd.to_numeric(ia_df[col], errors='coerce').fillna(0)
            before = len(ia_df)
            ia_df = ia_df[ia_df[col] == 0].copy()
            print(f'  After {col}==0 filter: {len(ia_df):,}  (removed {before - len(ia_df):,})')
        else:
            print(f'  WARNING: exclusion column not found among {col_cands} — skipping filter')

    return ia_df


def _add_eyebench_ids(ia_df: pd.DataFrame) -> pd.DataFrame:
    """Add unique_paragraph_id and unique_trial_id columns matching EyeBench format."""
    # Coerce key columns to clean types
    for col in ['article_batch', 'article_id', 'paragraph_id']:
        if col in ia_df.columns:
            ia_df[col] = pd.to_numeric(ia_df[col], errors='coerce').fillna(0).astype(int)

    # unique_paragraph_id = "batch_articleid_level_paraid"  e.g. "1_2_Adv_1"
    ia_df['unique_paragraph_id'] = (
        ia_df['article_batch'].astype(str) + '_' +
        ia_df['article_id'].astype(str) + '_' +
        ia_df['difficulty_level'].astype(str) + '_' +
        ia_df['paragraph_id'].astype(str)
    )

    # unique_trial_id = "participantid_uniqueparagraphid_0_0"  (reread=0, practice=0 after filter)
    ia_df['unique_trial_id'] = (
        ia_df['participant_id'].astype(str) + '_' +
        ia_df['unique_paragraph_id'].astype(str) + '_0_0'
    )

    return ia_df


def _add_is_correct(ia_df: pd.DataFrame) -> pd.DataFrame:
    """Derive is_correct from selected_answer if not already present."""
    if 'is_correct' not in ia_df.columns:
        ans_col = _find_col(ia_df, ['selected_answer', 'answer', 'correct_answer'])
        if ans_col:
            ia_df['is_correct'] = (ia_df[ans_col] == 'A').astype(int)
        else:
            print('WARNING: cannot derive is_correct — column will be filled from fold CSV later')
            ia_df['is_correct'] = np.nan
    else:
        ia_df['is_correct'] = pd.to_numeric(ia_df['is_correct'], errors='coerce')
    return ia_df


def _compute_gaze_stats(ia_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 6 trial-level gaze statistics per unique_trial_id."""
    agg_map = {}
    present_ia = [c for c in config.IA_WORD_FEATURE_COLS if c in ia_df.columns]
    for col in ['IA_DWELL_TIME', 'IA_REGRESSION_OUT_COUNT', 'IA_SKIP',
                'IA_FIXATION_COUNT', 'IA_REGRESSION_IN_COUNT']:
        if col in ia_df.columns:
            ia_df[col] = pd.to_numeric(ia_df[col], errors='coerce').fillna(0)

    if 'IA_DWELL_TIME' in ia_df.columns:
        agg_map['stat_mean_dwell']  = pd.NamedAgg('IA_DWELL_TIME', 'mean')
        agg_map['stat_total_dwell'] = pd.NamedAgg('IA_DWELL_TIME', 'sum')
    if 'IA_REGRESSION_OUT_COUNT' in ia_df.columns:
        agg_map['stat_regression_count'] = pd.NamedAgg('IA_REGRESSION_OUT_COUNT', 'sum')
    if 'IA_SKIP' in ia_df.columns:
        agg_map['stat_skip_rate'] = pd.NamedAgg('IA_SKIP', 'mean')
    if 'IA_FIXATION_COUNT' in ia_df.columns:
        agg_map['stat_fixation_count'] = pd.NamedAgg('IA_FIXATION_COUNT', 'sum')
    if 'IA_REGRESSION_IN_COUNT' in ia_df.columns:
        agg_map['stat_mean_regression_in'] = pd.NamedAgg('IA_REGRESSION_IN_COUNT', 'mean')

    stats = ia_df.groupby('unique_trial_id').agg(**agg_map).reset_index()

    # Fill any missing stat columns with 0
    for col in config.GAZE_STAT_COLS:
        if col not in stats.columns:
            stats[col] = 0.0

    return stats[['unique_trial_id'] + config.GAZE_STAT_COLS]


def _extract_trial_meta(ia_df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique_trial_id with metadata columns."""
    keep = ['unique_trial_id', 'participant_id', 'unique_paragraph_id',
            'article_batch', 'article_id', 'difficulty_level', 'paragraph_id',
            'is_correct', 'question']
    available = [c for c in keep if c in ia_df.columns]
    meta = (ia_df[available]
            .drop_duplicates('unique_trial_id')
            .reset_index(drop=True))
    return meta


def _merge_passage_text(trials_df: pd.DataFrame, paragraphs_path: str) -> pd.DataFrame:
    """
    Merge passage text from trial_level_paragraphs.csv.
    Joins on unique_paragraph_id (same passage for all readers of that paragraph).
    """
    paras = pd.read_csv(paragraphs_path)
    for col in ['article_batch', 'article_id', 'paragraph_id']:
        if col in paras.columns:
            paras[col] = pd.to_numeric(paras[col], errors='coerce').fillna(0).astype(int)

    paras['unique_paragraph_id'] = (
        paras['article_batch'].astype(str) + '_' +
        paras['article_id'].astype(str) + '_' +
        paras['difficulty_level'].astype(str) + '_' +
        paras['paragraph_id'].astype(str)
    )

    para_map = (paras[['unique_paragraph_id', 'paragraph']]
                .drop_duplicates('unique_paragraph_id')
                .set_index('unique_paragraph_id')['paragraph'])

    trials_df['paragraph'] = trials_df['unique_paragraph_id'].map(para_map)
    missing = trials_df['paragraph'].isna().sum()
    if missing:
        print(f'WARNING: {missing} trials have no passage text — filling with empty string')
        trials_df['paragraph'] = trials_df['paragraph'].fillna('')

    coverage = (trials_df['paragraph'].str.len() > 0).mean()
    print(f'Passage text coverage: {coverage:.1%}  ({len(trials_df)} trials)')
    return trials_df


# ─────────────────────────────────────────────────────────────────────────────
# Word-gaze cache builder
# ─────────────────────────────────────────────────────────────────────────────

def build_word_gaze_cache(ia_path: str, cache_path: str) -> dict:
    """
    Build word_gaze_sequences: dict[unique_trial_id -> np.array(n_words, NUM_IA_FEATURES)]
    Loads from cache_path if it exists, otherwise builds from IA CSV and saves.
    """
    if os.path.exists(cache_path):
        print(f'Loading word-gaze cache from {cache_path} ...')
        with open(cache_path, 'rb') as f:
            wg = pickle.load(f)
        print(f'  Loaded {len(wg):,} trials  |  {config.NUM_IA_FEATURES} IA features')
        return wg

    print('Building word-gaze cache from raw IA CSV (~3-5 min) ...')
    ia_df = _load_raw_ia(ia_path)
    ia_df = _add_eyebench_ids(ia_df)

    # Find IA word-index column
    ia_id_col = _find_col(ia_df, ['IA_ID', 'IA_INTEREST_AREA_INDEX', 'IA_INDEX', 'ia_index'])
    assert ia_id_col, f'IA word-index column not found. Columns: {ia_df.columns.tolist()}'

    # Convert IA feature columns to numeric
    ia_feat_cols = [c for c in config.IA_WORD_FEATURE_COLS if c in ia_df.columns]
    missing_feats = [c for c in config.IA_WORD_FEATURE_COLS if c not in ia_df.columns]
    if missing_feats:
        print(f'WARNING: {len(missing_feats)} IA feature columns not found: {missing_feats}')

    for col in ia_feat_cols + [ia_id_col]:
        ia_df[col] = pd.to_numeric(ia_df[col], errors='coerce').fillna(0)

    word_gaze = {}
    for trial_id, grp in tqdm(ia_df.groupby('unique_trial_id'), desc='Building word-gaze matrices'):
        grp = grp.sort_values(ia_id_col).reset_index(drop=True)
        ia_idx = grp[ia_id_col].values.astype(int)
        ia_idx = (ia_idx - ia_idx.min()).clip(min=0)   # 0-indexed, trial-relative
        n_words = min(ia_idx.max() + 1, config.MAX_WORDS)

        mat = np.zeros((n_words, len(ia_feat_cols)), dtype=np.float32)
        feats = grp[ia_feat_cols].fillna(0.0).values.astype(np.float32)
        for i, idx in enumerate(ia_idx):
            if 0 <= idx < n_words:
                mat[idx] = feats[i]
        word_gaze[trial_id] = mat

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(word_gaze, f)
    print(f'Saved word-gaze cache: {len(word_gaze):,} trials  →  {cache_path}')
    return word_gaze


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_trials_df(
    ia_path: str,
    paragraphs_path: str,
    cache_path: str,
) -> pd.DataFrame:
    """
    Build trials_df: one row per trial with all metadata including passage text.
    Loads from cache_path if present; otherwise builds and caches.

    Returns:
        trials_df with columns:
          unique_trial_id, participant_id, unique_paragraph_id,
          article_batch, article_id, is_correct, question, paragraph,
          stat_mean_dwell, stat_total_dwell, stat_regression_count,
          stat_skip_rate, stat_fixation_count, stat_mean_regression_in
    """
    if os.path.exists(cache_path):
        print(f'Loading trials_df cache from {cache_path} ...')
        trials_df = pd.read_parquet(cache_path)
        print(f'  {len(trials_df):,} trials  |  columns: {trials_df.columns.tolist()}')
        return trials_df

    print('Building trials_df from scratch ...')
    ia_df = _load_raw_ia(ia_path)
    ia_df = _add_eyebench_ids(ia_df)
    ia_df = _add_is_correct(ia_df)

    # Build gaze stats and metadata separately, then merge
    stats = _compute_gaze_stats(ia_df)
    meta  = _extract_trial_meta(ia_df)
    trials_df = meta.merge(stats, on='unique_trial_id', how='left')

    # Merge passage text (the critical Fix 1)
    trials_df = _merge_passage_text(trials_df, paragraphs_path)

    # Fill missing stats with 0
    for col in config.GAZE_STAT_COLS:
        if col in trials_df.columns:
            trials_df[col] = trials_df[col].fillna(0.0)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    trials_df.to_parquet(cache_path, index=False)
    print(f'Saved trials_df: {len(trials_df):,} trials  →  {cache_path}')
    return trials_df
