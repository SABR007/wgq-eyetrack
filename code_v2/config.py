"""
Central configuration: all paths and hyperparameters in one place.
Edit DRIVE to match your Google Drive mount point.

v2 changes vs v1:
  - RoBERTa-Large (24 layers, 1024-d) instead of RoBERTa-Base
  - Top 2 encoder layers unfrozen; rest remain frozen
  - Separate (lower) LR for unfrozen RoBERTa params
  - Batch size reduced from 48 → 16 (larger model footprint)
  - Separate results/checkpoint dirs so v1 results are preserved
"""
import os

# ── Google Drive paths ────────────────────────────────────────────────────────
DRIVE             = '/content/drive/MyDrive/eyebench_project'
EYEBENCH_CODEBASE = f'{DRIVE}/eyebench_codebase'
ONESTOP_EB        = f'{EYEBENCH_CODEBASE}/data/OneStop'
ONESTOP_RAW       = f'{DRIVE}/data/OneStop'
PROJECT_CODE      = f'{DRIVE}/Project_codebase_v2'

# ── Cache dirs (shared with v1 — data is model-independent) ──────────────────
CACHE_DIR = f'{DRIVE}/cache_v3'

# ── Result / checkpoint dirs — separate from v1 ───────────────────────────────
RESULTS_DIR = f'{DRIVE}/results_large_unfreeze2'
CKPT_DIR    = f'{DRIVE}/checkpoints_large_unfreeze2'

# ── Raw data files ────────────────────────────────────────────────────────────
IA_PATH         = f'{ONESTOP_RAW}/precomputed_reading_measures/ia_Paragraph.csv'
PARAGRAPHS_PATH = f'{ONESTOP_EB}/additional_raw/trial_level_paragraphs.csv'
QA_JSON_PATH    = f'{ONESTOP_EB}/additional_raw/onestop_qa.json'
FOLD_META_DIR   = f'{ONESTOP_EB}/folds_metadata/trial_ids'

# ── Cache file paths (reused from v1 — same tokenizer, same data) ─────────────
TRIALS_CACHE_PATH    = f'{CACHE_DIR}/trials_df.parquet'
WORD_GAZE_CACHE_PATH = f'{CACHE_DIR}/word_gaze_sequences.pkl'
TOKENIZED_CACHE_PATH = f'{CACHE_DIR}/tokenized_tensors.pt'  # same: roberta-large uses same tokenizer vocab

# ── Column names ──────────────────────────────────────────────────────────────
PARTICIPANT_COL   = 'participant_id'
UNIQUE_PARA_COL   = 'unique_paragraph_id'
UNIQUE_TRIAL_COL  = 'unique_trial_id'
IS_CORRECT_COL    = 'is_correct'
QUESTION_COL      = 'question'
PASSAGE_COL       = 'paragraph'
ARTICLE_BATCH_COL = 'article_batch'
ARTICLE_ID_COL    = 'article_id'

# ── IA word-level features ────────────────────────────────────────────────────
IA_WORD_FEATURE_COLS = [
    'IA_DWELL_TIME',
    'IA_FIRST_FIXATION_DURATION',
    'IA_FIRST_RUN_DWELL_TIME',
    'IA_REGRESSION_PATH_DURATION',
    'IA_FIXATION_COUNT',
    'IA_REGRESSION_IN_COUNT',
    'IA_REGRESSION_OUT_COUNT',
    'IA_REGRESSION_OUT_FULL_COUNT',
    'IA_SKIP',
    'IA_RUN_COUNT',
    'IA_FIRST_RUN_LANDING_POSITION',
    'IA_AVERAGE_FIX_PUPIL_SIZE',
]
NUM_IA_FEATURES = len(IA_WORD_FEATURE_COLS)

# ── Trial-level global gaze statistics ───────────────────────────────────────
GAZE_STAT_COLS = [
    'stat_mean_dwell',
    'stat_total_dwell',
    'stat_regression_count',
    'stat_skip_rate',
    'stat_fixation_count',
    'stat_mean_regression_in',
]
NUM_GAZE_STATS = len(GAZE_STAT_COLS)

# ── Model hyperparameters ─────────────────────────────────────────────────────
ROBERTA_NAME        = 'roberta-large'   # v1 used 'roberta-base'
ROBERTA_DIM         = 1024              # v1 was 768
UNFREEZE_TOP_LAYERS = 2                 # unfreeze top N encoder layers (0 = fully frozen)
D_MODEL             = 256
N_HEADS             = 4
DROPOUT             = 0.2

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE   = 16       # reduced from 48 — roberta-large is ~2.7× larger in memory
NUM_EPOCHS   = 15
LR           = 3e-4     # learning rate for all non-RoBERTa params
LR_ROBERTA   = 2e-5     # lower LR for unfrozen RoBERTa layers (prevents catastrophic forgetting)
WEIGHT_DECAY = 1e-2
PATIENCE     = 5
LABEL_SMOOTH = 0.05
MAX_WORDS    = 200
N_FOLDS      = 10

# ── Ablation variant names ────────────────────────────────────────────────────
ABLATION_VARIANTS = [
    ('full_wgq',      'Full WGQModel'),
    ('no_q_cond',     'A1: No Q-conditioning (self-attn)'),
    ('text_only',     'A2: Text-only (no gaze)'),
    ('no_gaze_stats', 'A3: No global gaze stats'),
    ('no_word_gaze',  'A4: No word-level IA (zeroed)'),
]

REGIME_NAMES = [
    'Seen reader, unseen text',
    'Unseen reader, seen text',
    'Unseen reader, unseen text',
]
REGIME_SHORT = ['Sr/Ut', 'Ur/St', 'Both']

# ── EyeBench regime keys in fold CSV ─────────────────────────────────────────
REGIME_TEST_KEYS = [
    'test_seen_subject_unseen_item',
    'test_unseen_subject_seen_item',
    'test_unseen_subject_unseen_item',
]
REGIME_VAL_KEYS = [
    'val_seen_subject_unseen_item',
    'val_unseen_subject_seen_item',
    'val_unseen_subject_unseen_item',
]
