"""
Central configuration — V3.

Changes vs V2:
  1. UNFREEZE_TOP_LAYERS 2 → 4   : more encoder capacity; V2 showed large gains from partial fine-tuning
  2. LR_ROBERTA 2e-5 → 1e-5      : more conservative with 4 unfrozen layers (more catastrophic-forgetting risk)
  3. BATCH_SIZE 16 → 12           : 4 unfrozen layers doubles the RoBERTa backward memory requirement
  4. NUM_EPOCHS 15 → 20           : V2 folds 4, 7, 9 were still improving at ep 14-15; give more room
  5. PATIENCE 5 → 7               : consistent with longer training
  6. New result/checkpoint dirs   : V2 results are preserved untouched
"""
import os

# ── Google Drive paths ────────────────────────────────────────────────────────
DRIVE             = '/content/drive/MyDrive/eyebench_project'
EYEBENCH_CODEBASE = f'{DRIVE}/eyebench_codebase'
ONESTOP_EB        = f'{EYEBENCH_CODEBASE}/data/OneStop'
ONESTOP_RAW       = f'{DRIVE}/data/OneStop'
PROJECT_CODE      = f'{DRIVE}/Project_codebase_v3'

# ── Shared data caches (model-independent — reuse from V1/V2) ─────────────────
CACHE_DIR = f'{DRIVE}/cache_v3'

# ── V3-specific result / checkpoint dirs ─────────────────────────────────────
RESULTS_DIR = f'{DRIVE}/results_large_unfreeze4'
CKPT_DIR    = f'{DRIVE}/checkpoints_large_unfreeze4'

# ── Raw data files ────────────────────────────────────────────────────────────
IA_PATH         = f'{ONESTOP_RAW}/precomputed_reading_measures/ia_Paragraph.csv'
PARAGRAPHS_PATH = f'{ONESTOP_EB}/additional_raw/trial_level_paragraphs.csv'
QA_JSON_PATH    = f'{ONESTOP_EB}/additional_raw/onestop_qa.json'
FOLD_META_DIR   = f'{ONESTOP_EB}/folds_metadata/trial_ids'

# ── Cache file paths (shared with V1/V2 — same tokenizer vocab, same data) ───
TRIALS_CACHE_PATH    = f'{CACHE_DIR}/trials_df.parquet'
WORD_GAZE_CACHE_PATH = f'{CACHE_DIR}/word_gaze_sequences.pkl'
TOKENIZED_CACHE_PATH = f'{CACHE_DIR}/tokenized_tensors.pt'

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
ROBERTA_NAME         = 'roberta-large'
ROBERTA_DIM          = 1024
UNFREEZE_TOP_LAYERS  = 4       # V2 used 2; more capacity but need lower LR_ROBERTA
D_MODEL              = 256
N_HEADS              = 4
DROPOUT              = 0.2

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE   = 12       # reduced from 16 — 4 unfrozen layers doubles backward memory
NUM_EPOCHS   = 20       # increased from 15 — V2 folds were still improving at ep 15
LR           = 3e-4     # head / fusion / cross-attn learning rate (unchanged)
LR_ROBERTA   = 1e-5     # lowered from 2e-5 — more conservative with 4 unfrozen layers
WEIGHT_DECAY = 1e-2
PATIENCE     = 7        # increased from 5 — consistent with longer training budget
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
