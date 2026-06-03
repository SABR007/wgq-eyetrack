"""
Central configuration — V4.

V4 design rationale:
  V3 showed that 4 unfrozen layers causes overfitting (~6K training samples, 51M params).
  V2's 2-layer partial fine-tuning was stable. V4 keeps V2's encoder settings and adds:

  1. Mean+max pooling in Stream 2 — in the original proposal, adds peak-signal capacity
  2. Gated gaze stream — text_cls gates how much the gaze pool contributes per trial.
     Addresses V2 ablation finding: gaze can interfere with strong text encoder.
     The gate closes when text is already confident; opens when gaze adds unique signal.
  3. Longer training (20 epochs, patience 7) — V2 folds were cut short at 15 epochs.

  Encoder settings identical to V2: 2 unfrozen layers, LR_ROBERTA=2e-5, BATCH_SIZE=16.
"""

# ── Google Drive paths ────────────────────────────────────────────────────────
DRIVE             = '/content/drive/MyDrive/eyebench_project'
EYEBENCH_CODEBASE = f'{DRIVE}/eyebench_codebase'
ONESTOP_EB        = f'{EYEBENCH_CODEBASE}/data/OneStop'
ONESTOP_RAW       = f'{DRIVE}/data/OneStop'
PROJECT_CODE      = f'{DRIVE}/Project_codebase_v4'

# ── Shared data caches (reuse from V1/V2/V3 — same tokenizer, same data) ─────
CACHE_DIR = f'{DRIVE}/cache_v3'

# ── V4-specific output dirs ───────────────────────────────────────────────────
RESULTS_DIR = f'{DRIVE}/results_gated_v4'
CKPT_DIR    = f'{DRIVE}/checkpoints_gated_v4'

# ── Raw data files ────────────────────────────────────────────────────────────
IA_PATH         = f'{ONESTOP_RAW}/precomputed_reading_measures/ia_Paragraph.csv'
PARAGRAPHS_PATH = f'{ONESTOP_EB}/additional_raw/trial_level_paragraphs.csv'
QA_JSON_PATH    = f'{ONESTOP_EB}/additional_raw/onestop_qa.json'
FOLD_META_DIR   = f'{ONESTOP_EB}/folds_metadata/trial_ids'

# ── Shared cache paths ────────────────────────────────────────────────────────
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
    'IA_DWELL_TIME', 'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME',
    'IA_REGRESSION_PATH_DURATION', 'IA_FIXATION_COUNT',
    'IA_REGRESSION_IN_COUNT', 'IA_REGRESSION_OUT_COUNT', 'IA_REGRESSION_OUT_FULL_COUNT',
    'IA_SKIP', 'IA_RUN_COUNT', 'IA_FIRST_RUN_LANDING_POSITION', 'IA_AVERAGE_FIX_PUPIL_SIZE',
]
NUM_IA_FEATURES = len(IA_WORD_FEATURE_COLS)

# ── Trial-level global gaze statistics ───────────────────────────────────────
GAZE_STAT_COLS = [
    'stat_mean_dwell', 'stat_total_dwell', 'stat_regression_count',
    'stat_skip_rate', 'stat_fixation_count', 'stat_mean_regression_in',
]
NUM_GAZE_STATS = len(GAZE_STAT_COLS)

# ── Model hyperparameters — same encoder as V2 ────────────────────────────────
ROBERTA_NAME         = 'roberta-large'
ROBERTA_DIM          = 1024
UNFREEZE_TOP_LAYERS  = 2       # V2's proven setting — 4 layers caused overfitting in V3
D_MODEL              = 256
N_HEADS              = 4
DROPOUT              = 0.2

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE   = 16      # same as V2 (2 unfrozen layers, fits on A100 with bs=16)
NUM_EPOCHS   = 20      # extended from 15 — V2 folds were still improving at ep 15
LR           = 3e-4
LR_ROBERTA   = 2e-5    # same as V2 (2 unfrozen layers, no need to lower further)
WEIGHT_DECAY = 1e-2
PATIENCE     = 7       # extended from 5
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
    ('no_gate',       'A5: No gaze gate (remove gate, keep mean+max pool)'),
]

REGIME_NAMES  = ['Seen reader, unseen text', 'Unseen reader, seen text', 'Unseen reader, unseen text']
REGIME_SHORT  = ['Sr/Ut', 'Ur/St', 'Both']
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
