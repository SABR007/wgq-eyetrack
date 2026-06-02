"""
Pre-tokenizes all trials into tensors stored in CPU memory.

For each trial, three things are stored:
  joint_input_ids / joint_attn_mask  : [CLS] passage [SEP][SEP] Question: {q} [SEP]
  passage_wids                        : per-token passage word index (-1 = special / question)
  question_input_ids / question_attn_mask : question-only encoding (for cross-attn KV)

Uses RobertaTokenizerFast so sequence_ids() and word_ids() are available.
passage_wids is what enables scatter-mean alignment of RoBERTa token
embeddings to word-level gaze features.
"""

import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import RobertaTokenizerFast

import config


def build_tokenized_tensors(
    trials_df: pd.DataFrame,
    tokenizer: RobertaTokenizerFast,
    cache_path: str,
) -> dict[str, torch.Tensor]:
    """
    Pre-tokenize all trials in trials_df.

    Returns dict with keys:
      joint_input_ids     : (N, 512)  long
      joint_attn_mask     : (N, 512)  long
      passage_wids        : (N, 512)  long  — passage word index per token, -1 elsewhere
      question_input_ids  : (N, 64)   long
      question_attn_mask  : (N, 64)   long

    N = len(trials_df), indexed in the same order as trials_df.
    """
    if os.path.exists(cache_path):
        print(f'Loading tokenized tensors from {cache_path} ...')
        tok = torch.load(cache_path, map_location='cpu', weights_only=True)
        if tok['joint_input_ids'].shape[0] == len(trials_df):
            print(f'  Loaded. N={tok["joint_input_ids"].shape[0]}')
            return tok
        print('  Size mismatch — re-tokenizing.')
        os.remove(cache_path)

    N = len(trials_df)
    joint_ids  = torch.zeros(N, 512, dtype=torch.long)
    joint_mask = torch.zeros(N, 512, dtype=torch.long)
    p_wids     = torch.full((N, 512), -1, dtype=torch.long)
    q_ids      = torch.zeros(N, 64,  dtype=torch.long)
    q_mask     = torch.zeros(N, 64,  dtype=torch.long)

    passage_col  = config.PASSAGE_COL
    question_col = config.QUESTION_COL

    missing_passage  = 0
    missing_question = 0

    print(f'Pre-tokenizing {N} trials ...')
    for idx in tqdm(range(N)):
        row = trials_df.iloc[idx]

        passage  = str(row[passage_col])  if passage_col  in trials_df.columns and pd.notna(row.get(passage_col))  else ''
        question = str(row[question_col]) if question_col in trials_df.columns and pd.notna(row.get(question_col)) else ''

        if not passage:  missing_passage  += 1
        if not question: missing_question += 1

        # ── Joint encoding: [CLS] passage [SEP][SEP] Question: {q} [SEP] ─────
        q_text = f'Question: {question}' if question else 'Question:'
        je = tokenizer(
            passage, q_text,
            max_length=512, padding='max_length',
            truncation=True, return_tensors='pt',
        )
        joint_ids[idx]  = je['input_ids'].squeeze(0)
        joint_mask[idx] = je['attention_mask'].squeeze(0)

        # passage_wids: for each token, its 0-indexed word position in the passage
        # sequence_ids() returns 0 = passage, 1 = question, None = special
        seq_ids  = je.sequence_ids(batch_index=0)
        word_ids = je.word_ids(batch_index=0)
        for t, (s, w) in enumerate(zip(seq_ids, word_ids)):
            if s == 0 and w is not None:
                p_wids[idx, t] = min(w, config.MAX_WORDS - 1)

        # ── Question-only encoding (for cross-attention key/value) ────────────
        qe = tokenizer(
            q_text,
            max_length=64, padding='max_length',
            truncation=True, return_tensors='pt',
        )
        q_ids[idx]  = qe['input_ids'].squeeze(0)
        q_mask[idx] = qe['attention_mask'].squeeze(0)

    if missing_passage > 0:
        print(f'WARNING: {missing_passage}/{N} trials have empty passage text.'
              f'  Check that PASSAGE_COL="{passage_col}" is populated in trials_df.')
    if missing_question > 0:
        print(f'WARNING: {missing_question}/{N} trials have empty question text.')

    result = {
        'joint_input_ids':    joint_ids,
        'joint_attn_mask':    joint_mask,
        'passage_wids':       p_wids,
        'question_input_ids': q_ids,
        'question_attn_mask': q_mask,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(result, cache_path)
    print(f'Saved tokenized tensors  →  {cache_path}')
    return result
