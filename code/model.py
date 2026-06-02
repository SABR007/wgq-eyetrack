"""
WGQModel — Word-Gaze-Question model.

Three complementary streams, one classifier:

  Stream 1 — Text (joint passage+question CLS)
      RoBERTa( [CLS] passage [SEP][SEP] Question: {q} [SEP] ) → CLS token (768-d)
      This is the strong text baseline; passage and question both visible.

  Stream 2 — Word-Gaze-Question cross-attention
      For each passage word:
        - RoBERTa word embedding (scatter-meaned from token embeddings)
        - IA gaze features (dwell time, regressions, skip, etc.)
      Fused per-word representation cross-attends to question tokens:
        "For words relevant to this question, did the reader pay attention?"
      This is question-conditioned and works for unseen texts (uses actual word content).

  Stream 3 — Global gaze statistics
      6 trial-level handcrafted features (total reading time, regression count, etc.)
      Equivalent to Random Forest feature set (BalAcc ~55 standalone).

Ablation variants:
  TextOnlyModel    (A2): Stream 1 CLS only, no gaze
  NoQCondModel     (A1): Replace cross-attn with self-attn (question-agnostic gaze)
  NoGazeStatsModel (A3): Zero out Stream 3
  ZeroGazeModel    (A4): Zero out word-level IA features (text+question+stats only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

import config


class WGQModel(nn.Module):
    def __init__(
        self,
        num_ia_features: int = config.NUM_IA_FEATURES,
        num_gaze_stats:  int = config.NUM_GAZE_STATS,
        d:               int = config.D_MODEL,
        nhead:           int = config.N_HEADS,
        dropout:         float = config.DROPOUT,
        roberta_name:    str = config.ROBERTA_NAME,
    ):
        super().__init__()
        self.d = d

        # ── Frozen RoBERTa-base shared across both streams ────────────────────
        self.roberta = RobertaModel.from_pretrained(roberta_name, add_pooling_layer=False)
        for p in self.roberta.parameters():
            p.requires_grad = False
        self.roberta = self.roberta.to(torch.bfloat16)
        TEXT_DIM = self.roberta.config.hidden_size   # 768

        # ── Stream 2: IA gaze projection ─────────────────────────────────────
        self.gaze_proj = nn.Sequential(
            nn.Linear(num_ia_features, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── Stream 2: word-level (text + gaze) fusion → d ────────────────────
        self.word_fusion = nn.Sequential(
            nn.Linear(TEXT_DIM + 64, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(dropout * 0.5),
        )

        # ── Stream 2: question token projection to d ─────────────────────────
        self.q_proj = nn.Linear(TEXT_DIM, d)

        # ── Stream 2: cross-attention — fused passage words (Q) × question (KV)
        self.cross_attn   = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.post_attn_ln = nn.LayerNorm(d)
        self.attn_drop    = nn.Dropout(dropout)

        # ── Classifier: concat all 3 streams ─────────────────────────────────
        head_in = TEXT_DIM + d + num_gaze_stats
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    @torch.no_grad()
    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.roberta(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state.float()

    def _scatter_word_embs(
        self,
        token_hidden: torch.Tensor,  # (B, T, D)
        passage_wids: torch.Tensor,  # (B, T)  — word index; -1 = not a passage token
        W: int,
    ) -> torch.Tensor:
        """Average RoBERTa sub-word token embeddings per passage word via scatter_add."""
        B, T, D = token_hidden.shape
        valid  = (passage_wids >= 0)                        # (B, T)
        w_safe = passage_wids.clamp(min=0, max=W - 1)      # (B, T)

        h = token_hidden * valid.unsqueeze(-1).float()     # zero non-passage tokens
        w_exp = w_safe.unsqueeze(-1).expand(B, T, D)       # (B, T, D)

        word_sum = torch.zeros(B, W, D, device=token_hidden.device, dtype=token_hidden.dtype)
        word_cnt = torch.zeros(B, W,    device=token_hidden.device, dtype=token_hidden.dtype)
        word_sum.scatter_add_(1, w_exp,  h)
        word_cnt.scatter_add_(1, w_safe, valid.float())

        return word_sum / (word_cnt.unsqueeze(-1) + 1e-8)   # (B, W, D)

    def forward(
        self,
        joint_ids:    torch.Tensor,   # (B, 512)
        joint_mask:   torch.Tensor,   # (B, 512)
        passage_wids: torch.Tensor,   # (B, 512)
        q_ids:        torch.Tensor,   # (B, 64)
        q_mask:       torch.Tensor,   # (B, 64)
        word_gaze:    torch.Tensor,   # (B, W, NUM_IA_FEATURES)
        word_mask:    torch.Tensor,   # (B, W)  — 1=valid, 0=pad
        global_stats: torch.Tensor,   # (B, NUM_GAZE_STATS)
    ) -> torch.Tensor:                # (B,)

        W = word_mask.size(1)

        # One RoBERTa forward pass covers both streams
        joint_hidden = self._encode(joint_ids, joint_mask)    # (B, 512, 768)

        # Stream 1: CLS from joint passage+question encoding
        text_cls = joint_hidden[:, 0, :]                       # (B, 768)

        # Stream 2a: passage word embeddings via scatter mean
        word_text = self._scatter_word_embs(joint_hidden, passage_wids, W)  # (B, W, 768)

        # Stream 2b: question-only encoding (separate short pass)
        q_hidden = self._encode(q_ids, q_mask)                 # (B, 64, 768)
        q_proj   = self.q_proj(q_hidden)                       # (B, 64, d)

        # Stream 2: fuse word text + gaze
        gaze_emb = self.gaze_proj(word_gaze)                   # (B, W, 64)
        fused    = self.word_fusion(
            torch.cat([word_text, gaze_emb], dim=-1)           # (B, W, 768+64)
        )                                                       # (B, W, d)

        # Stream 2: cross-attention — fused passage words (Q) attend to question (KV)
        q_key_pad = ~q_mask.bool()                             # True = padding position
        att_out, _ = self.cross_attn(fused, q_proj, q_proj,
                                     key_padding_mask=q_key_pad)
        att_out  = att_out.nan_to_num(0.0)
        attended = self.post_attn_ln(fused + self.attn_drop(att_out))  # (B, W, d)

        # Stream 2: mask-weighted mean pool over passage words
        wm     = word_mask.unsqueeze(-1).float()               # (B, W, 1)
        pooled = (attended * wm).sum(1) / (wm.sum(1) + 1e-8)  # (B, d)

        combined = torch.cat([text_cls, pooled, global_stats], dim=-1)
        return self.classifier(combined).squeeze(-1)           # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation variants
# ─────────────────────────────────────────────────────────────────────────────

class TextOnlyModel(nn.Module):
    """A2: joint passage+question CLS only — no gaze streams."""
    def __init__(self, dropout: float = config.DROPOUT, roberta_name: str = config.ROBERTA_NAME):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_name, add_pooling_layer=False)
        for p in self.roberta.parameters():
            p.requires_grad = False
        self.roberta = self.roberta.to(torch.bfloat16)
        dim = self.roberta.config.hidden_size
        self.clf = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, joint_ids, joint_mask, passage_wids=None,
                q_ids=None, q_mask=None, word_gaze=None, word_mask=None, global_stats=None):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            h = self.roberta(input_ids=joint_ids, attention_mask=joint_mask).last_hidden_state
        return self.clf(h[:, 0, :].float()).squeeze(-1)


class NoQCondModel(nn.Module):
    """A1: No question conditioning — self-attention over fused passage words instead of cross-attn."""
    def __init__(self, num_ia_features=config.NUM_IA_FEATURES, num_gaze_stats=config.NUM_GAZE_STATS,
                 d=config.D_MODEL, nhead=config.N_HEADS, dropout=config.DROPOUT,
                 roberta_name=config.ROBERTA_NAME):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_name, add_pooling_layer=False)
        for p in self.roberta.parameters():
            p.requires_grad = False
        self.roberta = self.roberta.to(torch.bfloat16)
        TEXT_DIM = self.roberta.config.hidden_size
        self.gaze_proj   = nn.Sequential(nn.Linear(num_ia_features, 64), nn.GELU(), nn.Dropout(dropout*0.5))
        self.word_fusion = nn.Sequential(nn.Linear(TEXT_DIM + 64, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(dropout*0.5))
        self.self_attn   = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ln          = nn.LayerNorm(d)
        self.classifier  = nn.Sequential(
            nn.Linear(TEXT_DIM + d + num_gaze_stats, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(64, 1),
        )

    @torch.no_grad()
    def _enc(self, ids, mask):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.roberta(input_ids=ids, attention_mask=mask).last_hidden_state.float()

    def _scatter(self, h, wids, W):
        B, T, D = h.shape
        valid  = (wids >= 0)
        w_safe = wids.clamp(0, W-1)
        ws     = torch.zeros(B, W, D, device=h.device, dtype=h.dtype)
        wc     = torch.zeros(B, W,    device=h.device, dtype=h.dtype)
        ws.scatter_add_(1, w_safe.unsqueeze(-1).expand(B,T,D), h * valid.unsqueeze(-1).float())
        wc.scatter_add_(1, w_safe, valid.float())
        return ws / (wc.unsqueeze(-1) + 1e-8)

    def forward(self, joint_ids, joint_mask, passage_wids,
                q_ids, q_mask, word_gaze, word_mask, global_stats):
        W = word_mask.size(1)
        jh       = self._enc(joint_ids, joint_mask)
        text_cls = jh[:, 0, :]
        wt       = self._scatter(jh, passage_wids, W)
        fused    = self.word_fusion(torch.cat([wt, self.gaze_proj(word_gaze)], -1))
        pad_mask = ~word_mask.bool()
        sa, _    = self.self_attn(fused, fused, fused, key_padding_mask=pad_mask)
        fused    = self.ln(fused + sa.nan_to_num(0.0))
        wm       = word_mask.unsqueeze(-1).float()
        pooled   = (fused * wm).sum(1) / (wm.sum(1) + 1e-8)
        return self.classifier(torch.cat([text_cls, pooled, global_stats], -1)).squeeze(-1)


class NoGazeStatsModel(WGQModel):
    """A3: Zero out global gaze statistics — keep word-level IA."""
    def forward(self, joint_ids, joint_mask, passage_wids,
                q_ids, q_mask, word_gaze, word_mask, global_stats):
        return super().forward(
            joint_ids, joint_mask, passage_wids,
            q_ids, q_mask, word_gaze, word_mask,
            torch.zeros_like(global_stats),
        )


class ZeroGazeModel(WGQModel):
    """A4: Zero out word-level IA features — only text CLS + question + global stats."""
    def forward(self, joint_ids, joint_mask, passage_wids,
                q_ids, q_mask, word_gaze, word_mask, global_stats):
        return super().forward(
            joint_ids, joint_mask, passage_wids,
            q_ids, q_mask,
            torch.zeros_like(word_gaze),
            word_mask, global_stats,
        )


def build_model(variant: str = 'full_wgq', **kwargs) -> nn.Module:
    """Factory — returns model for given ablation variant key."""
    text_only_kwargs = {k: v for k, v in kwargs.items()
                        if k in ('dropout', 'roberta_name')}
    mapping = {
        'full_wgq':      (WGQModel,         kwargs),
        'no_q_cond':     (NoQCondModel,     kwargs),
        'text_only':     (TextOnlyModel,    text_only_kwargs),
        'no_gaze_stats': (NoGazeStatsModel, kwargs),
        'no_word_gaze':  (ZeroGazeModel,    kwargs),
    }
    if variant not in mapping:
        raise ValueError(f'Unknown variant: {variant!r}. Choose from {list(mapping.keys())}')
    cls, kw = mapping[variant]
    return cls(**kw)
