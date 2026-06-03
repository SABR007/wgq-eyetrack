"""
WGQModel — Word-Gaze-Question model (V4).

V4 changes vs V2:
  1. Mean+max pooling in Stream 2
     Mean pool: average attended signal across all passage words
     Max pool:  peak attended signal — the words the model is MOST confident matter
     Concatenated → pooled shape (B, 2*d) instead of (B, d)

  2. Gated gaze stream
     A learned gate is computed from text_cls and applied element-wise to pooled:
       gate   = sigmoid( Linear(TEXT_DIM → 2*d)(text_cls) )    # (B, 2*d)
       pooled = gate * pooled                                   # (B, 2*d)
     Why: V2 ablation showed gaze can interfere with the strong text encoder on some
     folds. The gate lets the model learn WHEN gaze adds value (e.g. seen texts where
     specific word-question patterns are learnable) vs when text alone is sufficient.
     The gate adds only TEXT_DIM × 2*d = 1024 × 512 ≈ 0.5M parameters.

  3. A5 ablation added: NoGateModel — removes gate but keeps mean+max pooling.
     Allows isolating the gate's contribution from mean+max pooling.

Encoder unchanged from V2: roberta-large, top 2 layers unfrozen.
Classifier head: TEXT_DIM + 2*d + num_gaze_stats = 1024 + 512 + 6 = 1542.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

import config


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: selective RoBERTa freezing
# ─────────────────────────────────────────────────────────────────────────────

def _setup_roberta_freezing(roberta: RobertaModel,
                             unfreeze_top: int = config.UNFREEZE_TOP_LAYERS) -> None:
    for p in roberta.parameters():
        p.requires_grad = False
    if unfreeze_top > 0:
        n_layers = len(roberta.encoder.layer)
        for i in range(n_layers - unfreeze_top, n_layers):
            for p in roberta.encoder.layer[i].parameters():
                p.requires_grad = True
        unfrozen = sum(p.numel() for p in roberta.parameters() if p.requires_grad)
        print(f'    RoBERTa: {n_layers} layers, top {unfreeze_top} unfrozen '
              f'({unfrozen:,} trainable RoBERTa params)')


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class WGQModel(nn.Module):
    def __init__(
        self,
        num_ia_features:     int   = config.NUM_IA_FEATURES,
        num_gaze_stats:      int   = config.NUM_GAZE_STATS,
        d:                   int   = config.D_MODEL,
        nhead:               int   = config.N_HEADS,
        dropout:             float = config.DROPOUT,
        roberta_name:        str   = config.ROBERTA_NAME,
        unfreeze_top_layers: int   = config.UNFREEZE_TOP_LAYERS,
    ):
        super().__init__()
        self.d = d

        self.roberta = RobertaModel.from_pretrained(roberta_name, add_pooling_layer=False)
        _setup_roberta_freezing(self.roberta, unfreeze_top_layers)
        TEXT_DIM = self.roberta.config.hidden_size  # 1024

        # Stream 2: IA gaze projection
        self.gaze_proj = nn.Sequential(
            nn.Linear(num_ia_features, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Stream 2: word-level fusion (text + gaze → d)
        self.word_fusion = nn.Sequential(
            nn.Linear(TEXT_DIM + 64, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(dropout * 0.5),
        )

        # Stream 2: question token projection to d
        self.q_proj = nn.Linear(TEXT_DIM, d)

        # Stream 2: cross-attention
        self.cross_attn   = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.post_attn_ln = nn.LayerNorm(d)
        self.attn_drop    = nn.Dropout(dropout)

        # V4: gaze gate — learned from text_cls, applied to mean+max pooled gaze
        # Allows text representation to control how much gaze contributes
        self.gaze_gate = nn.Linear(TEXT_DIM, 2 * d)

        # Classifier: TEXT_DIM + 2*d (gated mean+max) + num_gaze_stats
        head_in = TEXT_DIM + 2 * d + num_gaze_stats
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.roberta(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state.float()

    def _scatter_word_embs(self, token_hidden, passage_wids, W):
        B, T, D = token_hidden.shape
        valid  = (passage_wids >= 0)
        w_safe = passage_wids.clamp(min=0, max=W - 1)
        h      = token_hidden * valid.unsqueeze(-1).float()
        w_exp  = w_safe.unsqueeze(-1).expand(B, T, D)
        wsum = torch.zeros(B, W, D, device=token_hidden.device, dtype=token_hidden.dtype)
        wcnt = torch.zeros(B, W,    device=token_hidden.device, dtype=token_hidden.dtype)
        wsum.scatter_add_(1, w_exp,  h)
        wcnt.scatter_add_(1, w_safe, valid.float())
        return wsum / (wcnt.unsqueeze(-1) + 1e-8)

    def _mean_max_pool(self, attended, word_mask):
        """Concatenate mean and max pool over valid passage words."""
        wm        = word_mask.unsqueeze(-1).float()
        mean_pool = (attended * wm).sum(1) / (wm.sum(1) + 1e-8)
        max_pool  = (attended - (1.0 - wm) * 1e9).max(dim=1).values.nan_to_num(0.0)
        return torch.cat([mean_pool, max_pool], dim=-1)   # (B, 2*d)

    def forward(
        self,
        joint_ids:    torch.Tensor,
        joint_mask:   torch.Tensor,
        passage_wids: torch.Tensor,
        q_ids:        torch.Tensor,
        q_mask:       torch.Tensor,
        word_gaze:    torch.Tensor,
        word_mask:    torch.Tensor,
        global_stats: torch.Tensor,
    ) -> torch.Tensor:

        W = word_mask.size(1)

        joint_hidden = self._encode(joint_ids, joint_mask)
        text_cls     = joint_hidden[:, 0, :]                              # (B, TEXT_DIM)
        word_text    = self._scatter_word_embs(joint_hidden, passage_wids, W)

        q_hidden = self._encode(q_ids, q_mask)
        q_proj   = self.q_proj(q_hidden)                                  # (B, 64, d)

        gaze_emb = self.gaze_proj(word_gaze)
        fused    = self.word_fusion(torch.cat([word_text, gaze_emb], dim=-1))  # (B, W, d)

        q_key_pad  = ~q_mask.bool()
        att_out, _ = self.cross_attn(fused, q_proj, q_proj, key_padding_mask=q_key_pad)
        att_out    = att_out.nan_to_num(0.0)
        attended   = self.post_attn_ln(fused + self.attn_drop(att_out))   # (B, W, d)

        # Mean+max pool → (B, 2*d)
        pooled = self._mean_max_pool(attended, word_mask)

        # V4: gated gaze — text_cls learns when to trust gaze
        gate   = torch.sigmoid(self.gaze_gate(text_cls))                  # (B, 2*d)
        pooled = gate * pooled                                            # (B, 2*d)

        combined = torch.cat([text_cls, pooled, global_stats], dim=-1)
        return self.classifier(combined).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation variants
# ─────────────────────────────────────────────────────────────────────────────

class TextOnlyModel(nn.Module):
    """A2: joint passage+question CLS only — no gaze."""
    def __init__(self, dropout=config.DROPOUT, roberta_name=config.ROBERTA_NAME,
                 unfreeze_top_layers=config.UNFREEZE_TOP_LAYERS):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_name, add_pooling_layer=False)
        _setup_roberta_freezing(self.roberta, unfreeze_top_layers)
        dim = self.roberta.config.hidden_size
        self.clf = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, joint_ids, joint_mask, passage_wids=None,
                q_ids=None, q_mask=None, word_gaze=None, word_mask=None, global_stats=None):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = self.roberta(input_ids=joint_ids, attention_mask=joint_mask).last_hidden_state
        return self.clf(h[:, 0, :].float()).squeeze(-1)


class NoQCondModel(nn.Module):
    """A1: self-attention instead of cross-attention (no question conditioning)."""
    def __init__(self, num_ia_features=config.NUM_IA_FEATURES, num_gaze_stats=config.NUM_GAZE_STATS,
                 d=config.D_MODEL, nhead=config.N_HEADS, dropout=config.DROPOUT,
                 roberta_name=config.ROBERTA_NAME, unfreeze_top_layers=config.UNFREEZE_TOP_LAYERS):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_name, add_pooling_layer=False)
        _setup_roberta_freezing(self.roberta, unfreeze_top_layers)
        TEXT_DIM = self.roberta.config.hidden_size
        self.gaze_proj   = nn.Sequential(nn.Linear(num_ia_features, 64), nn.GELU(), nn.Dropout(dropout*0.5))
        self.word_fusion = nn.Sequential(nn.Linear(TEXT_DIM+64, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(dropout*0.5))
        self.self_attn   = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ln          = nn.LayerNorm(d)
        self.gaze_gate   = nn.Linear(TEXT_DIM, 2 * d)
        self.classifier  = nn.Sequential(
            nn.Linear(TEXT_DIM + 2*d + num_gaze_stats, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(64, 1),
        )

    def _enc(self, ids, mask):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.roberta(input_ids=ids, attention_mask=mask).last_hidden_state.float()

    def _scatter(self, h, wids, W):
        B, T, D = h.shape
        valid = (wids >= 0); w_safe = wids.clamp(0, W-1)
        ws = torch.zeros(B, W, D, device=h.device, dtype=h.dtype)
        wc = torch.zeros(B, W,    device=h.device, dtype=h.dtype)
        ws.scatter_add_(1, w_safe.unsqueeze(-1).expand(B,T,D), h * valid.unsqueeze(-1).float())
        wc.scatter_add_(1, w_safe, valid.float())
        return ws / (wc.unsqueeze(-1) + 1e-8)

    def _mean_max_pool(self, attended, word_mask):
        wm = word_mask.unsqueeze(-1).float()
        mean_p = (attended * wm).sum(1) / (wm.sum(1) + 1e-8)
        max_p  = (attended - (1.0 - wm) * 1e9).max(dim=1).values.nan_to_num(0.0)
        return torch.cat([mean_p, max_p], dim=-1)

    def forward(self, joint_ids, joint_mask, passage_wids,
                q_ids, q_mask, word_gaze, word_mask, global_stats):
        W        = word_mask.size(1)
        jh       = self._enc(joint_ids, joint_mask)
        text_cls = jh[:, 0, :]
        wt       = self._scatter(jh, passage_wids, W)
        fused    = self.word_fusion(torch.cat([wt, self.gaze_proj(word_gaze)], -1))
        sa, _    = self.self_attn(fused, fused, fused, key_padding_mask=~word_mask.bool())
        fused    = self.ln(fused + sa.nan_to_num(0.0))
        pooled   = self._mean_max_pool(fused, word_mask)
        gate     = torch.sigmoid(self.gaze_gate(text_cls))
        pooled   = gate * pooled
        return self.classifier(torch.cat([text_cls, pooled, global_stats], -1)).squeeze(-1)


class NoGazeStatsModel(WGQModel):
    """A3: Zero out global gaze statistics — keep word-level IA + gate."""
    def forward(self, joint_ids, joint_mask, passage_wids,
                q_ids, q_mask, word_gaze, word_mask, global_stats):
        return super().forward(joint_ids, joint_mask, passage_wids,
                               q_ids, q_mask, word_gaze, word_mask,
                               torch.zeros_like(global_stats))


class ZeroGazeModel(WGQModel):
    """A4: Zero out word-level IA features."""
    def forward(self, joint_ids, joint_mask, passage_wids,
                q_ids, q_mask, word_gaze, word_mask, global_stats):
        return super().forward(joint_ids, joint_mask, passage_wids,
                               q_ids, q_mask, torch.zeros_like(word_gaze),
                               word_mask, global_stats)


class NoGateModel(WGQModel):
    """A5: Remove gaze gate — keeps mean+max pooling but no adaptive gating.
    Isolates the gate's contribution from the pooling change."""
    def forward(self, joint_ids, joint_mask, passage_wids,
                q_ids, q_mask, word_gaze, word_mask, global_stats):
        W = word_mask.size(1)
        joint_hidden = self._encode(joint_ids, joint_mask)
        text_cls     = joint_hidden[:, 0, :]
        word_text    = self._scatter_word_embs(joint_hidden, passage_wids, W)
        q_hidden     = self._encode(q_ids, q_mask)
        q_proj       = self.q_proj(q_hidden)
        gaze_emb     = self.gaze_proj(word_gaze)
        fused        = self.word_fusion(torch.cat([word_text, gaze_emb], dim=-1))
        q_key_pad    = ~q_mask.bool()
        att_out, _   = self.cross_attn(fused, q_proj, q_proj, key_padding_mask=q_key_pad)
        attended     = self.post_attn_ln(fused + self.attn_drop(att_out.nan_to_num(0.0)))
        # Mean+max pool WITHOUT gate
        pooled       = self._mean_max_pool(attended, word_mask)
        combined     = torch.cat([text_cls, pooled, global_stats], dim=-1)
        return self.classifier(combined).squeeze(-1)


def build_model(variant: str = 'full_wgq', **kwargs) -> nn.Module:
    text_only_kwargs = {k: v for k, v in kwargs.items()
                        if k in ('dropout', 'roberta_name', 'unfreeze_top_layers')}
    mapping = {
        'full_wgq':      (WGQModel,         kwargs),
        'no_q_cond':     (NoQCondModel,     kwargs),
        'text_only':     (TextOnlyModel,    text_only_kwargs),
        'no_gaze_stats': (NoGazeStatsModel, kwargs),
        'no_word_gaze':  (ZeroGazeModel,    kwargs),
        'no_gate':       (NoGateModel,      kwargs),
    }
    if variant not in mapping:
        raise ValueError(f'Unknown variant: {variant!r}. Choose from {list(mapping.keys())}')
    cls, kw = mapping[variant]
    return cls(**kw)
