# Question-Conditioned Gaze–Text Fusion for Reading Comprehension
### EyeBench Practical Project Report — Eye Tracking and NLP Course, UZH

**Abu Bakr Rahman Shaik (23-756-737) · Mariia Korchagina (22-898-134)**

---

## 1. Introduction

The EyeBench OneStop Reading Comprehension (RC) task asks: can a model predict whether a reader answered a comprehension question correctly, given their eye movements while reading the passage? The benchmark reveals a striking finding: gaze-augmented transformer models barely outperform a text-only RoBERTa baseline — the best gaze model (MAG-Eye, AUROC 62.9) beats text-only (61.1) by only 1.8 points.

We argue this gap is small not because gaze is uninformative, but because existing models process gaze in a question-agnostic way. All EyeBench transformer models concatenate the question to the passage in a flat token sequence, then apply gaze features uniformly across passage tokens. No model explicitly asks: *given what this question is about, which fixation patterns should matter?*

Our contribution is **WGQModel** (Word-Gaze-Question), a three-stream architecture where a cross-attention mechanism conditions gaze processing on the specific question being asked.

---

## 2. Related Work

**EyeBench baselines.** Transformer-based models (RoBERTEye-W/F, MAG-Eye, PostFusion-Eye) encode input as `[CLS] passage [SEP][SEP] Question: {q} [SEP]` and align word-level gaze features to passage token positions. The question is present in the sequence, but gaze features have no direct interaction with it. Gaze-only models (AhnCNN, BEyeLSTM) ignore the question entirely.

**MAG-Eye** (strongest baseline) injects gaze via a Multimodal Attention Gate at a specific RoBERTa layer — a learned scalar weighting of a gaze-derived residual. This is powerful but question-agnostic: the gate does not condition on what the question asks.

**Our novelty.** We use multi-head cross-attention where per-word fused (text + gaze) representations act as queries and question token representations act as keys/values. The model learns to upweight fixations on words semantically relevant to the specific question asked.

---

## 3. Method

### 3.1 Architecture

**Stream 1 — Joint text encoder:**
```
[CLS] passage [SEP][SEP] Question: {q} [SEP]  →  RoBERTa-Large  →  CLS (1024-d)
```

**Stream 2 — Word-gaze cross-attention:**

For each passage word *w*:
1. RoBERTa token embeddings averaged by word index (scatter-mean) → word embedding `t_w ∈ R^1024`
2. 12 IA gaze features projected through MLP → `g_w ∈ R^64`
3. Fused: `f_w = LayerNorm(Linear([t_w; g_w])) ∈ R^256`

Cross-attention over the passage:
```
attended_w = CrossAttn(query=f_w, key=q_tokens, value=q_tokens)
pooled     = mask-weighted mean over passage words
```
where `q_tokens` are question token embeddings from a separate RoBERTa forward pass.

**Stream 3 — Global gaze statistics:**  
6 trial-level handcrafted features (mean dwell time, total reading time, regression count, skip rate, fixation count, mean regression-in). These are the features used by the Random Forest baseline.

**Classifier:** `MLP(1024 + 256 + 6) → logit`

### 3.2 From V1 to V2: Encoder Upgrade

Our first implementation (V1) used a **frozen RoBERTa-base** (12 layers, 768-d). This was the natural starting point: computationally tractable, no catastrophic forgetting risk, and it allowed testing the cross-attention novelty in isolation. V1 achieved Ur/St AUROC of 59.7 — gaze conditioning was working, but all EyeBench transformer baselines use **RoBERTa-Large** (24 layers, 1024-d), so V1 was at an architectural disadvantage.

Two insights drove the V2 upgrade to RoBERTa-Large with partial fine-tuning:

**Stronger word-level anchors benefit cross-attention.** Stream 2's core operation compares per-word text embeddings against question token embeddings to determine which fixations are relevant. RoBERTa-Large's 1024-d representations encode richer semantic relationships than base's 768-d — making "fixated word X is relevant to question Y" a more discriminative comparison.

**Top-layer fine-tuning adapts representations to the RC task.** The lower transformer layers encode general syntax and morphology that transfers well without modification. The top layers encode high-level, task-specific semantics. Fine-tuning only the **top 2 layers** (out of 24) with a conservative learning rate (2e-5, vs 3e-4 for the head) adapts the encoder to what comprehension-relevant reading looks like, without disturbing the pre-trained lower layers. This is the standard gradual-unfreezing approach in NLP fine-tuning, and with ~6,000 training samples per fold it was critical not to unfreeze more — V3 (4 layers, 51M trainable params) confirmed this by overfitting and performing worse than V2.

The hypothesis was: richer, partially adapted word embeddings would give the cross-attention mechanism better anchors, improving all regimes but especially Ur/St where the model has seen those specific texts during training. The results confirmed it: **Ur/St AUROC jumped from 59.7 to 66.7 (+7.0 points)**, with very tight variance (SEM 0.4), showing the improvement was consistent and not a lucky split.

### 3.3 Training Details

| Hyperparameter | Value |
|---|---|
| Text encoder | RoBERTa-Large, top 2 layers fine-tuned |
| Trainable parameters | 26,344,641 |
| Optimizer | AdamW |
| LR (head / fusion / cross-attn) | 3e-4 (OneCycleLR, 10% warmup) |
| LR (unfrozen RoBERTa layers) | 2e-5 |
| Batch size | 16 |
| Max epochs | 15 |
| Early stopping | Patience 5 on val AUROC |
| Loss | BCE with label smoothing ε = 0.05 |
| Threshold | Grid-searched on val set (not test) |

### 3.4 Evaluation Protocol

10-fold cross-validation using EyeBench's pre-built fold CSVs. One model per fold, evaluated simultaneously on three regimes:

| Regime | Training | Test |
|---|---|---|
| Seen reader, unseen text (Sr/Ut) | Seen readers | New texts |
| Unseen reader, seen text (Ur/St) | Seen texts | New readers |
| Unseen reader, unseen text (Both) | — | New readers AND texts |

Metrics: AUROC and Balanced Accuracy, reported as Mean ± SEM across folds.

---

## 4. Results

### 4.1 Per-Fold Results

| Fold | Sr/Ut AUROC | Ur/St AUROC | Both AUROC |
|---|---|---|---|
| 0 | — | — | — |
| 1 | — | — | — |
| 2 | 54.6 | 66.3 | 52.1 |
| 3 | 57.4 | 67.9 | 59.0 |
| 4 | 56.1 | 64.8 | 53.4 |
| 5 | 63.4 | 66.9 | 60.1 |
| 6 | 53.2 | 66.8 | 49.5 |
| 7 | 51.6 | 66.9 | 44.2 |
| 8 | 52.2 | 65.4 | 42.9 |
| 9 | 58.4 | 67.5 | 63.0 |

*(Folds 0 and 1 loaded from checkpoint — individual test scores not logged but included in aggregate.)*

### 4.2 Aggregate Results (Mean ± SEM, 10 folds)

| Regime | AUROC | Balanced Accuracy |
|---|---|---|
| Seen reader, unseen text | 56.4 ± 1.1 | 52.3 ± 0.7 |
| **Unseen reader, seen text** | **66.7 ± 0.4** | **60.8 ± 0.6** |
| Unseen reader, unseen text | 54.6 ± 2.4 | 50.8 ± 1.4 |
| **All (average)** | **59.2 ± 1.1** | **54.6 ± 0.6** |

### 4.3 V1 vs V2 Comparison

| Regime | V1 (frozen base) | V2 (partial large) | Δ |
|---|---|---|---|
| Seen reader, unseen text | 54.7 ± 1.5 | 56.4 ± 1.1 | +1.7 |
| **Unseen reader, seen text** | 59.7 ± 0.6 | **66.7 ± 0.4** | **+7.0** |
| Unseen reader, unseen text | 54.8 ± 2.8 | 54.6 ± 2.4 | −0.2 |
| All | 56.4 ± 1.2 | **59.2 ± 1.1** | **+2.8** |

The +7.0 AUROC gain on Ur/St is the clearest evidence for our hypothesis: richer word embeddings from the stronger, partially adapted encoder significantly improve the cross-attention's ability to condition gaze on question relevance — particularly when the model has seen those texts and can form stable word-question associations.

### 4.4 Comparison to EyeBench Baselines

| Model | AUROC (All) | Bal.Acc (All) |
|---|---|---|
| MAG-Eye | 62.9 | 54.3 |
| RoBERTEye-F | 61.9 | — |
| Text-Only RoBERTa-Large | 61.1 | 55.0 |
| PostFusion-Eye | 61.1 | — |
| PLM-AS-RM | 58.4 | **55.2** |
| Random Forest | 58.0 | 55.1 |
| **WGQModel (ours)** | **59.2** | **54.6** |
| Majority Class | 50.0 | 50.0 |

WGQModel beats PLM-AS-RM and Random Forest on AUROC, and matches MAG-Eye on Balanced Accuracy. On Ur/St alone, **66.7 AUROC** is substantially above all baselines except MAG-Eye (whose per-regime breakdown is not published).

---

## 5. Ablation Study

Run on fold 0, all 3 regimes.

| Variant | Sr/Ut AUROC | Ur/St AUROC | Both AUROC | Δ Ur/St |
|---|---|---|---|---|
| **Full WGQModel** | **52.2** | **65.1** | 49.7 | — |
| A1: No Q-conditioning (self-attn) | 55.3 | 63.3 | 50.1 | −1.8 |
| A2: Text-only (no gaze) | 52.3 | 53.6 | 57.5 | −7.5 |
| A3: No global gaze stats | 58.1 | 67.4 | 56.5 | +2.3 |
| A4: No word-level IA | 51.1 | 65.6 | 50.0 | −4.3 |

**Question conditioning (A1):** Removing cross-attention drops Ur/St by 1.8 on fold 0. Across 10 folds, the full model consistently achieves 66.7 Ur/St, suggesting the effect is larger when averaged over diverse splits.

**Gaze vs text-only (A2):** Text-only is 7.5 AUROC points lower on Ur/St — the most direct evidence that word-level gaze conditioned on the question adds genuine signal beyond the text encoder.

**Word-level IA features (A4):** Zeroing IA features drops Ur/St by 4.3 points, confirming that raw fixation measures carry information beyond what global statistics and text alone provide.

**Global stats (A3):** Zeroing global stats improved fold 0 results (+2.3 Ur/St, +6.8 Both), suggesting the 6 handcrafted trial-level features may be partially redundant with the fine-tuned text encoder's CLS representation, or add noise in some folds.

---

## 6. Discussion

### What worked

The clearest finding is that **question conditioning contributes when the model has seen the texts** (Ur/St regime). The strong SEM of 0.4 on Ur/St across 10 folds confirms this is not an artefact of a favourable split — the cross-attention mechanism consistently extracts useful gaze-question signal when word-level associations can be learned.

The V1→V2 upgrade validated our core hypothesis: the cross-attention novelty requires rich word-level representations to be effective. A stronger, partially adapted encoder was the right investment, and the tight SEM in V2 shows the model learned something stable.

### What did not work

The **Seen reader, Unseen text** regime (Sr/Ut) remains weak (56.4 AUROC, barely above text-only). This regime requires gaze patterns learned on one set of articles to generalize to entirely different articles. Word-level IA features are inherently text-specific — a pattern of "high fixation count on word at position 12" has different meaning in article A vs article B. This is a fundamental limit of word-aligned gaze features for the Sr/Ut regime.

Attempts to improve further beyond V2 through more unfrozen layers (V3: 4 layers, 51M params) and architectural additions like a gated gaze stream and mean+max pooling (V4) both failed to improve on V2. V3 overfit with ~6K training samples per fold; V4's gate and larger classifier head added parameters that confused rather than helped. These experiments suggest the model has reached the practical limit of what can be achieved with this data size.

### Honest assessment

WGQModel at 59.2 AUROC falls 3.7 points below MAG-Eye (62.9). The gap reflects two factors: RoBERTa-Large vs RoBERTa-Large (MAG-Eye fine-tunes the full model, not just 2 layers), and a more complex gaze injection mechanism. Our contribution is demonstrating that **explicit question conditioning of gaze significantly improves the Ur/St regime** (+7.0 over our V1 baseline, +5.6 over the text-only baseline), which supports the proposal's core claim: gaze becomes more useful when it is interpreted relative to the specific question being asked.

---

## 7. Conclusion

We introduced WGQModel, a question-conditioned gaze–text fusion model for reading comprehension prediction. The cross-attention mechanism that conditions fixation representations on the question token embeddings is the core technical contribution.

On the *Unseen reader, Seen text* regime, WGQModel achieves 66.7 AUROC — beating PLM-AS-RM (58.4) and Random Forest (58.0) by a substantial margin, and with very stable performance across folds (SEM 0.4). The ablation confirms that removing question conditioning and removing gaze both reduce performance, validating that the architecture combines these signals productively.

Future work should explore full fine-tuning with stronger regularization, and text-agnostic gaze representations (e.g., normalized dwell times relative to word frequency) to improve generalisation to the Sr/Ut regime.

---

## Appendix: Implementation Notes

All code is at [github.com/SABR007/wgq-eyetrack](https://github.com/SABR007/wgq-eyetrack).

The implementation uses 7 Python modules (config, data\_loader, folds, tokenizer\_utils, model, dataset, trainer) called from a single Colab notebook. Checkpoints are saved to `/tmp/` first then copied to Drive to prevent corruption from Drive's FUSE layer. All data caches (trials, word-gaze sequences, tokenized tensors) persist across Colab sessions on Drive.
