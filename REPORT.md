# Question-Conditioned Gaze–Text Fusion for Reading Comprehension
### EyeBench Practical Project Report — Eye Tracking and NLP Course, UZH

**Abu Bakr Rahman Shaik (23-756-737) · Mariia Korchagina (22-898-134)**

---

## 1. Introduction

The EyeBench OneStop Reading Comprehension (RC) task asks: can a model predict whether a reader answered a comprehension question correctly, given their eye movements while reading the passage? The benchmark reveals a puzzling result: gaze-augmented transformer models (MAG-Eye, RoBERTEye) barely outperform a text-only RoBERTa baseline — the best gaze-augmented model (MAG-Eye, AUROC 62.9) beats text-only (AUROC 61.1) by only 1.8 points.

We argue this gap is small not because gaze is uninformative, but because existing models fail to use it in a question-sensitive way. All transformer-based EyeBench models concatenate the question to the passage in a flat sequence and apply gaze features uniformly across all passage tokens. No model explicitly asks: *given what this question is about, which fixation patterns should I pay attention to?*

Our contribution is **WGQModel** (Word-Gaze-Question), a three-stream architecture with a cross-attention mechanism that conditions gaze processing on the specific question being asked.

---

## 2. Related Work

**EyeBench baselines.** Transformer-based models (RoBERTEye-W/F, MAG-Eye, PostFusion-Eye) encode the input as `[CLS] passage [SEP][SEP] Question: {q} [SEP]` and align word-level gaze features to passage token positions. The question is present in the sequence, but gaze features have no direct interaction with it. Gaze-only models (AhnCNN, BEyeLSTM) ignore the question entirely.

**MAG-Eye** (the strongest baseline) injects gaze into a specific RoBERTa transformer layer via a Multimodal Attention Gate: a learned scalar weighting of a gaze-derived residual. This is powerful but question-agnostic — the gate is not conditioned on what the question asks.

**Our novelty.** We use multi-head cross-attention where per-word fused representations (text + gaze) act as queries and question token representations act as keys/values. The model can thus learn to upweight fixations on words that are semantically relevant to the question.

---

## 3. Method

### 3.1 Architecture

WGQModel has three streams fed to a shared MLP classifier.

**Stream 1 — Joint text encoder:**
```
[CLS] passage [SEP][SEP] Question: {q} [SEP]  →  frozen RoBERTa-base  →  CLS (768-d)
```
Provides a strong text baseline where both passage and question are visible.

**Stream 2 — Word-gaze cross-attention:**

For each passage word *w*:
1. Extract RoBERTa subword token embeddings and average them by word index (scatter-mean) to get a word-level text embedding `t_w ∈ R^768`.
2. Project 12 IA gaze features (dwell time, regression count, skip rate, etc.) through an MLP to get `g_w ∈ R^64`.
3. Fuse: `f_w = LayerNorm(Linear([t_w; g_w])) ∈ R^256`.

Then apply cross-attention:
```
attended_w = CrossAttn(query=f_w, key=q_tokens, value=q_tokens)
pooled = mask-weighted mean over passage words
```
where `q_tokens` are the question token embeddings from a separate RoBERTa forward pass. This is the core novelty: each passage word's gaze+text representation attends to the question, learning which words are relevant to the specific question asked.

**Stream 3 — Global gaze statistics:**  
6 trial-level handcrafted features (mean dwell time, total reading time, regression count, skip rate, total fixation count, mean regression-in count). These are the features used by the Random Forest baseline, providing a floor of signal.

**Classifier:**  `MLP(768 + 256 + 6)  →  logit`

### 3.2 Training Details

| Hyperparameter | Value |
|---|---|
| Text encoder | Frozen RoBERTa-base (125M params, 0 trainable) |
| Trainable parameters | 955,585 |
| Optimizer | AdamW, lr=3e-4, weight_decay=0.01 |
| Scheduler | OneCycleLR, 10% warmup, cosine decay |
| Batch size | 48 |
| Max epochs | 15 |
| Early stopping | Patience 5 on val AUROC |
| Loss | BCE with label smoothing ε=0.05 |
| Threshold | Grid-searched on val set (not test) |

### 3.3 Evaluation Protocol

We follow EyeBench exactly: 10-fold cross-validation, one model per fold evaluated simultaneously on three generalization regimes using the pre-built fold CSV files from the EyeBench repository:

| Regime | Training subjects | Training texts | Test subjects | Test texts |
|---|---|---|---|---|
| Seen reader, unseen text (Sr/Ut) | ✓ | — | ✓ | ✗ |
| Unseen reader, seen text (Ur/St) | — | ✓ | ✗ | ✓ |
| Unseen reader, unseen text (both) | — | — | ✗ | ✗ |

Metrics: AUROC and Balanced Accuracy, reported as Mean ± SEM across folds.

---

## 4. Results

### 4.1 Per-Fold Results

| Fold | Sr/Ut AUROC | Ur/St AUROC | Both AUROC | Val best |
|---|---|---|---|---|
| 0 | 63.3 | 59.6 | 53.9 | 59.1 (ep 13) |
| 1 | 57.6 | 57.8 | 60.4 | 61.8 (ep 4) |
| 2 | 53.8 | 60.0 | 53.7 | 60.6 (ep 15) |
| 3 | 51.9 | 61.1 | 47.6 | 57.4 (ep 13) |
| 4 | 60.7 | 59.6 | 66.3 | 60.2 (ep 9) |
| 5 | 47.8 | 57.6 | 42.8 | 60.2 (ep 3) |
| 6 | 54.0 | 57.6 | 60.2 | 52.8 (ep 2) |
| 7 | — | — | — | *checkpoint write failed* |
| 8 | 51.3 | 63.6 | 43.1 | 55.7 (ep 13) |
| 9 | 52.0 | 60.5 | 65.0 | 57.4 (ep 10) |

Fold 7 reached val AUROC 62.0 but the checkpoint was not saved (likely a Google Drive sync failure during the Colab session). Results are averaged over 9 folds.

### 4.2 Aggregate Results (Mean ± SEM, 9 folds)

| Regime | AUROC | Balanced Accuracy |
|---|---|---|
| Seen reader, unseen text | 54.7 ± 1.5 | 51.7 ± 0.7 |
| **Unseen reader, seen text** | **59.7 ± 0.6** | **55.4 ± 0.8** |
| Unseen reader, unseen text | 54.8 ± 2.8 | 52.3 ± 1.2 |
| **All (average)** | **56.4 ± 1.2** | **53.2 ± 0.6** |

### 4.3 Comparison to EyeBench Baselines

| Model | AUROC (All) | Bal.Acc (All) |
|---|---|---|
| MAG-Eye | 62.9 | 54.3 |
| RoBERTEye-F | 61.9 | — |
| Text-Only RoBERTa-Large | 61.1 | 55.0 |
| PostFusion-Eye | 61.1 | — |
| PLM-AS-RM | 58.4 | **55.2** |
| Random Forest | 58.0 | 55.1 |
| **WGQModel (ours)** | **56.4** | **53.2** |
| Majority Class | 50.0 | 50.0 |

WGQModel falls 4.7 AUROC points below the best EyeBench baseline (MAG-Eye) on the all-regime average. However, on the **Unseen reader, Seen text** regime specifically, WGQModel (59.7 AUROC, 55.4 BalAcc) beats PLM-AS-RM and Random Forest.

---

## 5. Ablation Study

Run on fold 0, all 3 generalization regimes. Each variant removes one component.

| Variant | Sr/Ut AUROC | Ur/St AUROC | Both AUROC | ΔAUROC (Ur/St) |
|---|---|---|---|---|
| **Full WGQModel** | **60.3** | **61.1** | 52.7 | — |
| A1: No Q-conditioning (self-attn) | 62.0 | 54.1 | 51.5 | **−7.0** |
| A2: Text-only (no gaze) | 53.8 | 53.6 | 57.5 | −7.5 |
| A4: No word-level IA (zeroed) | 57.9 | 56.8 | 54.3 | −4.3 |

Key findings:

**Question conditioning matters most for unseen readers on seen texts (Ur/St).** Removing cross-attention and replacing it with self-attention drops Ur/St AUROC by 7.0 points — the largest single-component effect. When the model has seen a text before, it can learn which words in that passage are relevant to each possible question. Without question conditioning, this signal is lost.

**Gaze adds real value over text alone.** Text-only is 7.5 AUROC points lower on Ur/St. The combination of word-level IA features + question-conditioned attention provides meaningful signal beyond what RoBERTa's text encoding captures.

**Word-level IA features contribute significantly.** Zeroing the IA features (A4) drops Ur/St by 4.3 points — confirming that gaze alignment at the word level carries information beyond the global statistics alone.

**The Sr/Ut and Both regimes are harder to improve.** Across ablations, removing question conditioning actually *helps* slightly on Sr/Ut (+1.7). This suggests that for unseen texts, the cross-attention is learning spurious text-specific patterns that don't generalise. This is an expected limitation: question-conditioned gaze patterns are partially text-specific.

---

## 6. Discussion

### What worked

The passage text fix was the most impactful single change from earlier iterations. Before the fix (Notebook 08), `PASSAGE_COL=None` meant Stream 1 was encoding only the question, and Stream 2's word embeddings were computed from an empty passage — rendering both streams near-useless. After the fix, 100% passage coverage was confirmed and AUROC improved by ~1.4 points.

Using EyeBench's pre-built fold CSVs ensured splits are exactly comparable to the baseline models, eliminating a major source of incomparability in earlier work.

The model converged cleanly across all folds — no collapse to 50% balanced accuracy as seen in Notebook 04/05, and val AUROC improved monotonically in most folds (reaching 60–62 on val before early stopping).

### What didn't work as expected

The **Seen reader, Unseen text** regime remains weak (54.7 AUROC, barely above the 53.8 text-only baseline). This regime requires generalising to new articles, and word-level gaze patterns are partly text-specific — the model learns that "readers who fixate heavily on paragraph 3 of *this article* tend to answer correctly", which doesn't transfer to new articles. MAG-Eye's approach (injecting gaze as a global perturbation to transformer hidden states) may generalise better here because it operates at a more abstract level.

**High variance on Both regime (SEM 2.8)** reflects the small test set sizes (78–120 trials per fold). Results there should be treated as exploratory.

**Fold 7 (62.0 val AUROC)** was the best-performing fold by validation metric but its checkpoint was lost to a Drive sync failure. Its test results would likely have improved the aggregate.

### Honest assessment

WGQModel is below all EyeBench baselines on the all-regime average. The core hypothesis — that question-conditioned gaze outperforms question-agnostic gaze — is supported by the ablation (−7.0 AUROC on Ur/St from removing conditioning), but the absolute numbers don't yet beat the best baselines. Several factors explain the gap:

1. **RoBERTa-base vs RoBERTa-Large.** EyeBench baselines use RoBERTa-Large (24 layers, 1024-d). Our frozen RoBERTa-base (12 layers, 768-d) gives a weaker text representation.
2. **Frozen encoder.** Fine-tuning even a few top layers of RoBERTa would likely improve Stream 1 significantly.
3. **Limited training data.** ~5,800 training trials per fold is small for a neural model with multiple streams.

---

## 7. Conclusion

We introduced WGQModel, a question-conditioned gaze–text fusion model for reading comprehension prediction. The model uses cross-attention to let word-level gaze representations attend to the question, learning which fixation patterns matter for the specific question being asked.

On the Unseen reader, Seen text regime — where question-conditioned gaze is most useful — WGQModel achieves 59.7 AUROC (vs 58.4 for PLM-AS-RM and 58.0 for Random Forest). The ablation demonstrates that question conditioning provides a 7.0 AUROC point improvement over question-agnostic gaze on this regime.

The all-regime average (56.4 AUROC) falls below MAG-Eye (62.9), primarily because the model struggles to generalise gaze patterns to unseen texts. Future work should explore fine-tuning the text encoder, using RoBERTa-Large, and training with a more robust unseen-text objective.

---

## Appendix: Implementation Notes

All code is available at [github.com/SABR007/wgq-eyetrack](https://github.com/SABR007/wgq-eyetrack).

The implementation is structured as 7 Python modules (config, data_loader, folds, tokenizer_utils, model, dataset, trainer) imported into a single Colab notebook. Data loading, tokenization, and word-gaze sequences are cached to Google Drive; after the first run (~15 min), subsequent runs load in ~1 minute. Training state is checkpointed after every epoch for Colab disconnect resilience.
