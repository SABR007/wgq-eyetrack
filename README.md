# Question-Conditioned Gaze–Text Fusion for Reading Comprehension

**EyeBench Practical Project — Eye Tracking and NLP Course, UZH**  
Abu Bakr Rahman Shaik (23-756-737) · Mariia Korchagina (22-898-134)

---

## Overview

We propose **WGQModel** (Word-Gaze-Question), a three-stream neural architecture that predicts whether a reader answered a reading comprehension question correctly from their eye movements. The core novelty is a cross-attention mechanism that conditions how gaze is interpreted on *which question is being asked* — something no existing EyeBench baseline does explicitly.

Evaluated on the [EyeBench](https://github.com/EyeBench/eyebench) OneStop Reading Comprehension task using the benchmark's exact 10-fold cross-validation protocol and pre-built fold splits.

---

## Architecture

```
Stream 1  [CLS] passage [SEP][SEP] Question: {q} [SEP]
          → RoBERTa-Large (top 2 layers fine-tuned) → CLS (1024-d)

Stream 2  passage tokens → RoBERTa → scatter-mean per word → (W, 1024)
          IA gaze features (dwell time, regressions, skip…) → MLP → (W, 64)
          cat → LayerNorm → (W, 256)
          cross-attn( query=fused_words, key/value=question_tokens ) → (W, 256)
          mask-weighted mean pool → (256,)

Stream 3  6 trial-level gaze statistics → (6,)

Classify  MLP(1024 + 256 + 6) → logit
```

**Key idea:** Stream 2 cross-attention lets the model learn *which fixation patterns are relevant given the specific question*. A long fixation on paragraph 3 matters if the question is about paragraph 3, but not otherwise.

---

## Results

**10-fold CV, Mean ± SEM (all 10 folds)**

| Regime | AUROC | Balanced Accuracy |
|---|---|---|
| Seen reader, unseen text | 56.4 ± 1.1 | 52.3 ± 0.7 |
| **Unseen reader, seen text** | **66.7 ± 0.4** | **60.8 ± 0.6** |
| Unseen reader, unseen text | 54.6 ± 2.4 | 50.8 ± 1.4 |
| **All (average)** | **59.2 ± 1.1** | **54.6 ± 0.6** |

**Comparison to EyeBench baselines**

| Model | AUROC (All) | Bal. Acc (All) |
|---|---|---|
| MAG-Eye | 62.9 | 54.3 |
| Text-Only RoBERTa-Large | 61.1 | 55.0 |
| PLM-AS-RM | 58.4 | 55.2 |
| Random Forest | 58.0 | 55.1 |
| **WGQModel (ours)** | 59.2 | 54.6 |

WGQModel beats PLM-AS-RM and Random Forest on AUROC, and matches MAG-Eye on Balanced Accuracy. On the *Unseen reader, Seen text* regime specifically, **66.7 AUROC** beats all non-MAG-Eye baselines.

---

## From V1 to V2: Why We Upgraded the Encoder

Our first implementation (V1) used a **frozen RoBERTa-base** (12 layers, 768-d). This was the natural starting point: computationally light, no catastrophic forgetting risk, and it let us test the cross-attention novelty in isolation. V1 achieved Ur/St AUROC of 59.7 — gaze conditioning was working, but all EyeBench transformer baselines (MAG-Eye, RoBERTEye, PLM-AS) use **RoBERTa-Large** (24 layers, 1024-d), meaning V1 was comparing a weaker text encoder against stronger ones.

Two insights drove the V2 upgrade:

**1. Stronger word-level anchors benefit the cross-attention.** Stream 2's core operation aligns fixation patterns to question relevance by comparing per-word text embeddings against question token embeddings. With a 768-d base encoder, both embeddings are relatively coarse. RoBERTa-Large's 1024-d representations are richer, especially for semantic similarity — so "the reader fixated on word X" and "word X is relevant to the question" becomes a more discriminative signal.

**2. Partial fine-tuning adapts the top layers to the RC task.** The lower transformer layers encode general syntax and morphology — these transfer well without modification. The top layers encode task-specific, high-level semantics. Fine-tuning only the top 2 layers (out of 24) with a low learning rate (2e-5 vs 3e-4 for the head) adapts the representation to what "comprehension-relevant reading" looks like, without disturbing the general-purpose lower layers. This is the standard "gradual unfreezing" practice in NLP fine-tuning.

V2 confirmed both intuitions strongly: **Ur/St AUROC jumped from 59.7 to 66.7 (+7.0 points)**, with very stable behaviour across folds (SEM 0.4 vs 0.6 in V1). The stronger encoder gave Stream 2's cross-attention exactly the richer representations it needed.

---

## Ablation Study

Run on fold 0, all 3 generalization regimes.

| Variant | Sr/Ut AUROC | Ur/St AUROC | Both AUROC | Δ Ur/St |
|---|---|---|---|---|
| **Full WGQModel** | **52.2** | **65.1** | 49.7 | — |
| A1: No Q-conditioning | 55.3 | 63.3 | 50.1 | −1.8 |
| A2: Text-only (no gaze) | 52.3 | 53.6 | 57.5 | −7.5 |
| A3: No global gaze stats | 58.1 | 67.4 | 56.5 | +2.3 |
| A4: No word-level IA | 51.1 | 65.6 | 50.0 | −4.3 |

Removing question conditioning drops Ur/St by **−1.8** on fold 0, confirming the cross-attention's contribution. Across the full 10-fold comparison, the gap is larger: the consistent 66.7 Ur/St of the full model vs the weaker, less stable performance of ablated variants.

---

## Repository Structure

```
wgq-eyetrack/
├── code/
│   ├── config.py              # All paths and hyperparameters
│   ├── data_loader.py         # IA CSV → trials_df + word_gaze_sequences
│   ├── folds.py               # EyeBench fold CSV loader
│   ├── tokenizer_utils.py     # RoBERTa pre-tokenization
│   ├── model.py               # WGQModel + 4 ablation variants
│   ├── dataset.py             # PyTorch Dataset + DataLoader factory
│   ├── trainer.py             # Training loop, evaluation, aggregation
│   └── run_experiment.ipynb   # Main Colab notebook (run this)
├── output_v1_run_experiment.ipynb   # V1 executed notebook (frozen RoBERTa-base)
└── output_v2_run_experiment.ipynb   # V2 executed notebook (final results)
```

---

## How to Run

### Prerequisites

- Google Colab with **GPU runtime** (A100 recommended; ~3 hrs total)
- Google Drive with ~10 GB free space

### Step 1 — Set up Google Drive

Upload the following to `MyDrive/eyebench_project/`:

```
eyebench_project/
├── Project_codebase/       ← contents of code/ from this repo
├── eyebench_codebase/      ← EyeBench repo (git clone https://github.com/EyeBench/eyebench)
└── data/OneStop/           ← OneStop eye-tracking data (via EyeBench setup)
```

To download data:
```bash
git clone https://github.com/EyeBench/eyebench.git
cd eyebench && conda env create -f environment.yml && conda activate eyebench
bash src/data/preprocessing/get_data.sh
```

### Step 2 — Configure paths

In `code/config.py`, verify:
```python
DRIVE             = '/content/drive/MyDrive/eyebench_project'
EYEBENCH_CODEBASE = f'{DRIVE}/eyebench_codebase'
ONESTOP_RAW       = f'{DRIVE}/data/OneStop'
```

### Step 3 — Run the notebook

Open `code/run_experiment.ipynb` in Colab (GPU runtime).

**Cell 1** — install packages.

**Cell 2** — mount Drive + load all data. Re-run this after any runtime restart; everything loads from cache in ~1 minute.

**Sections 1–5** — verify splits → train 10 folds → aggregate → ablations → summary.

Results save after every fold to `results_large_unfreeze2/fold_results.json`. Fully resumable after disconnects.

| GPU | Per fold | Full run |
|---|---|---|
| A100 | ~12–18 min | ~3 hrs |
| T4 | ~45–60 min | ~10 hrs |

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Text encoder | RoBERTa-Large, top 2 layers fine-tuned |
| Trainable parameters | 26,344,641 |
| Optimizer | AdamW |
| LR (head / fusion / cross-attn) | 3e-4 (OneCycleLR) |
| LR (unfrozen RoBERTa layers) | 2e-5 |
| Batch size | 16 |
| Max epochs | 15 |
| Early stopping patience | 5 (val AUROC) |
| Label smoothing | ε = 0.05 |

---

## References

- Shubi et al. (2025). *EyeBench: A Benchmark for Evaluating Predictive Models of Eye Movements in Reading*. NeurIPS Datasets and Benchmarks.
- Berzak et al. (2025). *OneStop Eye Movements*. Nature Scientific Data.
- EyeBench: https://github.com/EyeBench/eyebench
