# Question-Conditioned Gaze–Text Fusion for Reading Comprehension

**EyeBench Practical Project — Eye Tracking Course, UZH**  
Abu Bakr Rahman Shaik (23-756-737) · Mariia Korchagina (22-898-134)

---

## Overview

We propose **WGQModel** (Word-Gaze-Question), a three-stream neural architecture that predicts whether a reader answered a reading comprehension question correctly, using their eye movements. The core novelty is an explicit cross-attention mechanism that conditions how gaze is interpreted on *which question is being asked* — something no existing EyeBench baseline does explicitly.

The model is evaluated on the [EyeBench](https://github.com/EyeBench/eyebench) OneStop Reading Comprehension task using the benchmark's exact 10-fold cross-validation protocol.

---

## Architecture

```
Stream 1  [CLS] passage [SEP][SEP] Question: {q} [SEP]
          → frozen RoBERTa-base → CLS token (768-d)

Stream 2  passage tokens → RoBERTa → scatter-mean per word → (W, 768)
          IA gaze features (dwell time, regressions, skip…) → MLP → (W, 64)
          cat → Linear+LayerNorm → (W, 256)
          cross-attn( query=fused_words, key/value=question_tokens ) → (W, 256)
          mask-weighted pool → (256,)

Stream 3  6 trial-level gaze statistics → (6,)

Classify  MLP(768 + 256 + 6) → logit
```

**Key idea:** Stream 2 cross-attention lets the model learn *which fixation patterns are relevant given the specific question*. A long fixation on paragraph 3 matters if the question is about paragraph 3, but not otherwise.

---

## Results

**10-fold CV, Mean ± SEM (9 of 10 folds completed)**

| Regime | AUROC | Balanced Accuracy |
|---|---|---|
| Seen reader, unseen text | 54.7 ± 1.5 | 51.7 ± 0.7 |
| **Unseen reader, seen text** | **59.7 ± 0.6** | **55.4 ± 0.8** |
| Unseen reader, unseen text | 54.8 ± 2.8 | 52.3 ± 1.2 |
| **All (average)** | **56.4 ± 1.2** | **53.2 ± 0.6** |

**vs EyeBench baselines**

| Model | AUROC | Bal. Acc |
|---|---|---|
| MAG-Eye | 62.9 | 54.3 |
| Text-Only RoBERTa-Large | 61.1 | 55.0 |
| PLM-AS-RM | 58.4 | 55.2 |
| Random Forest | 58.0 | 55.1 |
| **WGQModel (ours)** | **56.4** | **53.2** |

On the *Unseen reader, Seen text* regime specifically, WGQModel (59.7 AUROC) beats PLM-AS-RM (58.4) and Random Forest (58.0).

See [`output_run_experiment.ipynb`](output_run_experiment.ipynb) for full training logs, per-fold results, and ablation tables.

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
└── output_run_experiment.ipynb  # Executed notebook with all results
```

---

## How to Run

### Prerequisites

- Google Colab with **GPU runtime** (A100 recommended; T4 works but ~8-10 hrs)
- Google Drive with ~10 GB free space

### Step 1 — Set up Google Drive

Upload the following to your Drive under `MyDrive/eyebench_project/`:

```
eyebench_project/
├── Project_codebase/       ← contents of code/ from this repo
├── eyebench_codebase/      ← full EyeBench repo (git clone https://github.com/EyeBench/eyebench)
└── data/OneStop/           ← OneStop eye-tracking data (downloaded by EyeBench setup)
```

To download the OneStop data, follow the EyeBench setup instructions:
```bash
git clone https://github.com/EyeBench/eyebench.git
cd eyebench
conda env create -f environment.yml
conda activate eyebench
bash src/data/preprocessing/get_data.sh
```
Then upload the `data/` folder to Drive.

### Step 2 — Configure paths

Open `code/config.py` and verify:

```python
DRIVE             = '/content/drive/MyDrive/eyebench_project'
EYEBENCH_CODEBASE = f'{DRIVE}/eyebench_codebase'
ONESTOP_RAW       = f'{DRIVE}/data/OneStop'
```

### Step 3 — Run the notebook

Open `code/run_experiment.ipynb` in Colab with a GPU runtime.

**Run Cell 1** — installs packages (once per session).

**Run Cell 2** — mounts Drive, imports modules, and loads all data:
- `trials_df`: 9,718 trials with passage text, questions, labels, gaze stats
- `word_gaze`: word-level IA feature sequences keyed by trial ID
- `tokenized`: pre-tokenized passage+question tensors

> **After any runtime restart**, re-run Cell 2. Everything is cached on Drive so it loads in ~1 minute.

**Sections 1–5** proceed in order: verify splits → train 10 folds → aggregate → ablations → summary.

Results are saved incrementally after each fold to `results_v3/fold_results.json`. Training is fully resumable after disconnections.

### Expected runtime

| GPU | Per fold | All 10 folds + ablations |
|---|---|---|
| A100 | ~12–18 min | ~3 hrs |
| T4 | ~45–60 min | ~10 hrs |

---

## Ablation Study

Run on fold 0, all 3 generalization regimes.

| Variant | Sr/Ut AUROC | Ur/St AUROC | Both AUROC |
|---|---|---|---|
| Full WGQModel | 60.3 | **61.1** | 52.7 |
| A1: No Q-conditioning (self-attn) | **62.0** | 54.1 | 51.5 |
| A2: Text-only (no gaze) | 53.8 | 53.6 | 57.5 |
| A4: No word-level IA (zeroed) | 57.9 | 56.8 | 54.3 |

Removing question conditioning drops Ur/St AUROC by **−7.0 points**, confirming that the cross-attention mechanism adds meaningful signal specifically when the model has seen those texts before.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Text encoder | frozen RoBERTa-base |
| Gaze features | 12 word-level IA measures |
| Cross-attention heads | 4 |
| Hidden dim (d) | 256 |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning rate | 3e-4 (OneCycleLR) |
| Batch size | 48 |
| Max epochs | 15 |
| Early stopping patience | 5 |
| Label smoothing | ε = 0.05 |

---

## References

- Shubi et al. (2025). *EyeBench: A Benchmark for Evaluating Predictive Models of Eye Movements in Reading*. NeurIPS Datasets and Benchmarks.
- Berzak et al. (2025). *OneStop Eye Movements*. Nature Scientific Data.
- EyeBench GitHub: https://github.com/EyeBench/eyebench
