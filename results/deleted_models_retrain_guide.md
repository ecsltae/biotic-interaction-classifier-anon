# Deleted Models — Retraining Guide

Models deleted on 2026-03-02 to free disk space.
All can be retrained from existing training data files.

## Environment
```bash
source /path/to/MetaP/MPvenv/bin/activate
cd /path/to/MetaP
```

---

## Kept Models (DO NOT DELETE)

| Directory | Dataset | EP Test F1 | Notes |
|-----------|---------|-----------|-------|
| `transformer_BiomedBERT_cv_regularized` | v7 (globi_v7_llm) | **0.788** | Gold standard baseline |
| `transformer_SciBERT_cv_regularized` | v7 | 0.774 | Best precision of discriminative |
| `transformer_BiomedBERT_v11_regularized` | v11 | 0.745 | |
| `transformer_BiomedBERT_v11_1` | v11.1 | 0.747 | Most recent discriminative |
| `transformer_BiomedBERT_v12_regularized` | v12 | 0.729 | |
| `flan_t5_v10` | v10 | 0.779 | Generative FLAN-T5-large |
| `flan_t5_v10.1` | v10.1 | **0.804** | Best overall to date |
| `flan_t5_v12` | v12 | TBD | Trained 2026-03-02 |

---

## Deleted Models — How to Retrain

### BiomedBERT / SciBERT (discriminative, regularized CV)

Script: `classifier/scripts/train_cv_regularized.py`

```bash
# v7 BiomedBERT (gold standard, F1=0.788)
python classifier/scripts/train_cv_regularized.py \
  --train-data classifier/data/training/training_data_globi_v8.csv \
  --models BiomedBERT --suffix cv_regularized

# v7 SciBERT (F1=0.774)
python classifier/scripts/train_cv_regularized.py \
  --train-data classifier/data/training/training_data_globi_v8.csv \
  --models SciBERT --suffix cv_regularized

# v10 BiomedBERT (F1=0.722)
python classifier/scripts/train_cv_regularized.py \
  --train-data classifier/data/training/training_data_v10.csv \
  --models BiomedBERT --suffix v10_cv_regularized

# v10.1 BiomedBERT (F1=0.717)
python classifier/scripts/train_cv_regularized.py \
  --train-data classifier/data/training/training_data_v10.1.csv \
  --models BiomedBERT --suffix v10.1

# v11 BiomedBERT (F1=0.745)
python classifier/scripts/train_cv_regularized.py \
  --train-data classifier/data/training/training_data_v11.csv \
  --models BiomedBERT --suffix v11_regularized

# v12 BiomedBERT (F1=0.729)
python classifier/scripts/train_cv_regularized.py \
  --train-data classifier/data/training/training_data_v12.csv \
  --models BiomedBERT --suffix v12_regularized

# v11.1 BiomedBERT (F1=0.747)
python classifier/scripts/train_cv_regularized.py \
  --train-data classifier/data/training/training_data_v11_1.csv \
  --models BiomedBERT --suffix v11_1

# v8 BiomedBERT (F1=0.695, standard script — not regularized)
python classifier/src/models/transformer_classifier.py \
  --model biobert --epochs 5 \
  --data classifier/data/training/training_data_globi_v8.csv
```

### LUKE (not competitive — EP F1 unknown, likely below BiomedBERT)

```bash
# luke, luke_diverse, luke_classifier were LUKE-based models
python classifier/src/models/luke_classifier.py \
  --train-data classifier/data/training/training_data_v11_1.csv
```

### Old BiomedBERT experiments (globi v1–v8, quality_v2/v3, etc.)

These were earlier experiments before the regularized CV script.
Use the regularized script above with the corresponding versioned CSV instead.

```bash
# Available versioned datasets:
# classifier/data/training/training_data_v10.csv
# classifier/data/training/training_data_v10.1.csv
# classifier/data/training/training_data_v12.csv
# classifier/data/training/training_data_v11_1.csv
# classifier/data/training/training_data_globi_v8.csv  (≈ v7 LLM-validated)
```

### FLAN-T5 generative (seq2seq)

Script: `classifier/src/models/flan_t5_classifier.py`

```bash
# FLAN-T5-large on any dataset version
python classifier/src/models/flan_t5_classifier.py \
  --model google/flan-t5-large \
  --train-data classifier/data/training/training_data_v11_1.csv \
  --output-dir classifier/models/flan_t5_v11_1 \
  --results-dir classifier/results/flan_t5_v11_1 \
  --epochs 5 --batch-size 16

# FLAN-T5-base (250M, faster)
python classifier/src/models/flan_t5_classifier.py \
  --model google/flan-t5-base \
  --train-data classifier/data/training/training_data_v11_1.csv \
  --output-dir classifier/models/flan-t5-base_v11_1 \
  --results-dir classifier/results/flan-t5-base_v11_1 \
  --epochs 5 --batch-size 16
```

### BioGPT (causal LM, new script)

Script: `classifier/src/models/biogpt_classifier.py`

```bash
python classifier/src/models/biogpt_classifier.py \
  --train-data classifier/data/training/training_data_v11_1.csv \
  --output-dir classifier/models/biogpt_v11_1 \
  --results-dir classifier/results/biogpt_v11_1 \
  --epochs 5 --batch-size 16
```

### Precision ensemble (old, F1=0.862 on unknown eval set — likely biased)

```bash
# Was a BiomedBERT+RoBERTa 50/50 weighted ensemble
# To recreate: train both models, then use ensemble_classifier.py
python classifier/src/models/ensemble_classifier.py \
  --models transformer_BiomedBERT_cv_regularized transformer_roberta_model \
  --weights 0.5 0.5
```

---

## Disk freed
~115 GB freed by this cleanup.
