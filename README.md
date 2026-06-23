# Biotic Interaction Classifier

Code and data for "Multi-task BiomedBERT for Biotic Interaction Detection via
Knowledge Distillation and Multi-task Learning with Named Entity Recognition"
(submitted for anonymous review). The submission manuscript is
`manuscript/biotic_interaction_classifier_ARR.tex` (official ACL Author Kit,
`\usepackage[review]{acl}`); `biotic_interaction_classifier.tex` is an earlier
non-official-template draft kept for reference.

## System overview

The proposed classifier (`experiments/multitask/model.py`) is a multi-task
BiomedBERT with two heads sharing one encoder: a binary classification head
(biotic interaction / no interaction) and a named entity recognition (NER)
head producing BIO-tagged entity spans (HOST, PATHOGEN, SPECIES, INT). It is
trained via knowledge distillation from soft probability labels (an ensemble
teacher), using warm-start initialization from a template-trained BiomedBERT
encoder. See the manuscript for full architecture and training details.

## Reproducing the paper's results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Unpack the training corpus and test set

```bash
gunzip -k data/training/distillation_44k.csv.gz
gunzip -k data/training/distillation_soft_labels.csv.gz
```

The final 500-sentence test set is `data/evaluation/biotic_interaction_test_set.csv`
(281 positives, 5 sources). It is rebuilt from raw sources (deduplicating two
near-identical export formats, fixing one internal duplicate) via:

```bash
python scripts/rebuild_test_set.py
```

### 3. Train the champion model

```bash
python experiments/multitask/train.py \
  --data data/training/distillation_44k.csv \
  --ner-scheme full_typed --alpha 0.5 --pretrain-ner-epochs 0 --epochs 3 \
  --encoder <path-to-template-trained-BiomedBERT-checkpoint> \
  --output-dir models/multitask/champion \
  --results-dir results/multitask/champion
```

The warm-start `--encoder` checkpoint is a BiomedBERT model fine-tuned on the
template-generated training corpus with standard cross-entropy (the
single-task baseline reported in Table 2 of the paper); see
`src/models/transformer_classifier.py` for that training script. Trained
model weights are not included in this repository (file size); retrain from
the corpus above, or contact the authors after review for the checkpoint.

### 4. Evaluate

```bash
python scripts/eval_corrected_testset.py --gpu
```

Reproduces the point metrics, bootstrap confidence intervals, and McNemar
significance tests reported in Section 3.2 of the paper. Per-source and
ablation evaluation: `scripts/eval_on_new_testset.py`.

## Repository structure

```
classifier/
├── manuscript/              # Paper source (LaTeX), figures, references
├── experiments/multitask/   # Core architecture: model.py, train.py, data.py, evaluate.py
├── scripts/
│   ├── rebuild_test_set.py        # Builds the final 500-sentence test set
│   ├── eval_corrected_testset.py  # Main evaluation harness (CIs, McNemar)
│   ├── eval_on_new_testset.py     # Per-source / ablation evaluation
│   ├── distill_ensemble.py        # Knowledge distillation training
│   └── teacher_scorer.py          # Ensemble soft-label scoring
├── src/models/transformer_classifier.py  # Single-task BiomedBERT baseline
├── data/
│   ├── training/    # distillation_44k.csv.gz (+ raw soft labels), dataset version history
│   └── evaluation/  # Final test set + raw per-source files used to build it
└── results/         # Saved metrics (train_summary.json, eval JSON) per model config
```

## Data

- `data/training/distillation_44k.csv.gz`: the 44K-sentence distillation
  corpus (Qwen3.5-122B binary labels + ensemble soft probability labels).
  Gzipped to stay under file-size limits; `gunzip` before use.
- `data/evaluation/biotic_interaction_test_set.csv`: the final 500-sentence
  test set (5 sources, 281 positives), built by `rebuild_test_set.py` from
  the raw `.tsv`/`.csv` files alongside it in the same directory.

## Contact

[contact details removed for anonymous review]
