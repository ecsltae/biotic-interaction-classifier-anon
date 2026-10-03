# Which Pair Is It About? Pair-Conditioned Verification of Literature-Mined Biotic Interactions

Anonymous code release for the submission. It holds the manuscript, the code that trains the
compared models and computes every table, and the computed outputs (`results/paperA_v2/`), so
the tables can be checked without retraining.

## Layout

| Path | What it is |
|---|---|
| `paperA/` | manuscript (`paperA.tex`, `references.bib`, ACL style files, compiled `paperA.pdf`) and the figure script |
| `experiments/multitask/train_student.py` | trains one verifier; `--input-format sentence\|pair\|triple` is the only difference between the compared arms |
| `scripts/xenc_format.py` | builds the encoder input of each format (passage only; pair + passage; pair + relation + passage) |
| `scripts/eval_unified.py` | scores a checkpoint on a candidate table (used by the two scripts below) |
| `scripts/paperA_tables.py` | every number of the results tables, from trained checkpoints |
| `scripts/biored_ep_eval.py` | BioRED entity-pair F1 (the standard document-level metric) |
| `scripts/llm_baseline.py` | zero-shot local LLM rows (Ollama), same three formulations as the trained arms |
| `scripts/teacher_label_triples.py` | the teacher's labelling prompt (also read by `llm_baseline.py` for the triple question) |
| `scripts/convert_biored_verify.py` | builds the sentence-scope BioRED candidate tables (BC8 protocol) |
| `scripts/convert_biored_doc.py` | builds the whole-abstract BioRED candidate tables |
| `scripts/run_biored_bc8.sh`, `scripts/run_biored_bc8_doc.sh` | train the BioRED arms (3 seeds each) |
| `scripts/candidate_rules.py` | the deterministic candidate rules (Section "Candidate rules" and its appendix) |
| `scripts/scan_contamination.py` | near-duplicate scan of the benchmark against the training passages (defines the 437 clean rows) |
| `src/eval/core.py` | the single loader of the 437 clean benchmark rows, used by every script |
| `results/paperA_v2/` | computed outputs: `tables_biodiv.json`, `tables_biored.json`, `biored_ep.json`, and the per-row ensemble scores `S_*.npy` |

## Data

* **BioRED.** Public. `convert_biored_verify.py` reads the PubTator files distributed with the
  BioREDirect release (Lai et al., 2025): `bioredirect_{train_dev,test,bc8_test}.pubtator`
  (BioRED train+dev, BioRED test, and the BioCreative VIII BioRED-track test set of 400 abstracts).
  Put them in `data/raw/bioredirect/`.
* **Biodiversity benchmark and training corpus.** The 449-row benchmark
  (`data/evaluation/unified_test_set.csv`, of which 437 rows are kept after the contamination scan
  `results/test_contamination_scan.csv`; per-row taxon counts in
  `results/paperA_rebuild_2026-09-28/n_taxa_per_row.csv`) and the 48,338-row teacher-labelled
  training corpus (`data/training/distill/v3_combined_train.csv`) will be released with the paper.
  They are not part of this anonymous package.

## Environment

Python 3.10; a GPU for training.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The LLM rows need [Ollama](https://ollama.com) on `localhost:11434` with the Qwen3 models pulled
(`ollama pull qwen3:32b`, and the smaller sizes for the scaling table).

## Reproducing the results

```bash
# 1. BioRED candidate tables (sentence scope, BC8 protocol; and whole abstract)
python3 scripts/convert_biored_verify.py --protocol bc8 --src data/raw/bioredirect --out data/benchmarks/biored_bc8
python3 scripts/convert_biored_doc.py --src data/raw/bioredirect --out data/benchmarks/biored_bc8_doc

# 2. Train the arms: three seeds per arm, one shared recipe (3 epochs, lr 2e-5, batch 32)
for fmt in sentence pair triple; do for s in 1 2 3; do
  python3 experiments/multitask/train_student.py --data data/training/distill/v3_combined_train.csv \
      --input-format $fmt --seed $s --out models/biodiv/${fmt}_s$s
done; done
bash scripts/run_biored_bc8.sh          # BioRED sentence / pair / marker variants
bash scripts/run_biored_bc8_doc.sh      # BioRED whole-abstract pair arm

# 3. Tables (edit ARMS_BIODIV / ARMS_BIORED at the top of paperA_tables.py if your
#    checkpoint directories differ)
python3 scripts/paperA_tables.py --bench biodiv     # -> results/paperA_v2/tables_biodiv.json
python3 scripts/paperA_tables.py --bench biored     # -> results/paperA_v2/tables_biored.json
python3 scripts/biored_ep_eval.py --src data/raw/bioredirect   # -> results/paperA_v2/biored_ep.json

# 4. Zero-shot LLM rows, then refresh only the LLM part of the tables
python3 scripts/llm_baseline.py --bench biodiv --model qwen3:32b
python3 scripts/llm_baseline.py --bench biored --model qwen3:32b --n 3000
python3 scripts/paperA_tables.py --bench biodiv --llm-only
python3 scripts/paperA_tables.py --bench biored --llm-only

# 5. Figure
python3 paperA/fig/make_threshold_figure.py
```

## Which output feeds which table

| Paper | Source |
|---|---|
| Controlled comparison on 437 rows (main table), where the gain lives (taxa counts, blocks), operating curves, per-seed and recall-side appendix sections | `tables_biodiv.json`: keys `sentence`, `pair`, `triple`, `mcnemar`, `curve` |
| Degrading segment A (ablation) | `tables_biodiv.json`: `ablation` |
| Candidate rules (body paragraph and appendix table) | `tables_biodiv.json`: `rules7`, `rules8`, `reject50` |
| BioRED table: AUPRC, F1, 2 vs. >=3 concepts | `tables_biored.json` |
| BioRED entity-pair F1 | `biored_ep.json` |
| Zero-shot Qwen3 rows (biodiversity and BioRED), and the scaling table by model size (appendix) | `tables_*.json`: `llm` |
| Threshold figure | `make_threshold_figure.py` (reads `S_biodiv_*.npy`, `tables_biodiv.json` and the benchmark) |
| Teacher prompts (appendix) | `teacher_label_triples.py`, `llm_baseline.py` |
| Larger-encoder negative result (encoder paragraph, negative-results appendix) | `tables_biored.json`: `sentence_large`, `pair_large`, `mcnemar_pair_vs_sentence_large`; `biored_ep.json`: same arm names (`paperA_tables.py --bench biored --arms large`) |

The `S_<bench>_<arm>.npy` arrays are the three-checkpoint mean scores, row-aligned with the clean
benchmark (biodiversity) or with `data/benchmarks/biored_bc8/test.csv` (BioRED). The deployed
model's direction table uses a checkpoint and expert direction labels that will be released with
the paper.
