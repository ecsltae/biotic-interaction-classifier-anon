# Pair-Conditioned Verification of Literature-Mined Biotic Interactions

Anonymous code release for the submission. It holds the manuscript, the code that trains the
compared models and computes every table, and the computed outputs (`results/paperA_v2/`), so
the tables can be checked without retraining.

## Layout

| Path | What it is |
|---|---|
| `paperA/` | manuscript (`paperA.tex`, `references.bib`, ACL style files, compiled `paperA.pdf`) and the figure scripts (`paperA/fig/`) |
| `experiments/multitask/train_student.py` | trains one verifier; `--input-format sentence\|pair\|triple` is the only difference between the compared arms |
| `scripts/xenc_format.py` | builds the encoder input of each format (passage only; pair + passage; pair + relation + passage) |
| `scripts/eval_unified.py` | scores a checkpoint on a candidate table (used by the two scripts below) |
| `scripts/paperA_tables.py` | every number of the results tables, from trained checkpoints |
| `scripts/biored_ep_eval.py` | BioRED entity-pair F1 (the standard document-level metric), from saved per-checkpoint scores |
| `scripts/biored_ep_scores_cpu.py` | computes and saves those per-checkpoint BioRED scores (CPU) |
| `scripts/paperA_pair_only.py` | the pair-only controls: the pair without its passage, trained and zero-shot (appendix "Pair-only controls") |
| `scripts/paperA_pair_only_long.py` | the pair-only model retrained for up to ten epochs (Section 4.5, Appendix Q), from `scripts/run_pair_only_long_2026-10-12.sh` |
| `scripts/llm_baseline.py` | zero-shot local LLM rows (Ollama), same three formulations as the trained arms |
| `scripts/teacher_label_triples.py` | the teacher's labelling prompt (also read by `llm_baseline.py` for the triple question) |
| `scripts/soft_relabel.py` | asks the teacher again for P(yes) on every training row (soft-label appendix) |
| `scripts/convert_biored_verify.py` | builds the sentence-scope BioRED candidate tables (BC8 protocol) |
| `scripts/convert_biored_doc.py` | builds the whole-abstract BioRED candidate tables |
| `scripts/run_biored_bc8.sh`, `scripts/run_biored_bc8_doc.sh` | train the BioRED arms (3 seeds each) |
| `scripts/candidate_rules.py` | the deterministic candidate rules (Appendix L; mentioned in Section 5) |
| `scripts/scan_contamination.py` | near-duplicate scan of the benchmark against the training passages (defines the 437 clean rows) |
| `scripts/cascade_tables.py` | escalation of the recommended model's least confident candidates to a zero-shot LLM (Appendix N) |
| `scripts/fresh_sample.py` | the 100 fresh candidates (Section 4.7) |
| `scripts/audit_paper_numbers.py` | checks every number in `paperA.tex` against the result files |
| `scripts/check_references.py` | checks `references.bib` against Semantic Scholar |
| `scripts/paperA_fig2_equal_share.py`, `scripts/paperA_table_shares.py` | F1 at equal share of candidates accepted (Figure 2, Section 4.1) and the share columns of Tables 3 and 4 |
| `paperA/fig/user_2026-10-07/` | `rows.csv` (gold label and three-checkpoint scores of the 437 rows) and `figures/paper_figures.py`, which draws Figures 2 and 3 |
| `src/eval/core.py` | the single loader of the 437 clean benchmark rows, used by every script |
| `results/paperA_v2/` | computed outputs: `tables_biodiv.json`, `tables_biored.json`, `biored_ep.json`, `pair_only.json`, the per-row ensemble scores `S_*.npy`, the per-checkpoint BioRED scores behind the entity-pair F1 (`biored_ep_scores/`) and the pair-only models' scores (`pair_only_scores/`) |

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
  They are not part of this anonymous package. The paper uses the revised labels (437 clean rows,
  251 positive, after the blind re-annotation reported in the Limitations): `src/eval/core.py`
  loads them with `clean_benchmark()`, and the earlier labels with
  `clean_benchmark("pre_review_2026-10-06")`; each version is hash-checked.

## Environment

Python 3.10; a GPU for training.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The LLM rows need [Ollama](https://ollama.com) on `localhost:11434` with the Qwen3 models pulled
(`ollama pull qwen3:32b`, and the smaller sizes for the scaling table). Use the original Qwen3
builds: `qwen3:0.6b`, `qwen3:1.7b`, `qwen3:4b-q4_K_M`, `qwen3:8b`, `qwen3:14b`,
`qwen3:30b-a3b-q4_K_M`, `qwen3:32b`. The default `qwen3:4b` and `qwen3:30b` tags now point to later
re-releases that ignore `think: false`; `llm_baseline.py` stops if a model does not answer YES/NO.
`qwen3.5:122b` (81 GB) does not fit on one 80 GB GPU, and Ollama's own layer split aborted in
cuBLAS on ours; `--num-gpu 44 --num-ctx 4096` (44 of 49 layers on the GPU, the rest on the CPU)
runs it at about 1.4 s per candidate.

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

# 3. Tables. paperA_tables.py reads the biodiversity arms from models/sentence_baseline/xenc_s{1,2,3},
#    models/pair_baseline/xenc_s{1,2,3} and models/student_v3/xenc_s{1,2,3}: set ARMS_BIODIV
#    (and ARMS_BIORED) at its top to models/biodiv/{sentence,pair,triple}_s{} to use step 2's checkpoints
python3 scripts/paperA_tables.py --bench biodiv     # -> results/paperA_v2/tables_biodiv.json
python3 scripts/paperA_tables.py --bench biored     # -> results/paperA_v2/tables_biored.json
python3 scripts/biored_ep_scores_cpu.py --arms sentence pair pair_mark doc_pair   # -> results/paperA_v2/biored_ep_scores/
python3 scripts/biored_ep_eval.py --src data/raw/bioredirect --arms sentence pair pair_mark doc_pair pair_only   # -> results/paperA_v2/biored_ep.json
python3 scripts/biored_ep_eval.py --src data/raw/bioredirect --fp32 --arms sentence_large pair_large sentence_linkbert pair_linkbert sentence_linkbertL pair_linkbertL   # encoder comparison; scores missing vectors on the GPU in float32

# 4. Zero-shot LLM rows, then refresh only the LLM part of the tables
python3 scripts/llm_baseline.py --bench biodiv --model qwen3:32b
python3 scripts/llm_baseline.py --bench biored --model qwen3:32b --n 3000
python3 scripts/paperA_tables.py --bench biodiv --llm-only
python3 scripts/paperA_tables.py --bench biored --llm-only

# 5. Soft-label students (appendix): teacher P(yes) on the training rows, three arms x three seeds
python3 scripts/soft_relabel.py          # -> data/training/distill/v3_combined_train_soft.csv
for fmt in sentence pair triple; do for s in 1 2 3; do
  python3 experiments/multitask/train_student.py --data data/training/distill/v3_combined_train_soft.csv \
      --soft-col soft_label --input-format $fmt --seed $s --out models/soft_students/${fmt}_s$s
done; done
python3 scripts/paperA_tables.py --bench biodiv --arms soft    # adds *_soft keys

# 6. Figures 1 to 4 (paper_figures.py writes to the current directory, so it runs from paperA/fig/)
python3 paperA/fig/make_concept_figure.py      # Figure 1
python3 paperA/fig/fig3_stats.py               # Figure 3 values and intervals -> paperA/fig/fig3_gain_by_taxa.json
(cd paperA/fig && python3 user_2026-10-07/figures/paper_figures.py fig2 --scores user_2026-10-07/rows.csv)   # Figure 2
(cd paperA/fig && python3 user_2026-10-07/figures/paper_figures.py fig3 --stats fig3_gain_by_taxa.json)      # Figure 3
python3 paperA/fig/make_scaling_figure.py      # Figure 4
```

## Which output feeds which table

| Paper | Source |
|---|---|
| Controlled comparison on 437 rows (main table), where the gain lives (taxa counts, blocks), operating curves, per-seed and recall-side appendix sections | `tables_biodiv.json`: keys `sentence`, `pair`, `triple`, `mcnemar`, `curve` |
| Degrading segment A (ablation) | `tables_biodiv.json`: `ablation` |
| Candidate rules (Appendix L, table and text) and recall on rejected candidates (Appendix I) | `tables_biodiv.json`: `rules7`, `rules8`, `reject50` |
| BioRED table: AUPRC, F1, 2 vs. >=3 concepts | `tables_biored.json` |
| BioRED entity-pair F1 (Table 1, Section 4.2, BioRED appendix table) | `biored_ep.json`, from the scores in `biored_ep_scores/`. A candidate's concept key joins its IDs with "\|", and variant IDs contain "\|" themselves, so the scorer looks each key's IDs up in the PubTator annotations |
| Pair-only controls (Table 1 row, Section 4.5, appendix "Pair-only controls") | `pair_only.json` (`paperA_pair_only.py report`, from the scores in `pair_only_scores/` and the no-passage LLM answers); the ten-epoch retraining: `pair_only_long.json` |
| Zero-shot Qwen3 rows (biodiversity and BioRED), and the scaling table by model size (appendix) | `tables_*.json`: `llm`. These files also keep exploratory zero-shot rows that the paper does not report (`medgemma-27b`, a re-run of `qwen3-32b` under another Ollama version tagged `v035`, and `qwen3.8-27b-v035`, with students distilled from it, keys `*_qwen38`); the paper's rows are the untagged Qwen3 and Qwen3.5 ones |
| Figure 1 (one passage, several possible pairs) | `paperA/fig/make_concept_figure.py` (scores from `S_biodiv_*.npy`, thresholds from `tables_biodiv.json`) |
| Figure 2 (precision against recall; F1 against share of candidates accepted) | `paperA/fig/user_2026-10-07/figures/paper_figures.py fig2 --scores user_2026-10-07/rows.csv` (rows.csv: gold labels and `S_biodiv_{sentence,pair,triple}.npy`; markers at the Table 1 operating points); share ranges in the caption and text: `fig2_equal_share.json` (`scripts/paperA_fig2_equal_share.py`) |
| Figure 3 (AUPRC by number of entities named) | `paperA/fig/fig3_stats.py` -> `paperA/fig/fig3_gain_by_taxa.json` (values, shares of positives, bootstrap intervals; checked against `fig_where.json` from `make_where_figure.py`), drawn by `paper_figures.py fig3` as in step 6 |
| Figure 4 (zero-shot scaling) | `paperA/fig/make_scaling_figure.py` (`tables_biodiv.json` and `tables_biored.json`: `llm`) |
| Earlier figures (not in the current paper) | `make_threshold_figure.py`, `make_curves_figure.py` |
| Share accepted (Table 3) and share of positives (Table 4) | `table_shares.json` (`scripts/paperA_table_shares.py`) |
| Teacher prompts (appendix) | `teacher_label_triples.py`, `llm_baseline.py` |
| Soft-label distillation (appendix table) | `tables_biodiv.json`: `sentence_soft`, `pair_soft`, `triple_soft`, `mcnemar_soft` (between soft arms), `mcnemar_vs_base_soft` (each soft arm against the verdict-trained arm of the same format) |
| Encoders appendix (BioLinkBERT-base; encoder paragraph) | `tables_biored.json` and `tables_biodiv.json`: `*_linkbert`, `mcnemar_vs_base_linkbert` (trained with `train_student.py --encoder michiyasunaga/BioLinkBERT-base`, scored with `paperA_tables.py --bench biored\|biodiv --arms linkbert`); BioLinkBERT-large, all arms: `*_linkbertL`, `mcnemar_linkbertL`, `mcnemar_vs_base_linkbertL` (trained with `--encoder michiyasunaga/BioLinkBERT-large --micro-batch 16`, scored with `--arms linkbertL`); entity-pair F1: `biored_ep.json`, keys `sentence_linkbert`, `pair_linkbert`, `sentence_linkbertL`, `pair_linkbertL` |
| Escalation to a zero-shot LLM (Appendix N, table) | `cascade.json` (`cascade_tables.py`; recommended model scores and the zero-shot LLMs' pair answers) |
| Fresh candidates (Section 4.7): AUPRC, AUROC, bootstrap interval of the pair-conditioned lead | `fresh_sample.json` |
| Retrieval pool statistics (Setting) and CPU throughput (recommended configuration) | `pool_scan.json`, `cpu_bench_handoff.json` |
| Numbers computed from cached scores that no table script stores: argument strata (Figure 3 and Table 4), false-positive counts, pair model with rules, per-seed F1 at 0.5, recall-side counts, base-sample model, rules against no rules for the recommended model | `derived_biodiv.json` (`paperA_derived_numbers.py`) |
| Larger-encoder negative result (encoder paragraph, negative-results appendix) | `tables_biored.json`: `sentence_large`, `pair_large`, `mcnemar_pair_vs_sentence_large` (`paperA_tables.py --bench biored --arms large`); entity-pair F1: `biored_ep.json`, keys `sentence_large`, `pair_large` |

The `S_<bench>_<arm>.npy` arrays are the three-checkpoint mean scores, row-aligned with the clean
benchmark (biodiversity) or with `data/benchmarks/biored_bc8/test.csv` (BioRED). The recommended
model's direction table uses a checkpoint and expert direction labels that will be released with
the paper.

## Licences

- Code (`src/`, `scripts/`, `experiments/`, the Python scripts in `paperA/fig/`): MIT, see `LICENSE`.
- Our labels and computed outputs (`results/`, the JSON and CSV files in `paperA/fig/`): CC BY 4.0.
- PubMed passages quoted in the manuscript, the figures and `paperA/fig/fig_concept.json`: courtesy
  of the U.S. National Library of Medicine; their copyright remains with the publishers or authors.
  A copy may differ from the current PubMed record.
- Candidate triples come from BiotXplorer (SIBiLS, CC BY 4.0).
- `paperA/acl.sty` and `paperA/acl_natbib.bst`: the ACL style files, under their own terms.

The benchmark, the 100 fresh candidates and the training corpus will be released with the paper,
with the PMID of every passage (Appendix O of the manuscript).
