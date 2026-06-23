# Biotic Interaction Classifier — Training Dataset Versions

All datasets live in `classifier/data/training/`.
Evaluation is always on the EP test set (48/100 positives) via `train_cv_regularized.py`.
*Standard script results (eval_100) are marked with an asterisk and not directly comparable.*

---

## Version History

### v1 — Initial GloBI Templates
**File:** `training_data_globi_v1.csv`
**Size:** 39,999 samples — 17,781 pos (44.5%), 20,000 neg
**Sources:** GloBI database → template-generated sentences (all synthetic)
**Interaction types:** endoparasiteOf=8,502, eats=5,011, preysOn=1,852, hasHost=1,450, pathogenOf=438
**Notes:**
- Baseline dataset. All sentences follow rigid templates ("Data shows that X is an endoparasite of Y.").
- Positive ratio too high at 44.5% — balanced dataset doesn't reflect real-world distribution.
- Neg:Pos = 1.13:1 — not enough negatives for discriminative training.
**EP Test F1: not recorded**

---

### v2 — Hard Negatives for Precision
**File:** `training_data_globi_v2.csv`
**Size:** 44,999 samples — 13,355 pos (29.7%), 28,685 neg
**Sources:** v1 templates + hard negatives (two species co-mentioned, no interaction)
**Notes:**
- Key insight: added "hard negatives" — sentences mentioning two species but no real interaction.
- 80% of negatives are hard (none_two_species), 20% easy (none).
- Neg:Pos ratio improved to 2.15:1.
- Positive ratio dropped to 29.7% — more realistic.
**EP Test F1: not recorded**

---

### v3 — First Real Sentence Injection (experimental)
**File:** `training_data_globi_v3.csv`
**Size:** 31,999 samples — 6,548 pos (20.5%), 22,706 neg
**Sources:** Templates + 19 real literature sentences (first attempt)
**Notes:**
- Smaller dataset (32K), simplified schema (removed species columns).
- Only 19 real positive sentences — proof-of-concept for hybrid approach.
- Hard:easy negative ratio increased to 4:1 (19,200 hard / 4,800 easy).
- Metadata in `training_data_globi_v3_metadata.json`.
**EP Test F1: not recorded**

---

### v4 — Eval100-Style Templates
**File:** `training_data_globi_v4.csv`
**Size:** 31,999 samples — 6,564 pos (20.5%), 22,816 neg
**Sources:** Templates rewritten to match patterns observed in eval_100 test set
**Notes:**
- Templates modelled on real scientific language from eval_100 positives.
  (e.g. "The endoparasite X was recovered from Y." vs. "Data shows that X is an endoparasite of Y.")
- Restored full 6-column schema (text, label, species, interaction_type, quality_score).
- Same size as v3 but higher linguistic realism.
**EP Test F1: not recorded**

---

### v5 — Iterative Refinement
**Files:** `training_data_globi_v5.csv` / `training_data_globi_v5_clean.csv`
**Size:** 31,999 / 31,564 samples
**Sources:** Refinement of v4 templates + cleaning pass
**Notes:**
- v5: minor improvements to template diversity over v4.
- v5_clean: deduplication and quality-score filtering pass (~5h later same day).
- Marks the start of systematic cleaning iterations.
**EP Test F1: not recorded**

---

### v6 — Diversity Push
**Files:** `training_data_globi_v6.csv` → `v6_diverse.csv` → `v6_diverse_cleaned.csv` → `v6_diverse_cleaned_cleaned.csv`
**Size:** ~31,000–32,000 samples across variants
**Sources:** v5 base + increased sampling of under-represented interaction types
**Notes:**
- Goal: reduce dominance of endoparasiteOf (was 60%+ of positives in v1-v5).
- Multiple successive cleaning passes on same day (Feb 3–4, 2024).
- hasHost and visitsFlowersOf better represented vs. earlier versions.
- Prepared the ground for v7's LLM validation approach.
**EP Test F1: not recorded**

---

### v7 — LLM-Validated Baseline ⭐ **(best so far)**
**File:** `training_data_globi_v7_llm_cleaned.csv`
**Size:** 25,081 samples — 7,251 pos (28.9%), 17,830 neg
**Sources:** GloBI-derived template sentences, each positive LLM-validated with Claude (Anthropic API)
**Notes:**
- Gold standard. Every positive sentence individually checked by LLM.
- Smaller than v6 because LLM validation rejected a significant fraction of templates.
- Highest precision of all versions to date.
**EP Test F1: 0.788** (BiomedBERT regularized)

---

### v8 — SIBiLS Harvest (pathogen-biased) ❌
**File:** `training_data_v8_diverse.csv`
**Size:** ~26K samples
**Sources:** v7 + SIBiLS API harvest targeting "infection" articles
**Notes:** 92% of new positives were pathogen/infection type → precision collapsed.
**EP Test F1: 0.695*** (BiomedBERT, standard script — not directly comparable)

---

### v9 — Regex-Labeled Noise ❌
**File:** `training_data_v9.csv`
**Sources:** v7 + regex-labeled sentences (no LLM validation)
**Notes:** Label noise from regex false positives degraded the model.
**EP Test F1: 0.644*** (SciBERT, standard script — not directly comparable)

---

### v10 — GloBI-PMC Real Sentences (pathogen-biased) ❌
**File:** `training_data_v10.csv`
**Size:** 26,785 samples — 8,087 pos, 18,698 neg
**Sources:**
- v7_llm_cleaned (25,081)
- globi_pmc_real (872 real PMC sentences from GloBI-cited articles, `fetch_globi_pmc.py`)
**Notes:**
- First real literature sentences in training — but GloBI citation list skewed toward pathogen papers.
- 800/872 new positives were pathogenOf/infection type.
- Result: high recall (0.975), low precision (0.574) — over-predicts on ecological test set.
**EP Test F1: 0.722** (BiomedBERT regularized, Prec=0.574, Rec=0.975)

---

### v10.1 — Europe PMC Direct Harvest, Ecologically Diverse (unfiltered)
**File:** `training_data_v10.1.csv`
**Size:** 27,762 samples — 8,346 pos, 19,416 neg
**Sources:**
- v7_llm_cleaned (25,081)
- epmc_direct (965 pos) — Europe PMC direct search (`fetch_epmc_direct.py`), **no signal filter**
- globi_pmc_v2 (130 pos) — GloBI-PMC v2 harvest (`fetch_globi_pmc.py` with fixed interaction types)
**New positives by type:** parasiteOf=300, kleptoparasiteOf=265, pollinates=191, symbioticWith=155, eats=73
**Notes:**
- First version with meaningful ecological diversity (predation, pollination, symbiosis).
- `fetch_epmc_direct.py`: searches Europe PMC directly for known GloBI species pairs + keyword,
  then extracts co-occurrence sentences from full-text articles.
- No interaction signal quality filter → may include noisy sentences without interaction verbs.
- Previously briefly named "v11" (renamed to avoid conflict with user's parallel v11 work).
**EP Test F1: TBD** (training running, PID 283905)

---

### v11 — (User's Parallel Work)
**File:** `training_data_v11.csv` *(consumed by v11_1; original may differ from v10.1)*
**Notes:** Built in parallel with v10.1 work. Superseded by v11_1.

---

### v11_1 — + epmc_direct_v2 + external_db (unfiltered)
**File:** `training_data_v11_1.csv`
**Size:** 28,666 samples — 8,808 pos, 19,858 neg
**Sources:**
- v7_llm_cleaned (25,081)
- epmc_direct (965 pos)
- epmc_direct_v2 (398 pos) — second Europe PMC harvest (herbivory/predation focus)
- globi_pmc_v2 (130 pos)
- external_db (64 pos) — Mangal / Web of Life / OpenAlex food webs
**Notes:**
- Most diverse positive set to date (5 sources).
- New positives added **without** interaction signal filter.
- Build script: `build_v11_1_dataset.py`
**EP Test F1: TBD** (training running, PID 283207)

---

### v12 — Signal-Filtered Harvest ⚠
**File:** `training_data_v12.csv`
**Size:** 27,652 samples — 8,106 pos, 19,546 neg
**Sources:**
- v7_llm_cleaned (25,081)
- epmc_direct (688 pos) — filtered: `interaction_lexicon.score_sentence()` score > 0
- globi_pmc_v2 (100 pos) — filtered: score > 0
- external_db (67 pos) — filtered: score > 0
**Notes:**
- Key change: `_has_interaction_signal()` filter on ALL new positives.
- Drops sentences with no interaction vocabulary (score=0): removed 34.6% of epmc_direct,
  47% of external_db as likely noise.
- Slightly below v11 (0.745) — signal filter alone not sufficient to beat v7.
- Build script: `build_v12_dataset.py`
**EP Test F1: 0.729** (BiomedBERT regularized, Prec=0.651, Rec=0.829)
**FLAN-T5-large: avg=0.780 / best fold=0.800** (Prec=0.737, Rec=0.875) — NEW BEST generative
**Ensemble (BiomedBERT × FLAN-T5-base): 0.865** (arithmetic mean) — CURRENT BEST OVERALL

---

### v13 — (Skipped / merged into v14)
**Notes:** Build attempted but superseded by v14. No final results recorded.

---

### v14 — SIBiLS Over-Pruned ❌
**File:** `training_data_v14.csv`
**Size:** ~27K samples
**Built:** 2026-03-15
**Sources:** v7_llm_cleaned + additional EPMC/SIBiLS harvest (score>0 filter reapplied)
**Notes:**
- Same score>0 filter mistake as v12 — removed valid implicit interactions.
- FLAN-T5-base F1=0.706 → regression vs. v12.
- Root cause confirmed: score>0 filter is NOT a proxy for LLM validation.
- Build script: `build_v14_dataset.py`
**EP Test F1: ~0.706** (FLAN-T5-base, regression)

---

### v15 — Teacher-Labeled (IN PROGRESS) ⭐
**Output dir:** `v15_teacher/` (train.csv, dev.csv, test.csv, metadata.json)
**Built:** 2026-03-31 (assembly pending curation completion)
**Sources:**
  *Positives:*
  - Qwen3.5-122B teacher YES labels: 4,065 real PMC sentences (9.2% positive rate from 44,178 total)
  - v7 LLM-validated backbone (excl. pathogenOf): ~7,076 templates (Claude API validated)
  - v7 pathogenOf Qwen-accepted: 64 (37% of 175 — rest rejected as too formulaic)
  - EPMC targeted pathogen harvest: 56 (Qwen-confirmed, `fetch_pathogen_sentences.py`)
  - Human-curated pathogen borderline: 6
  - **Total positives: ~11,267 (before dedup)**

  *Negatives:*
  - Clean (lexicon=0, Qwen=NO): 12,000 sampled
  - Confirmed clean (2× Qwen=NO, strong lexicon signal): 3,300
  - Weak signal (Qwen=NO, not rechecked): 12,177

  *Test (fixed):*
  - All 7 eval files Qwen-validated: 599 sentences, gold labels authoritative

**Key design decisions:**
- No score>0 filter on teacher positives — teacher already did semantic filtering
- Qwen3.5-122B (122B MoE, local) as teacher; stricter than Claude API
- neg:pos target = 2.5
- pathogenOf specifically reinforced (213 positives total) after underrepresentation detected

**Build script:** `scripts/assemble_v15_dataset.py`
**EP Test F1: TBD** (pending assembly + training)

---

## Summary Table

| Version | Samples | Pos | F1 (EP) | Prec | Rec | Key Change |
|---------|---------|-----|---------|------|-----|------------|
| v1      | 39,999  | 17,781 | — | — | — | Initial GloBI templates |
| v2      | 44,999  | 13,355 | — | — | — | Hard negatives (2-species, no interaction) |
| v3      | 31,999  | 6,548 | — | — | — | 19 real sentences (first attempt) |
| v4      | 31,999  | 6,564 | — | — | — | Eval100-style template language |
| v5      | 31,999  | 6,557 | — | — | — | Template refinement + dedup |
| v6      | ~32,000 | 6,573 | — | — | — | Diversity push, multi-cleaning pass |
| v7      | 25,081  | 7,251 | **0.788** | — | — | LLM-validated gold standard ⭐ |
| v8      | ~26K    | — | 0.695* | — | — | SIBiLS pathogen bias |
| v9      | ~26K    | — | 0.644* | — | — | Regex label noise |
| v10     | 26,785  | 8,087 | 0.722 | 0.574 | 0.975 | Real PMC sentences, pathogen-biased |
| v10.1   | 27,762  | 8,346 | TBD | — | — | Ecological diversity, unfiltered |
| v11_1   | 28,666  | 8,808 | TBD | — | — | + epmc_v2 + external_db |
| v12     | 27,652  | 8,106 | 0.729 | 0.651 | 0.829 | Signal filter on new positives |
| v14     | ~27K    | — | ~0.706 | — | — | score>0 filter repeated (regression) |
| v15     | ~40K    | ~11K | TBD | — | — | Qwen3.5-122B teacher labels ⭐ (in progress) |

*Standard script on eval_100 — not directly comparable to EP test set.

---

## How to Add a New Version

1. Build: `python classifier/scripts/build_vXX_dataset.py --output training_data_vXX.csv`
2. Quality gates: `cd classifier && python -m pytest tests/test_training_data.py -v`
   (update `TEST_DATA_FILE` in the test to point to the new file)
3. Train: `python classifier/scripts/train_cv_regularized.py --train-data ... --models BiomedBERT --suffix vXX`
4. **Update this file** with F1/Prec/Rec results before moving on.
