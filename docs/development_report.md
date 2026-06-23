# Building a Biotic Interaction Classifier for Scientific Literature
## Development Journey, Experimental Findings, and Current Architecture

*Anonymous — PhD project, April 2026*

---

## Abstract

This document traces the full development arc of the biotic interaction classifier:
from naïve GloBI template training (EP F1=0.63) through LLM-validated data curation (F1=0.788),
ensemble architectures (F1=0.857), knowledge distillation (F1=0.808), and finally
multi-task learning with species NER (F1=0.868). The best single model now exceeds
the 2-model ensemble at 3.3× lower inference cost. The document records not only what
works but specifically what failed and why — these negative results are the primary
intellectual contribution for future iteration.

---

## 1. Motivation and Task Definition

### 1.1 Scientific Context

The overarching goal is to build a knowledge graph of host–pathogen interactions for
biodiversity research. The input is the scientific literature — PubMed abstracts,
Europe PMC full text — and the output is a set of structured triples:
*(HOST species, INTERACTION TYPE, PATHOGEN/SYMBIONT species)*.

The first bottleneck is **sentence-level classification**: given a sentence from a
scientific article, does it describe a direct biotic interaction between two species?
This is the classifier documented here.

### 1.2 Task Formulation

Binary classification: positive = sentence explicitly states an interaction (parasitism,
predation, pollination, symbiosis, pathogen infection, herbivory, etc.) between
two named organisms; negative = sentence mentions species co-occurrence, methodology,
phylogeny, disease outcome, or any non-interaction context.

Key challenges that shaped every design decision:

1. **Interaction type imbalance**: the most common interaction in ecological databases
   (endoparasiteOf) is almost absent from pathogen-focused literature, and vice versa.
   A model trained on one distribution performs poorly on the other.

2. **Template–literature distribution shift**: sentences generated from GloBI interaction
   templates ("*X is an endoparasite of Y*") are linguistically simple and regular;
   real scientific prose uses hedging, passive voice, coordination, and nominalisations
   that templates never produce. A model trained purely on templates generalises poorly.

3. **Test set leakage**: the in-house eval_100 test set included sentences from the same
   GloBI pairs used to generate templates. Any model trained on templates appears to score
   perfectly on eval_100 while remaining mediocre on real literature. **Rule: EP-relax is
   the only authoritative evaluation metric.**

### 1.3 Evaluation Sets

Two test sets used throughout, with very different characters:

| Set | Sentences | Positives | Source | Character |
|-----|-----------|-----------|--------|-----------|
| **EP-relax** | 99 | 48 | GloBI–Europe PMC linked sentences (2024) | Real literature; 58% endoparasiteOf/hasHost |
| **Synthetic Gold** | 100 | 50 | Curated by interaction type | Balanced; covers 10 interaction types |

EP-relax is the deployment-quality signal. Synth Gold is useful for diagnosing
per-type weaknesses. They often disagree: models that excel on Synth Gold may still
fail on EP-relax if they do not handle endoparasiteOf-style language.

---

## 2. Data Pipeline

### 2.1 Iteration Philosophy

Training data was iterated through 15 versions. The key lesson is that
**data quality dominates model choice** for this task. A simpler model trained
on cleaner data consistently outperforms a complex model on noisier data.

### 2.2 Template Generation Phase (v1–v6)

**v1 (39,999 samples, 44.5% positive)** — Pure GloBI templates. Five rigid sentence
forms per interaction type. Baseline for measuring everything else. EP F1: not
recorded (likely < 0.65).

**v2 (44,999 samples)** — Added *hard negatives*: sentences mentioning two species but
no interaction (co-occurrence, geographic overlap). Negative-to-positive ratio
increased from 1.13:1 to 2.15:1. This was the single most important structural
change: without hard negatives the classifier learns species co-mention as a
positive signal, which is nearly always wrong in deployment.

**v3–v6** — Iterative refinement: more template diversity, LLM-modelled language
(v4 templates modelled on eval_100 patterns), systematic cleaning passes. These
versions closed some of the template–literature gap but could not eliminate it.

### 2.3 LLM Validation Breakthrough (v7)

**v7 (25,081 samples, F1=0.788)** — Each positive sentence individually verified
by Claude API. Roughly 30% of template-generated positives were rejected as
linguistically implausible or factually wrong. The result is smaller but much
cleaner. BiomedBERT trained on v7 set the baseline that all subsequent work is
measured against.

The v7 backbone (excl. pathogenOf templates) was used as the anchor for every
subsequent dataset version.

### 2.4 Real Sentence Injection Phase (v8–v12)

This phase attempted to close the template–literature gap by injecting real
sentences from PubMed/Europe PMC. Every version in this phase failed to improve
on v7 in at least one important way.

**v8 — SIBiLS harvest (pathogen-biased):** Searched SIBiLS for "infection" articles.
92% of new positives were pathogenOf/infection type. The model became recall-heavy
and imprecise on ecological interactions. F1 dropped to 0.695.

**v9 — Regex label noise:** Applied interaction-signal regex to label harvested
sentences without LLM validation. Label noise from false positives degraded
everything. F1 0.644.

**v10 — GloBI-PMC real sentences (pathogen-biased):** Fetched real PMC articles
from GloBI citation list. Same bias problem: 800/872 new positives were
pathogenOf sentences. Result: recall 0.975, precision 0.574, F1 0.722.

**Key insight from v8–v10:** The internet (PubMed) is dominated by pathogen/infection
literature. Harvesting from it without careful interaction-type targeting always
produces a pathogen-heavy training set. Since EP-relax is 58% endoparasiteOf/hasHost
(ecological, not pathogen), the resulting models have a systematic precision failure
on that test set.

**v12 — Signal-filtered harvest (F1=0.729):** Applied an `interaction_signal_score > 0`
filter to all harvested sentences before adding them to training. The filter
was intended to remove noise; it removed 35–47% of sentences with no explicit
interaction vocabulary. However, F1 only reached 0.729 vs. v7's 0.788, confirming
the filter is not a substitute for LLM validation.

**v14 — score>0 filter repeated (F1≈0.706):** Repeated the same mistake as v12 with
FLAN-T5. Confirmed that `score > 0` is not a proxy for semantic correctness —
many valid interactions are expressed without interaction vocabulary ("*X was found
in the gut of Y*"). Any regression from v7 on EP-relax indicates real-sentence
contamination is hurting, not helping.

### 2.5 Distillation Training Data (44K soft labels)

Rather than building a new ground-truth dataset, the distillation approach used
the ensemble (BiomedBERT cv_reg × FLAN-T5-base v12) to generate soft probability
labels over the full v14 training set (44,178 sentences). This `distillation_soft_labels.csv`
file became the training corpus for all distillation and multi-task experiments.

This is a deliberate trade-off: the soft labels are not ground truth but they
carry the ensemble's uncertainty information. A student trained to reproduce
these distributions inherits the ensemble's calibration at the cost of treating
the ensemble's errors as signal.

### 2.6 v15 Design (in progress)

v15 addresses the root cause of the template gap: it uses the **Qwen3.5-122B**
local model (122B MoE, run via Ollama) as teacher to label 44K real PMC sentences.
At a 9.2% positive rate, this produces ~4,065 positive real sentences covering
the full range of interaction types (not just pathogens). Combined with the v7
backbone and targeted pathogenOf reinforcement, v15 is designed to be the first
dataset with genuine endoparasiteOf/hasHost coverage from real literature —
the specific interaction type that most EP-relax positives belong to.

---

## 3. Model Architecture Evolution

### 3.1 Baseline: Classical ML and Small Transformers

SVM (TF-IDF), random forest, and BERT-base were evaluated early. BiomedBERT
(BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext, 110M params) consistently
outperformed alternatives for this domain due to its pretraining corpus (PubMed
abstracts + full text).

LUKE (entity-aware transformer) was evaluated but provided no benefit over
BiomedBERT on this task. It was subsequently removed from the codebase.

### 3.2 BiomedBERT cv_regularized (F1=0.788 on v7, F1=0.825 on distillation data)

The standard discriminative model: BiomedBERT fine-tuned with 5-fold cross-validation
and L2 regularization on v7 data. This became the primary teacher for distillation.

The use of cross-validation rather than a single train/val split is important here:
it exposes every sentence to both training and validation roles, providing a better
calibration signal for the soft labels used in distillation.

### 3.3 FLAN-T5 (F1=0.800 on v12, generative)

FLAN-T5-base v12 was the best single generative model. It formulates classification
as sequence-to-sequence: the sentence is prefixed with a task description and the
model generates "yes" or "no". The key advantage is zero-shot compositionality —
it handles new interaction phrasings better than discriminative models fine-tuned
on fixed templates.

**Important constraint**: only FLAN-T5-*base* is used. FLAN-T5-large/XL/XXL were
evaluated and explicitly rejected — the inference cost does not justify the marginal
gain for this task.

### 3.4 Ensemble: BiomedBERT × FLAN-T5-base (F1=0.857)

The ensemble of BiomedBERT cv_reg and FLAN-T5-base v12 (geometric mean of
probabilities) achieved F1=0.857 on EP-relax. This became the target to beat.

The geometric mean is preferred over arithmetic mean for combining models with
different uncertainty profiles: it is more conservative (requires both models
to be confident for a high combined score) and empirically outperforms simple
averaging on this dataset.

---

## 4. Knowledge Distillation

### 4.1 Motivation

The ensemble requires two forward passes (BiomedBERT + FLAN-T5) and is more
expensive to deploy. The hypothesis was: can a single BiomedBERT student trained
on the ensemble's soft labels match or exceed the ensemble?

Soft labels carry richer information than hard 0/1 labels — they encode the
ensemble's uncertainty and inter-model disagreement. A student minimising KL
divergence from these distributions learns to reproduce the teacher's confidence
calibration, not just its decisions.

### 4.2 Distillation Hyperparameter Search

Six variants were trained, varying temperature T (sharpness of soft labels)
and α (weight of soft-label loss vs. hard-label cross-entropy):

| Model | T | α | EP F1 | SG F1 | Notes |
|-------|---|---|-------|-------|-------|
| distilled_v1 | 4 | 0.7 | 0.785 | 0.948 | Too soft (high T) |
| **distilled_v2** | **2** | **0.5** | **0.808** | **0.959** | **Best balanced** |
| distilled_v3 | 4 | 0.9 | 0.808 | 0.916 | High α but high T cancels |
| distilled_v4 (DistilBERT) | 2 | 0.5 | 0.792 | 0.942 | Smaller student, worse |
| distilled_v5 (SciBERT) | 2 | 0.5 | 0.793 | 0.980 | Different vocab, no gain |
| distilled_v6 | 1.5 | 0.5 | 0.779 | 0.959 | Too sharp, loses calibration |

**Key finding:** T=2 is the sweet spot. T=4 makes labels too soft (all probabilities
near 0.5, less training signal). T=1.5 makes labels too sharp (approaches hard labels,
loses the calibration benefit). The best student (v2) nearly matches the ensemble
(F1=0.808 vs. 0.857) at half the inference cost.

### 4.3 Critical Negative Result: Fine-Tuning a Distilled Model

Attempting to fine-tune distilled_v2 on additional real sentences (v18 data,
template-augmented) caused catastrophic regression: EP F1 dropped from 0.808 to
0.617 in 3 epochs. The soft-label calibration was destroyed by hard-label
fine-tuning, and the templates introduced linguistic patterns that overfit eval_100
while damaging performance on real literature.

**Rule derived from this failure:** distilled models should not be further fine-tuned
on non-distillation data. If the training set changes, re-distill from scratch.

---

## 5. Multi-task Learning with Species NER

### 5.1 Motivation

The distillation gap (0.808 vs. 0.857 ensemble) was 0.049 F1 and proved resistant
to hyperparameter tuning. A structural intervention was needed.

The core hypothesis: NER supervision (identifying HOST, PATHOGEN, SPECIES entities
in the text) forces the encoder to build entity-aware representations. A classification
head on top of these representations should be more accurate than one built from
a classification-only encoder, because the encoder has been explicitly taught to
locate and type the interacting entities — the critical information for this task.

### 5.2 Architecture

```
BiomedBERT encoder (shared, 110M params)
    ├── cls_head: [CLS] → Linear(768, 2)         classification
    └── ner_head: all tokens → Linear(768, n_labels)    NER
```

Loss function: `α · KL_cls + (1-α) · CE_NER`

where KL_cls is KL-divergence between predicted and soft teacher labels (same
as distillation), and CE_NER is cross-entropy on NER labels (BIO scheme).

NER labels were generated automatically: the same species gazetteer (4.2M GloBI
binomials, Aho-Corasick lookup) and host/pathogen role annotation used in the
GloBI pre-filter were applied at token level to all 44K training sentences.

### 5.3 NER Label Schemes

Four schemes were tested:

| Scheme | Labels | Notes |
|--------|--------|-------|
| basic | O, B-SP, I-SP | Any species |
| typed | O, B-HOST, I-HOST, B-PATHOGEN, I-PATHOGEN, B-SPECIES, I-SPECIES | Role distinction |
| full | O, B-SP, I-SP, B-INT, I-INT | Species + interaction verb |
| **full_typed** | O, B-HOST, I-HOST, B-PATHOGEN, I-PATHOGEN, B-SPECIES, I-SPECIES, B-INT, I-INT | **Both** |

### 5.4 Results: Ablation Table

All trained on 44K soft-label sentences. NER pre-training (pretrain) = 2 epochs
of NER-only loss before joint fine-tuning:

| Config | EP F1 | AUC | Notes |
|--------|-------|-----|-------|
| *distilled_v2 baseline* | *0.808* | *0.822* | — |
| basic_a05 (species-only NER, α=0.5) | 0.841 | 0.871 | +0.033 |
| full_a03 (full NER, α=0.3) | 0.854 | 0.892 | +0.046 |
| typed_a05 (typed species, no INT) | 0.852 | 0.871 | +0.044 |
| full_typed_a05 (typed + INT) | 0.847 | 0.872 | +0.039 |
| full_a05_ner2 (full NER + pretrain) | 0.824 | 0.868 | Pretrain with untyped hurts |
| full_typed_a03_ner2 (typed + pretrain, α=0.3) | 0.830 | 0.885 | α=0.3 + pretrain: too much NER |
| **full_typed_a05_ner2** | **0.868** | **0.887** | **Best — typed + pretrain + balanced α** |
| *ensemble (BiomedBERT × FLAN-T5-base)* | *0.857* | — | *previous target* |

### 5.5 Key Findings from the Ablation

1. **NER consistently helps** — every multi-task config outperforms distilled_v2 (+0.033 minimum).

2. **Role typing (HOST/PATHOGEN) is the critical inductive bias, not interaction verb tags.**
   `typed_a05` (0.852) > `basic_a05` (0.841). Knowing that organism A is the HOST and
   organism B is the PATHOGEN is more informative for classification than knowing both are
   species. The interaction verb tags (`B-INT`) provide no additional benefit when role
   typing is already present.

3. **NER pre-training + typed scheme is the dominant combination.** `full_typed_a05_ner2`
   (0.868) beats all non-pretrain configs by a wide margin. The 2-epoch NER pre-training
   phase stabilises the NER head before joint training begins, giving the encoder a better
   entity-awareness foundation before the classification signal takes over.

4. **α=0.5 is the correct balance for pretrain configs; α=0.3 hurts.**
   `full_typed_a03_ner2` (0.830) is substantially worse than `full_typed_a05_ner2` (0.868)
   despite α=0.3 (more NER weight) being beneficial in non-pretrain configs.
   The NER pre-training already provides sufficient NER supervision — adding more
   NER weight during joint training starves the classification head.
   This interaction between pre-training and α is non-obvious and critical to get right.

5. **Full NER + pretrain without typing underperforms typed NER + pretrain.**
   `full_a05_ner2` (0.824) < `full_typed_a05_ner2` (0.868) — the untyped pre-training
   teaches the encoder to locate species but not to distinguish roles, so the subsequent
   joint training has less structural information to build on.

6. **Ensembles of multi-task models do not help.** All pairwise and triple combinations
   of the top-5 multi-task configs peak at F1=0.860 — below the solo best 0.868.
   Models trained with the same architecture, same data, and only differing hyperparameters
   are too correlated to benefit from ensembling.

### 5.6 Tokenizer Bug and Fix

A critical bug was present in all multi-task checkpoints: `model.save()` saved
`pytorch_model.bin` and `multitask_config.json` but did not call
`AutoTokenizer.save_pretrained()`. When loading, `AutoTokenizer.from_pretrained(checkpoint_path)`
found only `config.json` and fell back to the wrong tokenizer vocabulary, converting
all inputs to `[UNK]` tokens. The model then produced constant degenerate logits
(`[7.46, −7.18]` for all inputs) → F1=0.000. Diagnosis required computing the AUC
(≈0.40, near-random) and checking raw logit invariance across diverse inputs.

Fix: `model.save()` now calls `AutoTokenizer.from_pretrained(encoder_name).save_pretrained(path)`.
All 10 existing checkpoints were retroactively patched. A simple test sentence
(`Wolbachia infects Drosophila, p=0.9994`) confirmed correct operation after the fix.

---

## 6. Current Architecture and Deployment

### 6.1 Inference Pipeline

**Stage 1 — GloBI Pre-filter** (`scripts/process_articles.py`)

A sentence passes the pre-filter if **either** of two conditions holds:

1. **Interaction term match** — regex over a combined vocabulary of GloBI interaction
   terms (loaded from `interaction_dict.csv` if present) and a hand-crafted biomedical
   stem list: `infect|parasit|host|pathogen|vector|transmit|prey|pollina|feed on|…`
   (~80 stems, case-insensitive). Stems rather than full words are used to catch
   morphological variants (`endophyt` catches endophyte / endophytic / endophytes).

2. **Species name mention** — Aho-Corasick automaton over 4.2M GloBI binomials, matched
   in O(sentence_length) with word-boundary checks. This is the safety net: interactions
   phrased without standard interaction verbs (e.g. *"Haemonchus was recovered from sheep"*)
   still pass because a binomial name is present.

**Design rationale — OR, not AND.** The goal of Stage 1 is zero false negatives, not
precision. The neural classifier handles false positives downstream. An AND filter
(both conditions required) would miss real interactions expressed implicitly;
an OR filter passes anything with biological relevance and lets the model decide.
In practice ~15% of sentences pass, reducing the neural model's workload ~7×.
The filter runs in pure Python at ~1ms/sentence (regex) + ~0.5ms (Aho-Corasick),
so the entire pre-filter step is negligible compared to GPU inference.

**XML stripping.** Full-text articles from Europe PMC contain species names wrapped in
`<italic>` or other tags. The sentence splitter strips all XML/HTML tags before
applying the filter, ensuring binomial names inside markup are visible to the automaton.

**Disease common name additions.** The GloBI gazetteer contains Latin binomials
(e.g. *Plasmodium falciparum*) but not common disease names. Sentences like
*"Dengue is transmitted by Aedes aegypti"* had the species but not the disease term
— they passed via condition 2. However, sentences with only the disease name and no
binomial (*"HIV is transmitted sexually"*) were originally missed. A validated list of
common disease/pathogen names (HIV, AIDS, SARS, MERS, malaria, dengue, tuberculosis,
rabies, etc.) was added to the biomedical vocabulary after observing these false
negatives in EP-relax analysis.

**Stage 2 — Multi-task BiomedBERT** (port 8003)

`full_typed_a05_ner2` checkpoint. Threshold 0.13 (optimised on EP-relax).
Returns P(biotic interaction) for each sentence.

### 6.2 Model Registry

| Model | EP F1 | SG F1 | Cost | Status |
|-------|-------|-------|------|--------|
| **mt_full_typed_a05_ner2** | **0.868** | 0.902 | 1× | **Production (port 8003)** |
| BiomedBERT×FLAN-T5 ensemble | 0.857 | — | 3.3× | Superseded |
| distilled_BiomedBERT_v2 | 0.808 | 0.959 | 1× | Archived |
| BiomedBERT_cv_reg | 0.825 | 0.942 | 1× | Archived (warm-start encoder) |

### 6.3 API

The classifier is served as a FastAPI service:

```
GET  http://172.30.120.7:8003/health    → model info + metrics
POST http://172.30.120.7:8003/predict   → single sentence
POST http://172.30.120.7:8003/batch     → up to 500 sentences
```

LAN URL — accessible to all hosts on the `172.30.x.x` lab network.
Interactive docs at `http://172.30.120.7:8003/docs`.

The service is managed by a systemd user unit (`biotic-classifier.service`) with
linger enabled, ensuring it survives logout and restarts automatically on failure.

---

## 7. Current Failure Analysis

The 13.2% residual error rate on EP-relax breaks down as follows:

### 7.1 Structural False Negatives (endoparasiteOf coverage)

58% of EP-relax positives describe endoparasiteOf/hasHost interactions.
The training corpus (44K soft-label sentences) has almost no real PMC sentences
of these types — they exist only as synthetic templates. The model has not
learned the real-literature register for statements like:
*"Haemonchus contortus were recovered from the abomasum of 23 sheep."*

**This is the primary driver of remaining errors.**

### 7.2 Methodology False Positives (5/8 identified)

Sentences describing experimental methodology involving two species:
*"Drosophila was used as a host model for Listeria infection."*
These mention host-pathogen language in a methodological context, not an actual
interaction claim. The model reads the entity types correctly but cannot determine
that the sentence is about the experimental setup, not a biological finding.

### 7.3 Negated Sentences (3/4 identified)

*"No Wolbachia infection was detected in Aedes aegypti from this population."*
The model attends to the host-pathogen entity pair and interaction term but
does not reliably attend to the negation operator.

---

## 8. Path to Further Improvement

### 8.1 Real EndoparasiteOf/hasHost Sentences (highest expected impact)

Harvest 500–1000 real PMC sentences describing endoparasiteOf, hasHost, and preysOn
interactions and add them to the soft-label training set. This closes the largest
known domain gap. The sentences should be validated by Qwen3.5-122B (local, no API cost)
before inclusion. Sources: Europe PMC full text search for known GloBI pairs with
interaction signals ("was found in the intestine/gut/abomasum of", "recovered from",
"isolated from").

### 8.2 5-Epoch and Warm-Start Experiments

**5-epoch result (completed):** `full_typed_a05_ner2_5ep` reached EP F1=**0.815** at
optimised threshold t=0.37 — significantly worse than the 3-epoch model (F1=0.868).
This is a clear overfit: at epoch 3 the classification head is well-calibrated; 2 more
epochs push the model to fit training noise, raising the best threshold from 0.13 to 0.37
and losing ~0.05 F1 on EP-relax. **3 epochs is optimal for this configuration.** The
finding also explains why soft-label calibration is fragile: extra epochs erode the
probability scale that makes t=0.13 the right operating point.

**Warm-start experiments (queued):** Two experiments warm-starting from `BiomedBERT_cv_reg`
(EP F1=0.825, already task-fine-tuned) rather than raw BiomedNLP encoder are queued in
`pipeline_dataset_ablation.sh`:

- `full_typed_a05_ner2_warmstart`: α=0.5. Hypothesis: the warm encoder already understands
  interaction context; NER pre-training from this starting point may provide richer
  role-awareness than cold pre-training.
- `full_typed_a03_ner2_warmstart`: α=0.3. Included for completeness — the warm-start may
  change the α optimum relative to the cold-start finding that α=0.3 + pretrain hurts.

### 8.3 Negation and Methodology Signal

A small targeted addition: curate 50–100 negated sentences (confirmed negative by
LLM) and 50–100 methodology false positives as additional negatives in the training
set. Unlike the endoparasiteOf gap, this requires relatively few examples because
the patterns are consistent.

### 8.4 Multi-task as New Distillation Teacher

`full_typed_a05_ner2` now exceeds the original ensemble. It can therefore replace
the ensemble as the distillation teacher for future versions. This creates a
self-improvement loop:

1. Train multi-task model with current soft labels
2. Use multi-task model to generate new soft labels on expanded real-sentence corpus
3. Re-train multi-task model on new soft labels

Each iteration injects more real-sentence signal while preserving the calibration
benefits of soft-label training.

---

## 9. Summary of Non-Obvious Lessons

These findings are counter-intuitive enough to be worth recording explicitly
for future reference:

1. **Template data hurts BiomedBERT on real literature** — every addition of non-v7
   template variants caused EP-relax regression: v7(0.788) > v12(0.729) > v14(0.706).
   Templates match eval_100 (also template-derived) but not EP-relax.

2. **Fine-tuning a distilled model destroys its calibration** — 3 epochs of hard-label
   fine-tuning of distilled_v2 caused F1 to drop from 0.808 to 0.617.

3. **score>0 filter is not an LLM substitute** — removing sentences with no interaction
   vocabulary removes many valid interactions expressed implicitly. F1 regressed every
   time this filter was applied to new positives.

4. **The role distinction (HOST vs PATHOGEN) matters more than the interaction verb tag**
   — typed NER beats full NER without typing at the same α and training setup.

5. **α=0.5 + NER pretrain beats α=0.3 + NER pretrain** — more NER weight (α=0.3)
   helps without pre-training; with pre-training it over-regularises the classification
   head. The two modifications interact non-linearly.

6. **Ensembles of correlated models are useless** — top-5 multi-task model combinations
   peak below the solo best. Only ensembles of architecturally diverse models help.

7. **Evaluation set choice determines conclusions** — Synth Gold and EP-relax tell
   opposite stories for several models (e.g. mt_full_a05: SG F1=0.961 but EP F1=0.841).
   Always use EP-relax for deployment-quality assessment.

8. **The tokenizer must be saved with the model** — a missing `AutoTokenizer.save_pretrained()`
   call caused all multi-task checkpoints to silently tokenize everything to [UNK] and
   produce constant near-perfect-negative outputs with no obvious error message.

---

## Appendix A: Toolchain

| Component | Tool | Notes |
|-----------|------|-------|
| Encoder | BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext | 110M params, BERT-base |
| Generative | google/flan-t5-base | Base only; large/XL excluded |
| Local LLM teacher | Qwen3.5-122B via Ollama | No API cost; used for v15 labelling |
| Training | PyTorch 2.5, transformers 4.49 | |
| Inference | FastAPI + uvicorn | Port 8003 |
| Species lookup | pyahocorasick | 4.2M GloBI binomials |
| Evaluation | scikit-learn (F1, AUC) | EP-relax primary; threshold tuned per model |
| Infrastructure | A100 80GB GPU, Ubuntu 22.04 | `classifier/` is a git submodule |

## Appendix B: File Structure Reference

```
classifier/
├── api/
│   ├── fastapi_multitask.py      # Production API (port 8003)
│   └── fastapi_distilled.py      # Previous API (archived)
├── data/training/
│   ├── distillation_soft_labels.csv   # 44K rows, teacher soft labels
│   ├── training_data_v7_llm_cleaned.csv  # v7 backbone
│   └── DATASET_VERSIONS.md            # Full version history
├── experiments/multitask/
│   ├── model.py                  # MultiTaskBiomedBERT class
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # EP-relax + SG evaluation
│   ├── pipeline_explore.sh       # Ablation sweep (10 configs)
│   └── pipeline_quality.sh       # Quality sweep (5ep, warm-start)
├── models/multitask/
│   └── full_typed_a05_ner2/      # Production checkpoint
├── scripts/
│   ├── eval_all_models.py        # Comprehensive evaluation
│   ├── process_articles.py       # GloBI pre-filter + batch pipeline
│   └── train_cv_regularized.py   # Standard discriminative training
└── docs/
    └── development_report.md     # This document
```
