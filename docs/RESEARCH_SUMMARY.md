# PhD Research Summary: Knowledge Graph Construction for Host-Pathogen Interactions

## Overview

This document summarizes the research infrastructure developed for constructing mathematically rigorous host-pathogen interaction knowledge graphs from scientific literature.

**Research Question:**
> How can we construct mathematically rigorous host-pathogen interaction knowledge graphs from scientific text, leveraging graph-theoretic and geometric properties for link prediction and discovery?

---

## 1. Data Infrastructure

### 1.1 Host-Pathogen Sentence Corpus

Extracted from existing training data using keyword filtering:

| Metric | Value |
|--------|-------|
| Original dataset | 19,895 sentences |
| Host-pathogen subset | 4,832 sentences (24.3%) |
| Positive samples (has interaction) | 2,358 |
| Negative samples (no interaction) | 2,474 |

**Keyword Distribution:**
```
infect*    : 1,905 matches
virus*     :   945 matches
bacteri*   :   832 matches
disease*   :   400 matches
parasit*   :   369 matches
resistan*  :   352 matches
pathogen*  :   317 matches
```

### 1.2 PHI-base Knowledge Base

Downloaded from Zenodo (12.5 MB):

| Metric | Value |
|--------|-------|
| Total records | 18,984 |
| Pathogen species | 282 |
| Host species | 265 |
| Diseases | 535 |
| Publications | 4,611 |
| Extracted KG triples | 1,797 |

**Top Host Species:**
1. Mus musculus (mouse): 4,963
2. Oryza sativa (rice): 1,611
3. Homo sapiens (human): 1,260

**Top Pathogen Species:**
1. Fusarium graminearum: 1,725
2. Magnaporthe oryzae: 1,444
3. Salmonella enterica: 1,081

---

## 2. Annotation Schema

### 2.1 Entity Types

```
E = {HOST, PATHOGEN, VECTOR, RESERVOIR, DISEASE}
```

| Type | Description | Example |
|------|-------------|---------|
| HOST | Organism that can be infected | Homo sapiens, Mus musculus |
| PATHOGEN | Disease-causing microorganism | SARS-CoV-2, E. coli |
| VECTOR | Organism that transmits pathogens | Aedes aegypti, Ixodes ricinus |
| RESERVOIR | Natural host maintaining pathogen | Bat, Rodent |
| DISEASE | The disease condition | COVID-19, Malaria |

### 2.2 Relation Types

```
R = {INFECTED_BY, TRANSMITS, VECTOR_OF, RESERVOIR_FOR,
     SUSCEPTIBLE_TO, RESISTANT_TO, COLONIZED_BY,
     CAUSES_DISEASE, CO_INFECTS_WITH, ...}
```

**Relation Taxonomy:**

```
                    INTERACTS_WITH
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   INFECTION        TRANSMISSION      DISEASE
        │                │                │
   ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
INFECTED_BY  SUSCEPTIBLE_TO  TRANSMITS  VECTOR_OF  CAUSES_DISEASE
COLONIZED_BY RESISTANT_TO              RESERVOIR_FOR
```

### 2.3 Knowledge Graph Triple Format

```
Triple := (Head_Entity, Relation, Tail_Entity)

Example:
(Homo sapiens, INFECTED_BY, SARS-CoV-2)
(Aedes aegypti, TRANSMITS, Dengue virus)
(Plasmodium falciparum, CAUSES_DISEASE, Malaria)
```

---

## 3. Mathematical Foundations

### 3.1 Relation Extraction Model (SpERT Architecture)

#### 3.1.1 Text Encoding

Given input text $x = (x_1, ..., x_n)$, encode with BERT:

$$\mathbf{H} = \text{BERT}(x) \in \mathbb{R}^{n \times d}$$

where $d = 768$ (hidden dimension).

#### 3.1.2 Span Representation

For a candidate span $(i, j)$, the representation is:

$$\mathbf{h}_{span} = [\mathbf{h}_i; \mathbf{h}_j; \mathbf{w}_{j-i}]$$

where:
- $\mathbf{h}_i \in \mathbb{R}^d$ = hidden state at start position
- $\mathbf{h}_j \in \mathbb{R}^d$ = hidden state at end position
- $\mathbf{w}_{j-i} \in \mathbb{R}^{d/4}$ = learnable width embedding

**Span Classification:**

$$P(\text{entity\_type} | \text{span}) = \text{softmax}(\mathbf{W}_e \cdot \mathbf{h}_{span} + \mathbf{b}_e)$$

#### 3.1.3 Relation Classification

For entity pair $(e_1, e_2)$:

$$\mathbf{h}_{rel} = [\mathbf{h}_{e_1}; \mathbf{h}_{e_2}; \mathbf{c}_{12}]$$

where $\mathbf{c}_{12}$ is the context representation between spans:

$$\mathbf{c}_{12} = \text{Pool}(\mathbf{H}_{e_1.\text{end}:e_2.\text{start}})$$

**Relation Classification:**

$$P(\text{relation} | e_1, e_2) = \text{softmax}(\mathbf{W}_r \cdot \mathbf{h}_{rel} + \mathbf{b}_r)$$

#### 3.1.4 Joint Loss Function

$$\mathcal{L} = \mathcal{L}_{entity} + \mathcal{L}_{relation}$$

$$\mathcal{L}_{entity} = -\sum_{s \in \text{spans}} \log P(y_s | s)$$

$$\mathcal{L}_{relation} = -\sum_{(e_1, e_2) \in \text{pairs}} \log P(r_{12} | e_1, e_2)$$

---

### 3.2 Hyperbolic Embeddings (Poincaré Ball Model)

#### 3.2.1 Poincaré Ball Definition

The Poincaré ball is the open unit ball in $\mathbb{R}^d$:

$$\mathbb{B}^d = \{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\| < 1\}$$

with the Riemannian metric tensor:

$$g_{\mathbf{x}} = \left(\frac{2}{1 - \|\mathbf{x}\|^2}\right)^2 g_E$$

where $g_E$ is the Euclidean metric.

#### 3.2.2 Hyperbolic Distance

The geodesic distance between points $\mathbf{u}, \mathbf{v} \in \mathbb{B}^d$:

$$d(\mathbf{u}, \mathbf{v}) = \text{arcosh}\left(1 + \frac{2\|\mathbf{u} - \mathbf{v}\|^2}{(1 - \|\mathbf{u}\|^2)(1 - \|\mathbf{v}\|^2)}\right)$$

**Key Property:** Distance grows exponentially near the boundary, perfect for trees.

#### 3.2.3 Why Hyperbolic for Taxonomies?

**Euclidean Problem:**
- Tree with branching factor $b$ and depth $L$ has $O(b^L)$ leaves
- Euclidean space needs $O(b^L)$ dimensions to embed without distortion

**Hyperbolic Solution:**
- Hyperbolic space volume grows exponentially with radius
- Same tree embeds in $O(L) = O(\log n)$ dimensions

```
         EUCLIDEAN                    HYPERBOLIC (Poincaré Disk)

            Root                              Root
           /    \                            (center)
          /      \                              •
         A        B                       A •     • B
        /|\      /|\                    •  •  •  •  •
       ... ...  ... ...               (leaves near boundary)

    Linear growth              Exponential growth
    of circumference           of circumference
```

#### 3.2.4 Möbius Addition

The "addition" operation in hyperbolic space:

$$\mathbf{x} \oplus \mathbf{y} = \frac{(1 + 2\langle\mathbf{x}, \mathbf{y}\rangle + \|\mathbf{y}\|^2)\mathbf{x} + (1 - \|\mathbf{x}\|^2)\mathbf{y}}{1 + 2\langle\mathbf{x}, \mathbf{y}\rangle + \|\mathbf{x}\|^2\|\mathbf{y}\|^2}$$

#### 3.2.5 Exponential Map (Tangent Space → Ball)

Maps Euclidean vectors to the Poincaré ball:

$$\exp_{\mathbf{0}}(\mathbf{v}) = \tanh(\|\mathbf{v}\|) \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

#### 3.2.6 Training Loss for Taxonomy Embeddings

**Contrastive Loss:**

For parent-child pair $(p, c)$ and negative sample $n$:

$$\mathcal{L}_{contrastive} = \max(0, d(p, c) - d(p, n) + \gamma)$$

**Hierarchical Constraint:**

Parents should be closer to origin than children:

$$\mathcal{L}_{hierarchy} = \max(0, \|\mathbf{e}_p\| - \|\mathbf{e}_c\| + \epsilon)$$

**Total Loss:**

$$\mathcal{L} = \mathcal{L}_{contrastive} + \lambda \mathcal{L}_{hierarchy}$$

---

### 3.3 Knowledge Graph Embeddings

#### 3.3.1 TransE Model

For triple $(h, r, t)$, the scoring function:

$$f(h, r, t) = -\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|$$

**Intuition:** Relations are translations in embedding space.

```
    Head ──────r──────▶ Tail
     h      + r     =    t
```

#### 3.3.2 RotatE Model

Relations as rotations in complex space:

$$f(h, r, t) = -\|\mathbf{h} \circ e^{i\boldsymbol{\theta}_r} - \mathbf{t}\|$$

where $\circ$ is element-wise product and $\boldsymbol{\theta}_r \in [0, 2\pi)^d$.

**Properties:**
- Symmetric relations: $\theta = 0$ or $\pi$
- Inverse relations: $\theta_{r^{-1}} = -\theta_r$
- Composition: $\theta_{r_1 \circ r_2} = \theta_{r_1} + \theta_{r_2}$

#### 3.3.3 Link Prediction

Given $(h, r, ?)$, predict tail entity:

$$\hat{t} = \arg\max_{t' \in \mathcal{E}} f(h, r, t')$$

**Evaluation Metrics:**
- Hits@k: % of correct entities in top-k predictions
- Mean Reciprocal Rank (MRR): $\frac{1}{|\mathcal{Q}|}\sum_{i=1}^{|\mathcal{Q}|} \frac{1}{\text{rank}_i}$

---

## 4. Architecture Diagrams

### 4.1 Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. TEXT INPUT                                                   │
│  "The tick Ixodes ricinus transmits Borrelia to humans"         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. BERT ENCODING                                                │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐                          │
│  │CLS│The│tick│Ixo│des│ric│inu│...│SEP│  → H ∈ ℝ^(n×768)        │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┘                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────────┐
│ 3a. SPAN CLASSIFICATION │     │ 3b. TAXONOMY EMBEDDINGS         │
│                         │     │                                 │
│ [Ixodes ricinus] → VECTOR    │ Poincaré Ball:                   │
│ [Borrelia] → PATHOGEN   │     │      Bacteria                   │
│ [humans] → HOST         │     │        ●(center)                │
│                         │     │    Borrelia●  ●Salmonella       │
└─────────────────────────┘     │         ●●●●●(boundary)         │
              │                 └─────────────────────────────────┘
              ▼                               │
┌─────────────────────────┐                   │
│ 4. RELATION EXTRACTION  │◄──────────────────┘
│                         │   (taxonomy constraints)
│ (VECTOR, TRANSMITS, PATHOGEN)               │
│ (PATHOGEN, INFECTS, HOST)                   │
└─────────────────────────┘                   │
              │                               │
              ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. KNOWLEDGE GRAPH                                              │
│                                                                  │
│     Ixodes ricinus ──TRANSMITS──▶ Borrelia                      │
│                                      │                           │
│                                   INFECTS                        │
│                                      ▼                           │
│                                   Humans                         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Relation Extraction Model Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    RELATION EXTRACTION MODEL                        │
│                      (113M parameters)                              │
└────────────────────────────────────────────────────────────────────┘

INPUT: "The tick Ixodes ricinus transmits Borrelia burgdorferi"
        ↓
┌────────────────────────────────────────────────────────────────────┐
│  BIOMEDBERT ENCODER                                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [CLS] The tick Ixodes ricinus transmits Borrelia ... [SEP]   │  │
│  │   ↓    ↓   ↓    ↓      ↓        ↓        ↓              ↓    │  │
│  │  h₀   h₁  h₂   h₃     h₄       h₅       h₆            h_n   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              ↓                                      │
│                    H ∈ ℝ^(n × 768)                                 │
└────────────────────────────────────────────────────────────────────┘
                               ↓
        ┌──────────────────────┴──────────────────────┐
        ↓                                              ↓
┌───────────────────────┐                 ┌───────────────────────────┐
│  SPAN CLASSIFIER      │                 │  RELATION CLASSIFIER       │
│                       │                 │                            │
│  For span (i,j):      │                 │  For pair (e₁, e₂):       │
│                       │                 │                            │
│  h_span = [hᵢ; hⱼ; w] │                 │  h_rel = [h_e₁; h_e₂; c]  │
│      ↓                │                 │      ↓                     │
│  ┌───────┐            │                 │  ┌───────┐                 │
│  │Linear │            │                 │  │Linear │                 │
│  │  768  │            │                 │  │ 768   │                 │
│  └───┬───┘            │                 │  └───┬───┘                 │
│      ↓                │                 │      ↓                     │
│  ┌───────┐            │                 │  ┌───────┐                 │
│  │Softmax│            │                 │  │Softmax│                 │
│  │   6   │            │                 │  │  16   │                 │
│  └───────┘            │                 │  └───────┘                 │
│      ↓                │                 │      ↓                     │
│  P(entity_type)       │                 │  P(relation_type)          │
│                       │                 │                            │
│  Classes:             │                 │  Classes:                  │
│  - O (none)           │                 │  - NO_RELATION             │
│  - HOST               │                 │  - INFECTED_BY             │
│  - PATHOGEN           │                 │  - TRANSMITS               │
│  - VECTOR             │                 │  - VECTOR_OF               │
│  - RESERVOIR          │                 │  - CAUSES_DISEASE          │
│  - DISEASE            │                 │  - ... (16 total)          │
└───────────────────────┘                 └───────────────────────────┘
```

### 4.3 Hyperbolic Embedding Space

```
┌────────────────────────────────────────────────────────────────────┐
│               POINCARÉ BALL EMBEDDING (2D visualization)            │
└────────────────────────────────────────────────────────────────────┘

                         Euclidean                Hyperbolic

                            │                    ╭───────────╮
      Tree                  │                   ╱    ROOT     ╲
     Depth                  │                  │    (center)   │
                            │                  │       ●       │
        1        Root       │                  │    ╱  │  ╲    │
                 /  \       │                  │   ●   ●   ●   │  ← Depth 2
        2       A    B      │                  │  ╱│╲ ╱│╲ ╱│╲  │
               /|\  /|\     │                  │ ●●● ●●● ●●●   │  ← Depth 3
        3     ........      │                  │●●●●●●●●●●●●●●●│  ← Leaves
                            │                   ╲  (boundary)  ╱
                            │                    ╰─────────────╯

     PROBLEM:               │                   SOLUTION:
     Leaves grow as O(bᴸ)   │                   Boundary has exp growth
     Need O(bᴸ) dimensions  │                   Need O(L) dimensions

┌────────────────────────────────────────────────────────────────────┐
│  DISTANCE FORMULA                                                   │
│                                                                     │
│                        ⎛        2‖u - v‖²              ⎞           │
│  d(u,v) = arcosh ⎜ 1 + ─────────────────────── ⎟                   │
│                        ⎝    (1-‖u‖²)(1-‖v‖²)          ⎠           │
│                                                                     │
│  Properties:                                                        │
│  • d(0, x) grows logarithmically as ‖x‖ → 1                        │
│  • Preserves tree metric with low distortion                        │
│  • Parents (small ‖x‖) naturally closer to root                    │
└────────────────────────────────────────────────────────────────────┘
```

### 4.4 Knowledge Graph Embedding (TransE/RotatE)

```
┌────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH EMBEDDINGS                       │
└────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
                              TransE
═══════════════════════════════════════════════════════════════════════

  Triple: (Human, INFECTED_BY, SARS-CoV-2)

  Embedding Space:

       h + r ≈ t

       Human ─────────INFECTED_BY─────────▶ SARS-CoV-2
         ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶●
         h              r                   t

  Score: f(h,r,t) = -‖h + r - t‖

  Training: Push h + r closer to t, away from t' (negatives)

═══════════════════════════════════════════════════════════════════════
                              RotatE
═══════════════════════════════════════════════════════════════════════

  Triple: (Human, INFECTED_BY, SARS-CoV-2)

  Complex Embedding Space:

       h ∘ eⁱᶿ ≈ t     (rotation by θ)

                     θ = rotation angle
                   ╱
                 ╱
       Human   ╱    SARS-CoV-2
         ●────╱────────●
         h    ↺         t
              eⁱᶿʳ

  Advantages:
  • Symmetric relations:  θ = π  (180° rotation = self-inverse)
  • Inverse relations:    θ₋ᵣ = -θᵣ
  • Composition:          θᵣ₁∘ᵣ₂ = θᵣ₁ + θᵣ₂
```

---

## 5. Files Created

```
classifier/
├── docs/
│   └── RESEARCH_SUMMARY.md          # This document
├── src/
│   ├── annotation/
│   │   └── annotator.py             # Interactive annotation tool
│   └── models/
│       ├── relation_extractor.py    # SpERT-style joint NER+RE model
│       └── hyperbolic_embeddings.py # Poincaré ball embeddings
├── data/
│   ├── external/
│   │   └── phi_base/
│   │       ├── phi-base_4.13_data.csv   # PHI-base (18,984 records)
│   │       ├── phi_base_entities.json   # Entity lists
│   │       └── phi_base_kg_triples.csv  # Extracted triples
│   ├── schemas/
│   │   ├── host_pathogen_annotation_schema.json
│   │   └── example_annotation.json
│   └── training/
│       └── training_data_host_pathogen_subset.csv  # 4,832 sentences
└── scripts/
    ├── extract_host_pathogen_subset.py
    └── explore_phi_base.py
```

---

## 6. Next Steps

1. **Annotation**: Label 100-200 sentences using the annotation tool
2. **Train RE Model**: Fine-tune relation extractor on annotated data
3. **Train Hyperbolic Embeddings**: Embed NCBI taxonomy in Poincaré ball
4. **Integration**: Use taxonomy embeddings as constraints for link prediction
5. **Evaluation**: Compare against PHI-base gold standard

---

## 7. Publication Potential

| Paper Idea | Math Focus | Venue |
|------------|------------|-------|
| Hyperbolic KG Embeddings for Taxonomies | Poincaré geometry, tree metrics | EMNLP, NeurIPS |
| Taxonomically-Constrained Link Prediction | Regularization, spectral methods | Bioinformatics |
| Joint NER+RE for Host-Pathogen Extraction | Span representations, attention | ACL, NAACL |

---

*Generated: 2026-01-27*
*Project: MetaP - Host-Pathogen Knowledge Graph Construction*
