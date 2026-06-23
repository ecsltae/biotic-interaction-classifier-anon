# PhD Reading List: Knowledge Graphs, Hyperbolic Geometry & Biomedical NLP

Organized by topic, prioritized within each section (read top papers first).

---

## TIER 1 — Core Foundations (Read First)

These are the mathematical and methodological foundations everything else builds on.

### 1A. Hyperbolic Geometry & Embeddings

| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 1 | **Nickel & Kiela** — [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039) | NeurIPS 2017 | **THE foundational paper.** Introduces Poincaré ball embeddings for hierarchies. Your taxonomy work builds directly on this. Code: [github.com/facebookresearch/poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings) |
| 2 | **Ganea et al.** — [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112) | NeurIPS 2018 | Extends Poincaré embeddings to full neural networks. Defines hyperbolic linear layers, attention, etc. |
| 3 | **Chami et al.** — [Hyperbolic Graph Convolutional Neural Networks (HGCN)](https://arxiv.org/abs/1910.12933) | NeurIPS 2019 | GCN operations in hyperbolic space. 63% error reduction on link prediction. Directly relevant to your KG work. Code: [github.com/HazyResearch/hgcn](https://github.com/HazyResearch/hgcn) |
| 4 | **Nickel & Kiela** — [Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry](https://arxiv.org/abs/1806.03417) | ICML 2018 | Alternative to Poincaré: Lorentz model. More numerically stable for optimization. |

### 1B. Knowledge Graph Embeddings

| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 5 | **Bordes et al.** — [Translating Embeddings for Modeling Multi-relational Data (TransE)](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html) | NeurIPS 2013 | Foundational KG embedding. Simple: h + r ≈ t. Everything else extends this. |
| 6 | **Sun et al.** — [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://openreview.net/pdf?id=HkgEQnRqYQ) | ICLR 2019 | Rotation-based embeddings in complex space. Handles symmetric, antisymmetric, inverse, and composition patterns. |
| 7 | **Trouillon et al.** — [Complex Embeddings for Simple Link Prediction (ComplEx)](https://arxiv.org/abs/1606.06357) | ICML 2016 | Complex-valued embeddings. Naturally models asymmetric relations (infection directionality). |
| 8 | **Cao et al.** — [Knowledge Graph Embedding: A Survey from the Perspective of Representation Spaces](https://dl.acm.org/doi/10.1145/3643806) | ACM Computing Surveys 2024 | **Comprehensive 2024 survey.** Covers TransE, RotatE, ComplEx, and newer methods. Good for positioning your work. |

---

## TIER 2 — Biomedical Applications (Read Second)

How the above methods are applied in biology and medicine.

### 2A. Biomedical Knowledge Graphs

| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 9 | **Mohamed et al.** — [Benchmark and Best Practices for Biomedical Knowledge Graph Embeddings](https://pmc.ncbi.nlm.nih.gov/articles/PMC7971091/) | BioData Mining 2021 | Compares TransE, ComplEx, DistMult, RotatE on biomedical KGs. Essential baseline for your work. |
| 10 | **Bonner et al.** — [A Review of Biomedical Datasets Relating to Drug Discovery: A Knowledge Graph Perspective](https://arxiv.org/abs/2102.10062) | Briefings in Bioinformatics 2022 | Survey of biomedical KG datasets. Helps you position your host-pathogen KG. |
| 11 | **Santos et al.** — [BioKGrapher: Knowledge Graph Construction for Biomedical Applications](https://www.sciencedirect.com/science/article/pii/S2001037024003386) | Computational & Structural Biotech Journal 2024 | Automated KG construction from biomedical text. Directly comparable pipeline. |

### 2B. Host-Pathogen Interaction Prediction

| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 12 | **Barmada et al.** — [Predicting Host-Pathogen Interactions with ML: A Scoping Review](https://www.sciencedirect.com/science/article/pii/S1567134825000401) | Infection, Genetics & Evolution 2025 | **Most recent review.** 30 articles from 2019-2024. Maps the field for your thesis introduction. |
| 13 | **Loaiza et al.** — [deepHPI: Deep Learning for Host-Pathogen Protein Interaction Prediction](https://pubmed.ncbi.nlm.nih.gov/35511057/) | Briefings in Bioinformatics 2022 | Neural network approach to HPI prediction. Benchmark dataset available on Figshare. |
| 14 | **Le et al.** — [PHILM2Web: Host-Pathogen Interaction Database](https://academic.oup.com/database/article/doi/10.1093/database/baac042/6625823) | Database 2022 | The text-mined HPI database. 23,581 interactions from PubMed. Directly feeds your KG. |
| 15 | **Urban et al.** — [PHI-base: The Multi-species Pathogen-Host Interaction Database in 2025](https://academic.oup.com/nar/article/53/D1/D826/7908791) | NAR 2025 | Gold standard for validation. You already downloaded v4.13. |

### 2C. Hyperbolic Embeddings for Biology

| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 16 | **Corso et al.** — [Learning Hyperbolic Embedding for Phylogenetic Tree Placement](https://www.mdpi.com/2079-7737/11/9/1256) | Biology 2022 | Poincaré embeddings for phylogenetic trees. Fewer parameters, better phylogenetic distances. |
| 17 | **Penn & Scheidwasser-Clow** — [Differentiable Phylogenetics via Hyperbolic Embeddings (Dodonaphy)](https://academic.oup.com/bioinformaticsadvances/article/4/1/vbae082/7696335) | Bioinformatics Advances 2024 | Variational Bayesian phylogenetics in hyperbolic space. State-of-the-art for your taxonomy embedding. |

---

## TIER 3 — NLP Methods (Read Third)

The NLP techniques you'll use for text-to-graph extraction.

### 3A. Joint Entity & Relation Extraction

| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 18 | **Eberts & Ulges** — [SpERT: Span-based Joint Entity and Relation Extraction](https://arxiv.org/abs/1909.07755) | ECAI 2020 | **Architecture your model is based on.** Span classification + relation classification. Code available. |
| 19 | **Cabot & Navigli** — [REBEL: Relation Extraction By End-to-end Language Generation](https://aclanthology.org/2021.findings-emnlp.204/) | EMNLP 2021 | Generative approach: seq2seq produces triples directly. Alternative to SpERT for your pipeline. |
| 20 | **Huguet Cabot et al.** — [Enhancing Relation Extraction from Biomedical Texts by LLMs](https://link.springer.com/chapter/10.1007/978-3-031-60615-1_1) | CLEF 2024 | Combines LLMs + seq2seq + classification for biomedical RE. Three complementary approaches. |
| 21 | **Tran et al.** — [Enhancing Biomedical RE with Directionality](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i68/8199369) | Bioinformatics 2025 | Adds directionality to RE — critical for host→pathogen vs pathogen→host. |

### 3B. Biomedical NER & RE Surveys

| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 22 | **Wadhwa et al.** — [Surveying Biomedical Relation Extraction: Datasets and New Resource](https://academic.oup.com/bib/article/25/3/bbae132/7644532) | Briefings in Bioinformatics 2024 | Critical examination of current datasets. Proposes new benchmark. |
| 23 | **Li et al.** — [A Comprehensive Survey on Relation Extraction](https://dl.acm.org/doi/full/10.1145/3674501) | ACM Computing Surveys 2024 | 137 papers reviewed. Maps the entire RE landscape. Good for related work section. |
| 24 | **Uddin et al.** — [NER and RE for Biomedical Text: Comprehensive Survey](https://www.sciencedirect.com/science/article/abs/pii/S0925231224019428) | Neurocomputing 2024 | Recent advances in biomedical NER+RE. Covers BERT variants, transformers. |

### 3C. Few-Shot & Low-Resource Learning

| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 25 | **Chen et al.** — [Prompt Tuning in Biomedical Relation Extraction](https://pmc.ncbi.nlm.nih.gov/articles/PMC11052745/) | J. Biomed. Informatics 2024 | Prompt tuning for biomedical RE. Outperforms fine-tuning in few-shot. |
| 26 | **Zhang et al.** — [Few-shot Medical RE via Prompt Tuning Enhanced PLM](https://www.sciencedirect.com/science/article/abs/pii/S0925231225004242) | Neurocomputing 2025 | Latest prompt-based few-shot RE for medical texts. |
| 27 | **Wang et al.** — [Relation Extraction in Underexplored Biomedical Domains](https://direct.mit.edu/coli/article/50/3/953/121178/) | Computational Linguistics 2024 | Diversity-optimized sampling + synthetic data for rare domains. Relevant to host-pathogen. |

---

## TIER 4 — Advanced Mathematical Directions (Read for Depth)

For the mathematical rigor your PI wants.

### 4A. Causal Inference on Graphs

| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 28 | **Pearl** — [Causality: Models, Reasoning, and Inference](https://doi.org/10.1017/CBO9780511803161) | Cambridge 2009 | **The bible of causal inference.** Do-calculus, d-separation, backdoor criterion. Read Chapters 1-3. |
| 29 | **Chen et al.** — [Causal Inference Meets Deep Learning: A Comprehensive Survey](https://spj.science.org/doi/10.34133/research.0467) | Research 2024 | How DL and causal inference combine. Covers GNN-based causal methods. |
| 30 | **Job et al.** — [Exploring Causal Learning Through Graph Neural Networks](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.70024) | WIREs Data Mining 2025 | Latest review on GNN + causal inference integration. |
| 31 | **Yang et al.** — [Reconstructing Molecular Networks by Causal Diffusion Do-Calculus](https://pmc.ncbi.nlm.nih.gov/articles/PMC11633463/) | 2024 | Do-calculus applied to gene regulatory networks. Template for host-pathogen causal networks. |

### 4B. Graph Neural Networks for Biomedicine

| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 32 | **Kipf & Welling** — [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) | ICLR 2017 | Foundational GCN paper. Spectral graph theory → message passing. |
| 33 | **Veličković et al.** — [Graph Attention Networks (GAT)](https://arxiv.org/abs/1710.10903) | ICLR 2018 | Attention mechanism over graph neighbors. Key for learning interaction importance. |
| 34 | **Yang et al.** — [Hyperbolic Graph Neural Networks: A Review](https://arxiv.org/abs/2202.13852) | 2022 | Comprehensive review of hyperbolic GNNs. Covers all variants and applications. |

### 4C. GNN for Drug Discovery & Link Prediction

| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 35 | **Li et al.** — [Graph Neural Networks in Modern AI-Aided Drug Discovery](https://pubs.acs.org/doi/10.1021/acs.chemrev.5c00461) | Chemical Reviews 2025 | Latest comprehensive review. Shows where the field is heading. |
| 36 | **Santos et al.** — [Knowledge Graphs for Drug Repurposing: From ML to GNN](https://www.sciencedirect.com/science/article/pii/S0010482525012247) | Computers in Biology & Medicine 2025 | KG + GNN for drug repurposing. Similar methodology to host-pathogen prediction. |

---

## TIER 5 — Bonus / Textbooks (Reference as Needed)

### Textbooks

| # | Book | Why |
|---|------|-----|
| T1 | **Bronstein et al.** — [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478) (2021) | Free textbook. Unifies CNNs, GNNs, Transformers under geometric framework. Chapter on hyperbolic geometry. |
| T2 | **Hamilton** — [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) (2020) | Free book. GNNs, knowledge graphs, link prediction. Excellent for PhD foundations. |
| T3 | **Pearl, Glymour & Jewell** — [Causal Inference in Statistics: A Primer](https://doi.org/10.1002/9781119186441) (2016) | Accessible introduction to causal inference. Lighter than Pearl's full book. |

### Useful GitHub Repositories

| Repo | Description |
|------|-------------|
| [Awesome Hyperbolic](https://github.com/marlin-codes/Awesome-Hyperbolic-Representation-and-Deep-Learning) | Curated list of hyperbolic DL papers |
| [Awesome Biomedical KGs](https://github.com/YuxingLu613/awesome-biomedical-knowledge-graphs) | Biomedical KG databases, tools, papers |
| [Facebook Poincaré Embeddings](https://github.com/facebookresearch/poincare-embeddings) | Reference implementation |
| [HazyResearch HGCN](https://github.com/HazyResearch/hgcn) | Hyperbolic GCN code |

---

## Suggested Reading Order

### Phase 1: Foundations (2-3 weeks)
1. Paper #1 (Poincaré Embeddings) — understand hyperbolic geometry
2. Paper #5 (TransE) — understand KG embeddings
3. Paper #18 (SpERT) — understand joint NER+RE
4. Paper #32 (GCN) — understand graph neural networks

### Phase 2: Domain (2-3 weeks)
5. Paper #12 (HPI Scoping Review) — map your specific field
6. Paper #9 (Biomedical KGE Benchmark) — know the baselines
7. Paper #8 (KGE Survey 2024) — positioning for your thesis
8. Paper #6 (RotatE) — the method you'll likely use

### Phase 3: Advanced Methods (2-3 weeks)
9. Paper #3 (HGCN) — hyperbolic + graphs
10. Paper #16 or #17 (Hyperbolic phylogenetics) — biology application
11. Paper #25 (Prompt tuning for RE) — few-shot direction
12. Paper #29 (Causal + DL survey) — future direction

### Phase 4: Deep Dive (ongoing)
13. Textbook T1 (Geometric Deep Learning) — mathematical depth
14. Paper #28 (Pearl Causality) — if pursuing causal direction
15. Remaining papers as needed

---

*Compiled: 2026-02-17*
