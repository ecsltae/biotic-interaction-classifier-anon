#!/usr/bin/env python3
"""
Multi-task dataset: interaction classification + species NER + interaction-term NER.

NER label generation strategy (two layers):
  1. SPECIES — from source_species/target_species columns (gold annotations).
     Rows with no species annotation: auto-tagged via Aho-Corasick gazetteer
     (species_dict.csv + species_names_cache.json + training data species).
  2. INTERACTION TERMS — from interaction_dict.csv (591 GloBI terms) + a biomedical
     vocabulary, tagged as B-INT/I-INT in "full" and "full_typed" schemes.

Label schemes (must match model.py LABEL_SETS):
  "basic"      : O, B-SP, I-SP
  "typed"      : O, B-HOST, I-HOST, B-PATHOGEN, I-PATHOGEN, B-SPECIES, I-SPECIES
  "full"       : O, B-SP, I-SP, B-INT, I-INT                (recommended)
  "full_typed" : O, B-HOST, I-HOST, B-PATHOGEN, I-PATHOGEN, B-SPECIES, I-SPECIES, B-INT, I-INT

Gazetteer sources (loaded lazily, cached globally):
  - classifier/data/processed/species_dict.csv      (~4.2M clean binomials)
  - classifier/data/species_names_cache.json        (~21K + common names)
  - source_species / target_species from training CSV
  - classifier/data/processed/interaction_dict.csv  (591 terms)
"""

import json
import re
import threading
from pathlib import Path
from typing import Optional

import ahocorasick
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Mirrors model.py — keep in sync
LABEL_SETS = {
    "basic":      ["O", "B-SP", "I-SP"],
    "typed":      ["O", "B-HOST", "I-HOST", "B-PATHOGEN", "I-PATHOGEN", "B-SPECIES", "I-SPECIES"],
    "full":       ["O", "B-SP", "I-SP", "B-INT", "I-INT"],
    "full_typed": ["O", "B-HOST", "I-HOST", "B-PATHOGEN", "I-PATHOGEN",
                   "B-SPECIES", "I-SPECIES", "B-INT", "I-INT"],
}

_AMBIGUOUS_TYPES = {
    "mutualism", "symbiosis", "commensalism", "competition",
    "none_two_species", "none_three_species", "cooccurrence",
}

# ── Paths (relative to project root; resolved at build time) ─────────────────

def _project_root() -> Path:
    """Walk up from this file until we find MPvenv/ (only at top-level MetaP root)."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "MPvenv").exists():
            return parent
    return p.parents[4]  # fallback


_ROOT = _project_root()
_SPECIES_DICT_CSV   = _ROOT / "classifier/data/processed/species_dict.csv"
_SPECIES_CACHE_JSON = _ROOT / "classifier/data/species_names_cache.json"
_INTERACTION_CSV    = _ROOT / "classifier/data/processed/interaction_dict.csv"

# Biomedical interaction vocabulary not in GloBI (validated 100% recall on EP-relax)
_BIOMEDICAL_INTERACTION_TERMS = [
    "infect", "infected by", "infection of", "parasitizes", "parasitized by",
    "host of", "is a host", "pathogen of", "vector of", "transmits", "transmitted by",
    "zoonotic", "symbion", "symbiotic", "endophyte", "mycorrhiza", "mycorrhizal",
    "nodule", "nematode", "fungal", "fungi", "bacterial", "virus", "viral", "protozoa",
    "reservoir of", "definitive host", "intermediate host",
    "harbours", "harbors", "colonizes", "colonises", "life cycle",
    "prey", "predator", "predation", "pollinator", "pollination",
    "feed on", "feeds on", "eats", "ingests",
    "herbivory", "herbivore", "mutualism", "commensalism", "kleptoparasit",
]


# ── Global automata (built once, thread-safe) ─────────────────────────────────

_species_automaton: Optional[ahocorasick.Automaton] = None
_interaction_automaton: Optional[ahocorasick.Automaton] = None
_build_lock = threading.Lock()


def _build_species_automaton(extra_species: Optional[set] = None) -> ahocorasick.Automaton:
    """Build case-insensitive Aho-Corasick automaton over all known species names."""
    print("Building species gazetteer automaton...", flush=True)
    A = ahocorasick.Automaton()

    def add(name: str):
        name = name.strip()
        if len(name) < 4:
            return
        key = name.lower()
        if key not in A:
            A.add_word(key, name)

    # 1. Clean binomials from species_dict.csv (~4.2M, Genus species pattern)
    if _SPECIES_DICT_CSV.exists():
        binomial_re = re.compile(r'^[A-Z][a-z]+ [a-z]+$')
        count = 0
        with open(_SPECIES_DICT_CSV) as f:
            next(f)  # skip header
            for line in f:
                name = line.strip()
                if binomial_re.match(name):
                    add(name)
                    count += 1
        print(f"  Species dict (clean binomials): {count:,}", flush=True)

    # 2. species_names_cache.json — scientific + common names
    if _SPECIES_CACHE_JSON.exists():
        cache = json.load(open(_SPECIES_CACHE_JSON))
        for sci, commons in cache.items():
            add(sci)
            for c in commons:
                add(c)
        print(f"  Species cache (sci+common): {len(cache):,} entries", flush=True)

    # 3. Extra species from training data source_species / target_species
    if extra_species:
        for name in extra_species:
            add(name)
        print(f"  Training data species: {len(extra_species):,}", flush=True)

    A.make_automaton()
    print(f"  Total species automaton keys: {len(A):,}", flush=True)
    return A


def _build_interaction_automaton() -> ahocorasick.Automaton:
    """Build case-insensitive automaton over interaction terms (GloBI + biomedical)."""
    A = ahocorasick.Automaton()

    def add(term: str):
        term = term.strip()
        if len(term) < 3:
            return
        key = term.lower()
        if key not in A:
            A.add_word(key, term)

    # GloBI interaction_dict.csv
    if _INTERACTION_CSV.exists():
        df = pd.read_csv(_INTERACTION_CSV)
        col = df.columns[0]
        for term in df[col].dropna():
            add(str(term))
        print(f"  GloBI interaction terms: {len(df):,}", flush=True)

    # Biomedical vocabulary
    for term in _BIOMEDICAL_INTERACTION_TERMS:
        add(term)

    A.make_automaton()
    print(f"  Total interaction automaton keys: {len(A):,}", flush=True)
    return A


def get_automata(extra_species: Optional[set] = None):
    """Return (species_automaton, interaction_automaton), building once."""
    global _species_automaton, _interaction_automaton
    with _build_lock:
        if _species_automaton is None:
            _species_automaton = _build_species_automaton(extra_species)
        if _interaction_automaton is None:
            _interaction_automaton = _build_interaction_automaton()
    return _species_automaton, _interaction_automaton


# ── Span matching helpers ─────────────────────────────────────────────────────

def _parse_json_list(val) -> list[str]:
    """Parse a JSON-encoded list or comma-string from a DataFrame cell → list[str]."""
    if not val or (isinstance(val, float)):
        return []
    if isinstance(val, list):
        return [s.strip() for s in val if s and str(s).strip()]
    s = str(val).strip()
    if not s or s in ("nan", "[]", ""):
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (json.JSONDecodeError, ValueError):
        pass
    return [x.strip() for x in s.split(",") if x.strip()]


def _find_spans_exact(text: str, name: str) -> list[tuple[int, int]]:
    """All [start, end) char spans of `name` in `text`, case-insensitive, word-boundary aware."""
    if not name or not isinstance(name, str):
        return []
    name = name.strip()
    if not name:
        return []
    spans = []
    pattern = re.compile(r'(?<!\w)' + re.escape(name) + r'(?!\w)', re.IGNORECASE)
    for m in pattern.finditer(text):
        spans.append((m.start(), m.end()))
    return spans


def _find_spans_automaton(
    text: str, automaton: ahocorasick.Automaton
) -> list[tuple[int, int]]:
    """
    Find all non-overlapping, longest-match spans using the automaton.
    Aho-Corasick finds all occurrences (overlapping); we keep word-boundary matches only.
    """
    if len(automaton) == 0:
        return []
    text_lower = text.lower()
    # ahocorasick iter yields (end_index, stored_value); stored_value is the original-case name
    found = []
    for end_idx, value in automaton.iter(text_lower):
        key = value.lower()
        start = end_idx - len(key) + 1
        end = end_idx + 1
        # Word boundary check
        before_ok = start == 0 or not text_lower[start - 1].isalpha()
        after_ok  = end >= len(text_lower) or not text_lower[end].isalpha()
        if before_ok and after_ok:
            found.append((start, end))

    if not found:
        return []

    # Remove overlaps: keep longest match, prefer earlier start
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    merged = [found[0]]
    for s, e in found[1:]:
        if s >= merged[-1][1]:
            merged.append((s, e))
        elif e - s > merged[-1][1] - merged[-1][0]:
            merged[-1] = (s, e)
    return merged


# ── Core NER label builder ─────────────────────────────────────────────────────

def _build_token_ner_labels(
    text: str,
    offsets: list[tuple[int, int]],
    n_tokens: int,
    source_species: Optional[str],
    target_species: Optional[str],
    interaction_type: Optional[str],
    ner_scheme: str,
    species_auto: ahocorasick.Automaton,
    interaction_auto: ahocorasick.Automaton,
    has_gold_species: bool,
    source_sp_forms: Optional[list[str]] = None,
    target_sp_forms: Optional[list[str]] = None,
    gold_int_forms:  Optional[list[str]] = None,
) -> list[int]:
    """
    Build BIO token labels for one sentence.

    Priority order (later writes can overwrite earlier):
      1. O (default)
      2. Gazetteer auto-tagging (if no gold species)
      3. Gold source/target species:
           - If source_sp_forms/target_sp_forms are present (SibiLS surface forms): use directly
           - Else fall back to canonical source_species/target_species string matching
      4. Interaction terms:
           - If gold_int_forms present (SibiLS surface forms): use directly
           - Else scan with interaction automaton
    Special tokens (offset == (0,0)) → -100.
    """
    labels = [0] * n_tokens

    # Mask special tokens
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            labels[i] = -100

    if ner_scheme not in LABEL_SETS:
        return labels

    label2id = {l: idx for idx, l in enumerate(LABEL_SETS[ner_scheme])}
    has_sp_tags  = "B-SP" in label2id
    has_host_tag = "B-HOST" in label2id
    has_int_tags = "B-INT" in label2id

    def apply_spans(spans: list[tuple[int, int]], b_tag: int, i_tag: int):
        for span_s, span_e in spans:
            first = True
            for i, (ts, te) in enumerate(offsets):
                if ts == 0 and te == 0:
                    continue
                if te > span_s and ts < span_e:
                    labels[i] = b_tag if first else i_tag
                    first = False

    ambiguous = (
        not interaction_type
        or str(interaction_type).lower().strip() in _AMBIGUOUS_TYPES
    )

    # ── 1. Auto-tagging from gazetteer (only when no gold annotation) ────────
    if not has_gold_species:
        if has_sp_tags:
            b, i_ = label2id["B-SP"], label2id["I-SP"]
        elif has_host_tag:
            b, i_ = label2id["B-SPECIES"], label2id["I-SPECIES"]
        else:
            b, i_ = 0, 0

        if b != 0:
            auto_spans = _find_spans_automaton(text, species_auto)
            apply_spans(auto_spans, b, i_)

    # ── 2. Gold source/target species (overwrite auto tags) ──────────────────
    if has_gold_species:
        def _parse_names(s):
            return [n.strip() for n in str(s).split(",") if n.strip()] if s and isinstance(s, str) else []

        # Use SibiLS surface forms if available (exact match, handles abbreviations)
        # Otherwise fall back to canonical species name
        src_names = source_sp_forms if source_sp_forms else _parse_names(source_species)
        tgt_names = target_sp_forms if target_sp_forms else _parse_names(target_species)

        if has_sp_tags:
            b, i_ = label2id["B-SP"], label2id["I-SP"]
            for name in src_names + tgt_names:
                apply_spans(_find_spans_exact(text, name), b, i_)

        elif has_host_tag:
            if ambiguous:
                b, i_ = label2id["B-SPECIES"], label2id["I-SPECIES"]
                for name in src_names + tgt_names:
                    apply_spans(_find_spans_exact(text, name), b, i_)
            else:
                for name in src_names:
                    apply_spans(_find_spans_exact(text, name),
                                label2id["B-HOST"], label2id["I-HOST"])
                for name in tgt_names:
                    apply_spans(_find_spans_exact(text, name),
                                label2id["B-PATHOGEN"], label2id["I-PATHOGEN"])

    # ── 3. Interaction terms ──────────────────────────────────────────────────
    if has_int_tags:
        b_int = label2id["B-INT"]
        i_int = label2id["I-INT"]
        # Use SibiLS gold surface forms if available, else automaton scan
        if gold_int_forms:
            int_spans = []
            for form in gold_int_forms:
                int_spans.extend(_find_spans_exact(text, form))
        else:
            int_spans = _find_spans_automaton(text, interaction_auto)
        # Only overwrite O tokens (don't clobber species tags)
        for span_s, span_e in int_spans:
            first = True
            for i, (ts, te) in enumerate(offsets):
                if ts == 0 and te == 0:
                    continue
                if te > span_s and ts < span_e:
                    if labels[i] == 0:  # only O positions
                        labels[i] = b_int if first else i_int
                    first = False

    return labels


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiTaskDataset(Dataset):
    """
    PyTorch Dataset for joint classification + NER.

    Each item:
        input_ids       (seq_len,)
        attention_mask  (seq_len,)
        token_type_ids  (seq_len,)
        cls_label       scalar int — 0/1
        ner_labels      (seq_len,) — int, -100 for masked positions
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_name: str,
        ner_scheme: str = "full",
        max_length: int = 256,
        extra_species: Optional[set] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer   = AutoTokenizer.from_pretrained(tokenizer_name)
        self.ner_scheme  = ner_scheme
        self.max_length  = max_length
        # Build automata once (shared across instances via globals)
        self.sp_auto, self.int_auto = get_automata(extra_species)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row["text"])
        # Soft label from ensemble (p_ensemble column, -1.0 if absent → use hard label)
        soft_label = float(row["p_ensemble"]) if "p_ensemble" in row.index and not pd.isna(row.get("p_ensemble")) else -1.0

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offsets        = enc["offset_mapping"].squeeze(0).tolist()

        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(0)
        else:
            token_type_ids = torch.zeros_like(input_ids)

        # Prefer gold surface forms from SibiLS (fetch_sibils_ner_data.py output)
        # over canonical names from source_species/target_species.
        # Surface forms are exact strings guaranteed to appear in the text.
        source_sp_forms = _parse_json_list(row.get("source_sp_forms", None))
        target_sp_forms = _parse_json_list(row.get("target_sp_forms", None))
        int_forms       = _parse_json_list(row.get("interaction_forms", None))

        # Fall back to canonical species names if no surface forms
        source_sp = row.get("source_species", None)
        target_sp = row.get("target_species", None)
        itype     = row.get("interaction_type", None)

        has_gold = bool(source_sp_forms or target_sp_forms) or (
            (isinstance(source_sp, str) and source_sp.strip()) or
            (isinstance(target_sp, str) and target_sp.strip())
        )

        ner_labels_list = _build_token_ner_labels(
            text       = text,
            offsets    = offsets,
            n_tokens   = len(input_ids),
            source_species       = source_sp,
            target_species       = target_sp,
            interaction_type     = itype,
            ner_scheme           = self.ner_scheme,
            species_auto         = self.sp_auto,
            interaction_auto     = self.int_auto,
            has_gold_species     = has_gold,
            source_sp_forms      = source_sp_forms,
            target_sp_forms      = target_sp_forms,
            gold_int_forms       = int_forms,
        )

        # Mask padded positions
        for i, mask in enumerate(attention_mask.tolist()):
            if mask == 0:
                ner_labels_list[i] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cls_label":      torch.tensor(int(row["label"]), dtype=torch.long),
            "soft_label":     torch.tensor(soft_label, dtype=torch.float),
            "ner_labels":     torch.tensor(ner_labels_list, dtype=torch.long),
        }


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_multitask_splits(
    csv_path: str,
    tokenizer_name: str,
    ner_scheme: str = "full",
    val_frac: float = 0.1,
    seed: int = 42,
    max_length: int = 256,
    soft_labels_path: Optional[str] = None,
) -> tuple["MultiTaskDataset", "MultiTaskDataset"]:
    """Stratified train/val split. Merges soft labels if provided."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    # Accept hard_label as alias for label (distillation_soft_labels.csv convention)
    if "label" not in df.columns and "hard_label" in df.columns:
        df = df.rename(columns={"hard_label": "label"})
    if {"text", "label"} - set(df.columns):
        raise ValueError(f"Missing required columns in {csv_path}")

    # Merge ensemble soft labels if not already present in the data
    if "p_ensemble" in df.columns:
        coverage = df["p_ensemble"].notna().sum()
        print(f"Soft labels: {coverage}/{len(df)} rows already in data ({coverage/len(df):.1%})", flush=True)
    elif soft_labels_path and Path(soft_labels_path).exists():
        soft = pd.read_csv(soft_labels_path, usecols=["text", "p_ensemble"])
        before = len(df)
        df = df.merge(soft, on="text", how="left")
        coverage = df["p_ensemble"].notna().sum()
        print(f"Soft labels merged: {coverage}/{before} rows ({coverage/before:.1%})", flush=True)
    else:
        df["p_ensemble"] = float("nan")

    # Collect all unique species mentioned in THIS dataset for the gazetteer
    extra_species: set[str] = set()
    for col in ("source_species", "target_species"):
        if col in df.columns:
            for val in df[col].dropna():
                for name in str(val).split(","):
                    name = name.strip()
                    if len(name) >= 4:
                        extra_species.add(name)

    train_df, val_df = train_test_split(
        df, test_size=val_frac, stratify=df["label"], random_state=seed
    )

    train_ds = MultiTaskDataset(train_df, tokenizer_name, ner_scheme, max_length, extra_species)
    val_ds   = MultiTaskDataset(val_df,   tokenizer_name, ner_scheme, max_length, extra_species)
    return train_ds, val_ds
