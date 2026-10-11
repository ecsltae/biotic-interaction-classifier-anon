#!/usr/bin/env python3
"""Near-duplicate scan of the benchmark against the students' training passages.

For every benchmark row, the maximum 5-gram Jaccard similarity between its passage and any
training passage, after lower-casing and replacing non-alphanumerics with spaces. The exact-match
`in_train` column of the benchmark flags 9 rows; this scan adds 3 more (Jaccard 1.000, all in
biotx100, differing only in mojibake), and every clean-benchmark loader drops in_train or
maxj > 0.5, leaving 437 rows. No row sits at exactly 0.5.

results/test_contamination_scan.csv was first written by this logic on 2026-09-25 and rewritten
on 2026-10-02 with no change to `maxj`: the first copy carried row 400's label from before its
2026-09-25 gold revision (kept as test_contamination_scan.pre_goldfix.csv). `--check` recomputes
the scan and asserts the result is byte-identical to the file.

Usage
  python3 scripts/scan_contamination.py            # write results/test_contamination_scan.csv
  python3 scripts/scan_contamination.py --check    # recompute and compare with the shipped file
"""
from __future__ import annotations

import argparse
import io
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data/evaluation/unified_test_set.csv"
TRAIN = REPO / "data/training/distill/v4_species_train.csv"   # same 48,338 passages as v3_combined_train
OUT = REPO / "results/test_contamination_scan.csv"


def norm(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def shingles(s, k: int = 5) -> set:
    t = norm(s).split()
    return set(tuple(t[i:i + k]) for i in range(max(1, len(t) - k + 1)))


def scan() -> pd.DataFrame:
    d = pd.read_csv(BENCH)
    train = pd.read_csv(TRAIN, usecols=["text"]).text.astype(str)
    inv, tr_sh = defaultdict(set), []
    for i, t in enumerate(train):
        sh = shingles(t)
        tr_sh.append(sh)
        for g in sh:
            inv[g].add(i)
    best = []
    for s in d.sentence:
        sh = shingles(s)
        cand = defaultdict(int)
        for g in sh:
            for i in inv.get(g, ()):
                cand[i] += 1
        # exact maximum over every training passage sharing a shingle
        best.append(max((ov / len(sh | tr_sh[i]) for i, ov in cand.items()), default=0.0))
    return d.assign(maxj=best)[["source", "sentence", "label", "in_train", "maxj"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    res = scan()
    if a.check:
        buf = io.StringIO()
        res.to_csv(buf, index=False)
        same = buf.getvalue().encode() == OUT.read_bytes()
        print(f"recomputed scan is byte-identical to {OUT.relative_to(REPO)}: {same}")
        if not same:
            old = pd.read_csv(OUT)
            diff = (old.maxj - res.maxj).abs()
            print(f"  rows differing: {int((diff > 1e-12).sum())}, max |diff| {diff.max():.3g}")
            raise SystemExit(1)
        return
    res.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO)}: {int((res.maxj > 0.5).sum())} rows above 0.5, "
          f"{int(((res.maxj > 0.5) | res.in_train).sum())} dropped with in_train")


if __name__ == "__main__":
    main()
