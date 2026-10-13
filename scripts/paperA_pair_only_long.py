#!/usr/bin/env python3
"""Is the pair-only model of Appendix Q under-trained? (review response, 2026-10-12)

The three-epoch pair-only models' development AUPRC was still rising at their last epoch. The same
recipe was rerun for up to ten epochs, keeping the best development epoch
(scripts/run_pair_only_long_2026-10-12.sh, models/pair_only_long/), and scored once on the 437 clean
TaxaPairs rows (scripts/paperA_pair_only.py score). This compares them with the three-epoch models
and with the pair-conditioned model: per-seed AUPRC, the three-checkpoint ensemble, and a paired
bootstrap of the ensemble difference (2,000 resamples of candidates, numpy default_rng seed 0, a
single-class resample redrawn, percentile 95% interval: the convention of paperA_pair_only.py).

Usage
  python3 scripts/paperA_pair_only_long.py   # -> results/paperA_v2/pair_only_long.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as ap

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from eval.core import clean_benchmark  # noqa: E402

SCORES = REPO / "results/paperA_v2/pair_only_scores"
OUT = REPO / "results/paperA_v2/pair_only_long.json"


def boot(y, a, b, n=2000, seed=0):
    rng, d = np.random.default_rng(seed), []
    while len(d) < n:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(ap(y[i], b[i]) - ap(y[i], a[i]))
    return {"diff": float(ap(y, b) - ap(y, a)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def main() -> None:
    y = clean_benchmark().label.to_numpy().astype(int)
    out = {"n": int(len(y)), "positives": int(y.sum())}
    ens = {}
    for name, stem, mdir in (("three_epochs", "biodiv_pair_only_s{}", "models/pair_only/biodiv_pair_only_s{}"),
                             ("up_to_ten_epochs", "biodiv_pair_only_long_s{}", "models/pair_only_long/biodiv_pair_only_s{}")):
        per, best_epochs = [], []
        for s in (1, 2, 3):
            v = np.load(SCORES / f"{stem.format(s)}.npy")
            assert len(v) == len(y)
            per.append(v)
            cfg = json.loads((REPO / mdir.format(s) / "student_config.json").read_text())
            hist = cfg["history"]
            best_epochs.append(int(max(hist, key=lambda h: h["dev_auprc"])["epoch"]))
        aps = [float(ap(y, v)) for v in per]
        ens[name] = np.mean(per, axis=0)
        out[name] = {"auprc_per_seed": aps, "auprc_mean": float(np.mean(aps)), "auprc_sd": float(np.std(aps, ddof=1)),
                     "auprc_ensemble": float(ap(y, ens[name])), "best_dev_epoch_per_seed": best_epochs}
    pair = np.load(REPO / "results/paperA_v2/S_biodiv_pair.npy")
    out["pair_conditioned_ensemble_auprc"] = float(ap(y, pair))
    out["pair_minus_pair_only_long"] = boot(y, ens["up_to_ten_epochs"], pair)
    out["long_minus_three_epochs"] = boot(y, ens["three_epochs"], ens["up_to_ten_epochs"])
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
