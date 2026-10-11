#!/usr/bin/env python3
"""Check every number in Paper A against the result files.

Collects every numeric value in the result JSONs (recursively; also x100 for percentages and
EP F1, and differences between same-level values of two arms for "gap" numbers), extracts every
number from paperA.tex with its line, and reports the numbers that no result value explains at
the paper's printed precision. Unexplained numbers are not necessarily wrong (dataset sizes,
hyperparameters, years, section counts); they are the list to check by hand.

Usage
  python3 scripts/audit_paper_numbers.py [--all]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Superseded snapshots and one-off outputs scored with old labels, models or protocols. Left in the
# sources they make a stale printed number look explained, so they are not read:
SUPERSEDED = {
    "results/paperA_v2/tables_biodiv.pre_llm.json",             # 2026-10-02 snapshot
    "results/paperA_v2/tables_biored.original_protocol.json",
    "results/paperA_v2/tables_biored.pre_llmpair.json",
}
# Every file in results/paperA_rebuild_2026-09-28 is superseded or scored on old labels:
# ablation.json, eval_student_v3.json, vs_v1_full.json, rules_eval.json, joint_rules_eval.json and
# rebuild_Verifier-{base,PC}.json (superseded models, thresholds or protocols), and
# rebuild_Sentence-baseline.json (old labels; its per-seed F1 at tau = 0.5 is now in
# results/paperA_v2/derived_biodiv.json). That directory is no longer a source.
SOURCES = [*(f for f in sorted((REPO / "results/paperA_v2").glob("*.json"))
             if str(f.relative_to(REPO)) not in SUPERSEDED),
           REPO / "results/shipping_2026-10-02/eval_a05.json",
           REPO / "results/shipping_2026-10-02/eval_v3.json"]
for _f in ("derived_biodiv.json", "cascade.json"):   # label-dependent numbers no other file holds
    if REPO / "results/paperA_v2" / _f not in SOURCES:
        SOURCES.append(REPO / "results/paperA_v2" / _f)
# The four body figures' scripts write the values each figure and its caption rely on, computed
# from the files above (and, for Figure 4's BioRED base rates, from the label column of
# results/paperA_v2/llm/biored_*_pair.csv, which no JSON holds).
for _f in ("fig_concept.json", "fig_curves.json", "fig_where.json", "fig_scaling.json"):
    if REPO / "paperA/fig" / _f not in SOURCES:
        SOURCES.append(REPO / "paperA/fig" / _f)
# In those figure files, the plotted series (thousands of curve points) and the layout values
# would explain almost any three-decimal number by coincidence, so they are not read.
FIGURE_SKIP = re.compile(r"\.(panel_a\.curves|panel_b\.(tau|pair_P|pair_R|pair_F1|sentence_F1|band)"
                         r"|\w*_at_recall|layout_in|size_in|width_in|height_in|x_positions|x_span|ylim"
                         r"|separator_x)\b")


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool) and math.isfinite(o):
        yield path, float(o)


def collect():
    vals = []
    for f in SOURCES:
        if not f.exists() or "backup" in str(f):
            continue
        try:
            o = json.loads(f.read_text())
        except Exception:
            continue
        flat = list(walk(o, f.stem))
        if f.parent == REPO / "paperA/fig":
            flat = [(p, v) for p, v in flat if not FIGURE_SKIP.search(p)]
        vals += flat
        vals += [(p + "*100", v * 100) for p, v in flat if abs(v) <= 1.0]
        # differences between sibling metrics of two arms (e.g. pair.auprc_mean - sentence.auprc_mean)
        by_leaf = {}
        for p, v in flat:
            leaf = re.sub(r"^[^.]*\.[^.]*", "", p)       # drop file and arm
            by_leaf.setdefault(leaf, []).append((p, v))
        for leaf, items in by_leaf.items():
            if len(items) > 40:
                continue
            for (p1, v1), (p2, v2) in combinations(items, 2):
                vals.append((f"{p1} - {p2}", v1 - v2))
                vals.append((f"{p2} - {p1}", v2 - v1))
    return vals


# Also matches numbers written without a leading zero, as in table cells (".858") and subscripts
# ("_{\pm .006}").
NUM = re.compile(r"(?<![\w.\\{])([+-]?(?:\d+(?:\{,\}\d{3})*(?:\.\d+)?|\.\d+))(?:\s*\\times\s*10\^\{(-?\d+)\})?")


def explained(tok, exp, vals):
    s = tok.replace("{,}", "")
    try:
        x = float(s)
    except ValueError:
        return None
    if exp is not None:                                  # a p-value like 1.3\times10^{-7}
        target = x * 10 ** int(exp)
        digits = len(s.split(".")[1]) if "." in s else 0
        return [p for p, v in vals if v > 0 and abs(float(f"{v:.{digits}e}") - target) <= target * 1e-9]
    digits = len(s.split(".")[1]) if "." in s else 0
    tol = 0.5 * 10 ** -digits + 1e-9
    return [p for p, v in vals if abs(v - x) <= tol]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="also list explained numbers with one source each")
    a = ap.parse_args()
    vals = collect()
    tex = (REPO / "paperA/paperA.tex").read_text().split("\n")
    start = next(i for i, l in enumerate(tex) if l.startswith(r"\begin{abstract}"))
    unexplained, n = [], 0
    for i, line in enumerate(tex[start:], start + 1):
        if line.lstrip().startswith("%"):
            continue
        for m in NUM.finditer(line):
            tok, exp = m.group(1), m.group(2)
            core = tok.lstrip("+-").replace("{,}", "")
            if exp is None and "." not in core and int(float(core)) < 10:
                continue                                 # small integers: section numbers, seeds...
            n += 1
            hit = explained(tok, exp, vals)
            if not hit:
                unexplained.append((i, tok + (f"e{exp}" if exp else ""), line.strip()[:110]))
            elif a.all:
                print(f"ok  {i:5d} {tok:>10}  {hit[0][:70]}")
    print(f"\n{n} numbers checked, {len(unexplained)} not explained by a result file:\n")
    for i, tok, ctx in unexplained:
        print(f"{i:5d} {tok:>12}  {ctx}")


if __name__ == "__main__":
    main()
