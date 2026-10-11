#!/usr/bin/env python3
"""Paper A's label-dependent biodiversity numbers that no other result file stores.

scripts/paperA_tables.py writes the tables; this script writes the numbers the paper prints from
them or next to them that were computed by hand until 2026-10-06: the argument strata of Table 2,
the per-seed F1 at tau = 0.5 of the per-seed table, the candidate rules on the pair model, the pair
model's false positives (and results/paperA_v2/false_positives_pair.csv), the Figure 4 claim, the
Biotx100 halves, the share of positives in multi-taxon passages, the Reject50 recovery counts, the
model trained on the base sample alone, the recommended model's rules against no rules, and the
zero-shot gaps quoted in the body. Everything is computed from cached scores and the benchmark
labels; nothing is copied from another result file (where another file holds the same number, the
two are checked against each other).

Inputs
  * labels: src/eval/core.py clean_benchmark(--labels), hash-checked
  * <results-dir>/S_biodiv_{sentence,pair,triple}.npy and <results-dir>/per_seed/S_biodiv_*.npy,
    written by scripts/paperA_tables.py --bench biodiv (run it first, with the same --labels and
    --out-dir); <results-dir>/tables_biodiv.json is read only to cross-check
  * results/paperA_v2/llm/biodiv_*.csv (p_yes and verdict; their stored label column is ignored)
  * the recommended model's p_interact (--bench-scores, from scripts/eval_shipping.py)
  * TaxoNERD counts: results/paperA_rebuild_2026-09-28/n_taxa_per_row.csv and
    results/paperA_v2/arg_recognition.csv (row-aligned with the 437 clean rows)
  * data/evaluation/biotx_retrieval_eval_100.csv (which Biotx100 rows are GloBI-matched or random)
  * models/student/xenc_s{1,2,3} (the triple model trained on the 20,000-row base sample): scored
    once on the GPU, order-free like every pair or triple arm, and cached in
    <results-dir>/per_seed/S_biodiv_triple_basesample.npy (label-independent; --rescore redoes it)

Outputs: <results-dir>/derived_biodiv.json and <results-dir>/false_positives_pair.csv

Usage
  python3 scripts/paperA_derived_numbers.py
  python3 scripts/paperA_derived_numbers.py --labels pre_review_2026-10-06 --results-dir <dir> \\
      --bench-scores <dir>/bench_a05.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from eval.core import LABEL_VERSIONS, benchmark_provenance, clean_benchmark  # noqa: E402
import candidate_rules as CR  # noqa: E402
from paperA_tables import GRID, block_held_out, mcnemar, n_taxa_multi, order_free, prf  # noqa: E402

RESULTS = REPO / "results/paperA_v2"
LLM_DIR = REPO / "results/paperA_v2/llm"
ARMS = ("sentence", "pair", "triple")
BASESAMPLE = "models/student/xenc_s{}"          # triple input, 20,000-row base sample only
TEACHER = "qwen3-32b"                           # the labelling model
SCALING = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b-q4_K_M", "qwen3-8b", "qwen3-14b",
           "qwen3-30b-a3b-q4_K_M", "qwen3-32b", "qwen3.5-122b")
FP_COLUMNS = ["source", "species1", "relation", "species2", "score", "rule", "n_taxa", "both_args_taxa", "sentence"]


def rel(p) -> str:
    """A path as recorded in the output: relative to the repository when it lies inside it."""
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def norm(s) -> str:
    return re.sub(r"\W+", " ", str(s)).lower().strip()


def counts(y, m) -> dict:
    return {"n": int(m.sum()), "positives": int(y[m].sum()), "negatives": int((y[m] == 0).sum())}


def mc(a, b, y) -> dict:
    p, fx, br = mcnemar(a, b, y)
    return {"mcnemar_p": p, "fixed": fx, "broken": br}


def with_errors(y, p) -> dict:
    return {**prf(y, p), "FP": int(((p == 1) & (y == 0)).sum()), "FN": int(((p == 0) & (y == 1)).sum())}


def seed_stats(vals) -> dict:
    return {"per_seed": [float(v) for v in vals], "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}


def first_rule(d, rules) -> np.ndarray:
    """Name of the first rule (in list order) that rejects each candidate, '' when none does."""
    return np.array([next((n for n, fn in rules if fn(str(p), a, r, b)), "")
                     for p, a, r, b in zip(d.sentence, d.species1, d.relation, d.species2)], dtype=object)


def biotx100_halves(d, y) -> dict:
    """GloBI-matched and random halves of Biotx100, joined to the source file on the normalised
    passage and both taxon surface forms (every clean row matches exactly one source row)."""
    src = pd.read_csv(REPO / "data/evaluation/biotx_retrieval_eval_100.csv", sep=";", encoding="utf-8-sig")
    kind = {}
    for _, r in src.iterrows():
        kind.setdefault((norm(r.sentence), norm(r.species1_form), norm(r.species2_form)), set()).add(r.type)
    m = (d.source == "biotx100").to_numpy()
    types = []
    for s, a, b in zip(d.sentence[m], d.species1[m], d.species2[m]):
        k = kind.get((norm(s), norm(a), norm(b)))
        assert k is not None and len(k) == 1, f"Biotx100 row not matched to one source type: {a} / {b}"
        types.append(next(iter(k)))
    types, yb = np.array(types), y[m]
    return {t: {"n": int((types == t).sum()), "positives": int(yb[types == t].sum())} for t in sorted(set(types))}


def basesample_scores(d, cache: Path, rescore: bool) -> np.ndarray:
    if cache.exists() and not rescore:
        per = np.load(cache)
    else:
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_float32_matmul_precision("high")
        per = np.array([order_free(REPO / BASESAMPLE.format(s), d, dev) for s in (1, 2, 3)])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, per)
        print(f"scored {BASESAMPLE} -> {cache}", file=sys.stderr)
    assert per.shape == (3, len(d)), f"{cache}: shape {per.shape}, expected (3, {len(d)})"
    return per


def llm_frame(name, d) -> pd.DataFrame:
    o = pd.read_csv(LLM_DIR / f"biodiv_{name}.csv")
    for c in ("species1", "species2", "sentence"):
        assert (o[c].astype(str).to_numpy() == d[c].astype(str).to_numpy()).all(), f"{name}: {c} misaligned"
    return o


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--labels", choices=sorted(LABEL_VERSIONS), default="current",
                     help="label version of the benchmark (src/eval/core.py LABEL_VERSIONS)")
    ap_.add_argument("--results-dir", default=str(RESULTS),
                     help="where paperA_tables.py wrote tables_biodiv.json and the score vectors; outputs go here too")
    ap_.add_argument("--bench-scores", default=str(REPO / "results/shipping_2026-10-02/bench_a05.csv"),
                     help="the recommended model's per-row scores (only p_interact is read)")
    ap_.add_argument("--recommended-threshold", type=float, default=0.5,
                     help="the recommended model's fixed threshold (its checkpoint's threshold_dev)")
    ap_.add_argument("--rescore", action="store_true", help="re-score the base-sample model even if cached")
    a = ap_.parse_args()
    res_dir = Path(a.results_dir)

    d = clean_benchmark(a.labels)
    prov = benchmark_provenance(a.labels)
    y, blocks = d.label.to_numpy(), d.source.to_numpy()
    multi = n_taxa_multi(d)
    nt = pd.read_csv(REPO / "results/paperA_rebuild_2026-09-28/n_taxa_per_row.csv")
    ar = pd.read_csv(REPO / "results/paperA_v2/arg_recognition.csv")
    assert len(ar) == len(d) and (ar.n_taxa.to_numpy() == nt.n_taxa.to_numpy()).all(), \
        "arg_recognition.csv is not row-aligned with n_taxa_per_row.csv"
    both = ar.both_args_taxa.to_numpy().astype(bool)
    tables = json.loads((res_dir / "tables_biodiv.json").read_text())
    if tables.get("benchmark_sha256") != prov["sha256"]:
        sys.exit(f"{res_dir / 'tables_biodiv.json'} was not scored against the {a.labels} labels; run "
                 f"scripts/paperA_tables.py --bench biodiv --labels {a.labels} --out-dir {res_dir} first")

    # ── cached scores: ensembles and the three checkpoints of each arm ──────────────────────────
    S = {arm: np.load(res_dir / f"S_biodiv_{arm}.npy") for arm in ARMS}
    PER = {arm: np.load(res_dir / "per_seed" / f"S_biodiv_{arm}.npy") for arm in ARMS}
    PRED, THR = {}, {}
    for arm in ARMS:
        assert np.array_equal(np.mean(PER[arm], axis=0), S[arm]), f"{arm}: per-seed scores do not average to the ensemble"
        PRED[arm], THR[arm] = block_held_out(y, S[arm], blocks)
        assert THR[arm] == tables[arm]["thresholds"] and prf(y, PRED[arm])["F1"] == tables[arm]["F1"], \
            f"{arm}: block-held-out operating point differs from tables_biodiv.json"
    out = {"benchmark": prov,
           "inputs": {"scores": rel(res_dir), "deployed_scores": rel(a.bench_scores),
                      "n_taxa": "results/paperA_rebuild_2026-09-28/n_taxa_per_row.csv",
                      "arg_recognition": "results/paperA_v2/arg_recognition.csv",
                      "llm": "results/paperA_v2/llm/biodiv_*.csv (p_yes, verdict)"}}

    # ── benchmark composition ───────────────────────────────────────────────────────────────────
    out["positives_ge3_taxa"] = {"positives": int(y[multi].sum()), "of_positives": int(y.sum()),
                                 "share": float(y[multi].sum() / y.sum())}
    out["biotx100_halves"] = biotx100_halves(d, y)

    # ── Table 2: where the gain lives (ensemble AUPRC by stratum) ───────────────────────────────
    strata = {"le2_taxa": ~multi, "ge3_taxa": multi,
              "both_taxa_le2": both & ~multi, "both_taxa_ge3": both & multi,
              "not_a_taxon_le2": ~both & ~multi, "not_a_taxon_ge3": ~both & multi,
              "not_a_taxon": ~both, "both_taxa": both}
    out["strata"] = {}
    for name, m in strata.items():
        auprc = {arm: float(ap(y[m], S[arm][m])) for arm in ARMS}
        out["strata"][name] = {**counts(y, m), "auprc": auprc,
                               "gain_pair_vs_sentence": auprc["pair"] - auprc["sentence"],
                               "gain_triple_vs_sentence": auprc["triple"] - auprc["sentence"]}
    for arm in ARMS:
        assert abs(out["strata"]["le2_taxa"]["auprc"][arm] - tables[arm]["auprc_le2_taxa"]) < 1e-12
        assert abs(out["strata"]["ge3_taxa"]["auprc"][arm] - tables[arm]["auprc_ge3_taxa"]) < 1e-12
    # the stratum where the sentence-level and the pair question coincide: both arguments are
    # recognised taxa and the passage names no third one
    out["coinciding_stratum"] = {"stratum": "both_taxa_le2", **counts(y, both & ~multi)}
    out["share_with_a_non_taxon_argument"] = float((~both).mean())

    # ── per-seed table: AUPRC and F1 at the fixed tau = 0.5 ─────────────────────────────────────
    out["per_seed"] = {}
    for arm in ARMS:
        aps = [ap(y, p) for p in PER[arm]]
        assert np.allclose(aps, tables[arm]["auprc_per_seed"], rtol=0, atol=1e-12)
        out["per_seed"][arm] = {"auprc": seed_stats(aps),
                                "f1_at_0.5": seed_stats([f1_score(y, (p >= 0.5).astype(int)) for p in PER[arm]])}

    # ── Figure 4: pair against sentence-level F1 over the 99-threshold grid ─────────────────────
    Fp = np.array([f1_score(y, (S["pair"] >= t).astype(int), zero_division=0) for t in GRID])
    Fs = np.array([f1_score(y, (S["sentence"] >= t).astype(int), zero_division=0) for t in GRID])
    gap = Fp - Fs
    out["fig4"] = {"thresholds": int(len(GRID)), "pair_ahead": int((gap > 0).sum()),
                   "min_gap": float(gap.min()), "min_gap_tau": float(round(GRID[int(np.argmin(gap))], 2))}

    # ── candidate rules on the pair and triple models (block-held-out thresholds) ───────────────
    R7 = np.array([any(fn(str(p), a_, r, b) for _, fn in CR.RULES)
                   for p, a_, r, b in zip(d.sentence, d.species1, d.relation, d.species2)])
    first8 = first_rule(d, CR.RULES_WITH_COLIST)
    R8 = first8 != ""
    out["rules"] = {}
    for arm in ("pair", "triple"):
        out["rules"][arm] = {"no_rules": prf(y, PRED[arm])}
        for tag, rj in (("rules7", R7), ("rules8", R8)):
            pr = PRED[arm] * ~rj
            out["rules"][arm][tag] = {**prf(y, pr), **mc(PRED[arm], pr, y)}
    for tag in ("rules7", "rules8"):             # the triple model's rows are also in tables_biodiv.json
        t = tables[tag]
        o = out["rules"]["triple"][tag]
        assert (o["fixed"], o["broken"], o["F1"]) == (t["fixed"], t["broken"], t["F1"]), f"triple {tag} differs"

    # ── the pair model's false positives at its operating point ─────────────────────────────────
    fp = (PRED["pair"] == 1) & (y == 0)
    out["pair_false_positives"] = {"n": int(fp.sum()), "ge3_taxa": int((fp & multi).sum()),
                                   "non_taxon_argument": int((fp & ~both).sum()),
                                   "caught_by_8_rules": int((fp & R8).sum()), "caught_by_7_rules": int((fp & R7).sum()),
                                   "by_rule": {k: int(v) for k, v in pd.Series(first8[fp & R8]).value_counts().items()},
                                   "file": "false_positives_pair.csv"}
    fpt = d.loc[fp, ["source", "species1", "relation", "species2"]].copy()
    fpt["score"] = S["pair"][fp]
    fpt["rule"] = first8[fp]
    fpt["n_taxa"] = nt.n_taxa.to_numpy()[fp]
    fpt["both_args_taxa"] = both[fp]
    fpt["sentence"] = d.sentence[fp].values
    fpt[FP_COLUMNS].to_csv(res_dir / "false_positives_pair.csv", index=False)

    # ── recall side ─────────────────────────────────────────────────────────────────────────────
    m = blocks == "reject50"
    out["reject50"] = {arm: {"recovered": int(((PRED[arm] == 1) & (y == 1) & m).sum()), "of_positives": int(y[m].sum()),
                             "readmitted": int(((PRED[arm] == 1) & (y == 0) & m).sum()),
                             "of_negatives": int((y[m] == 0).sum()),
                             "F1": float(f1_score(y[m], PRED[arm][m], zero_division=0))} for arm in ARMS}

    # ── the triple model trained on the 20,000-row base sample alone ────────────────────────────
    per = basesample_scores(d, res_dir / "per_seed" / "S_biodiv_triple_basesample.npy", a.rescore)
    ens = np.mean(per, axis=0)
    pred, thr = block_held_out(y, ens, blocks)
    out["basesample_triple"] = {"checkpoints": [BASESAMPLE.format(s) for s in (1, 2, 3)],
                                "auprc": seed_stats([ap(y, p) for p in per]), "auprc_ensemble": float(ap(y, ens)),
                                "block_held_out": {**prf(y, pred), "thresholds": thr},
                                "f1_at_0.5": seed_stats([f1_score(y, (p >= 0.5).astype(int)) for p in per])}

    # ── the recommended model: candidate rules against none ────────────────────────────────────────
    st = pd.read_csv(a.bench_scores)
    assert (st.species1.values == d.species1.values).all(), "recommended scores misaligned"
    s = st.p_interact.to_numpy()
    rej = np.array([CR.reject_reason(str(p), a_, r, b) is not None
                    for p, a_, r, b in zip(d.sentence, d.species1, d.relation, d.species2)])
    raw = (s >= a.deployed_threshold).astype(int)
    kept = raw * ~rej
    colist = np.array([CR.co_listed(str(p), a_, b) for p, a_, b in zip(d.sentence, d.species1, d.species2)])
    hit = (kept == 1) & colist
    out["recommended"] = {"threshold": a.deployed_threshold, "auprc": float(ap(y, s)),
                       "no_rules": with_errors(y, raw), "with_rules": with_errors(y, kept),
                       "rules_vs_no_rules": mc(raw, kept, y),
                       "rejected_by_rules": {k: int(v) for k, v in pd.Series(
                           [CR.reject_reason(str(p), a_, r, b) for p, a_, r, b, x in
                            zip(d.sentence, d.species1, d.relation, d.species2, raw) if x == 1]).value_counts().items()},
                       "co_listed_on_accept_set": {"fires": int(hit.sum()), "removes_fp": int((hit & (y == 0)).sum()),
                                                   "removes_tp": int((hit & (y == 1)).sum()),
                                                   **mc(kept, kept * ~colist, y)}}

    # ── zero-shot gaps quoted in the body (LLM AUPRC from p_yes against these labels) ───────────
    def llm_auprc(name, mask=None):
        o = llm_frame(name, d)
        mm = np.ones(len(d), bool) if mask is None else mask
        return float(ap(y[mm], o.p_yes.to_numpy()[mm]))

    t = {q: llm_auprc(f"{TEACHER}_{q}") for q in ARMS}
    tq = llm_frame(f"{TEACHER}_triple", d)
    tpred, _ = block_held_out(y, tq.p_yes.to_numpy(), blocks)
    student = {arm: out["per_seed"][arm]["auprc"]["mean"] for arm in ARMS}
    out["teacher_gaps"] = {
        "teacher": TEACHER, "auprc": t,
        "pair_minus_sentence": t["pair"] - t["sentence"],
        "pair_minus_sentence_ge3": llm_auprc(f"{TEACHER}_pair", multi) - llm_auprc(f"{TEACHER}_sentence", multi),
        "pair_minus_sentence_le2": llm_auprc(f"{TEACHER}_pair", ~multi) - llm_auprc(f"{TEACHER}_sentence", ~multi),
        "triple_minus_pair": t["triple"] - t["pair"],
        "student_pair_minus_sentence_per_seed": student["pair"] - student["sentence"],
        "student_pair_minus_sentence_ge3": out["strata"]["ge3_taxa"]["gain_pair_vs_sentence"],
        "student_pair_minus_sentence_le2": out["strata"]["le2_taxa"]["gain_pair_vs_sentence"],
        "student_triple_minus_pair_per_seed": student["triple"] - student["pair"],
        "student_pair_minus_teacher_sentence": student["pair"] - t["sentence"],
        "teacher_triple_minus_student_pair": t["triple"] - student["pair"],
        "student_pair_F1": prf(y, PRED["pair"])["F1"], "teacher_triple_F1": prf(y, tpred)["F1"]}
    out["scaling_pair_minus_sentence"] = {name: llm_auprc(f"{name}_pair") - llm_auprc(f"{name}_sentence")
                                          for name in SCALING}
    out["scaling_sentence_auprc"] = {name: llm_auprc(f"{name}_sentence") for name in SCALING}

    (res_dir / "derived_biodiv.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
