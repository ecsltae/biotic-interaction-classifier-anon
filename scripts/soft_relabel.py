#!/usr/bin/env python3
"""Re-query the teacher for P(yes) on every training row, for soft-label distillation.

Same model (qwen3:32b, local Ollama), same prompt (scripts/teacher_label_triples.py PROMPT), same
inputs as the corpus row, temperature 0 and thinking off; the only difference from the original
labelling run is that the first answer token's log-probabilities are kept, so the student can be
trained on the teacher's P(yes) instead of its verdict. Agreement between the new verdicts and the
corpus's hard labels is printed at the end as a consistency check.

Resumable: rows already in the output are skipped; progress is flushed every 500 rows.

Usage
  python3 scripts/soft_relabel.py --data data/training/distill/v3_combined_train.csv \
      --out data/training/distill/v3_combined_train_softlabels.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from llm_baseline import ask, teacher_prompt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/training/distill/v3_combined_train.csv")
    ap.add_argument("--out", default="data/training/distill/v3_combined_train_softlabels.csv")
    ap.add_argument("--model", default="qwen3:32b")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d = pd.read_csv(REPO / a.data)
    out = REPO / a.out
    done = set(pd.read_csv(out).row) if out.exists() else set()
    todo = [i for i in range(len(d)) if i not in done]
    print(f"{len(d)} rows, {len(done)} done, {len(todo)} to go", flush=True)
    tmpl = teacher_prompt()

    def job(i):
        r = d.iloc[i]
        prompt = tmpl.format(sent=r.text, s1=r.source_species, rel=r.interaction_type, s2=r.target_species)
        for attempt in range(3):                 # one timed-out call must not end a five-hour run
            try:
                p, t = ask(a.model, prompt)
                return {"row": i, "p_yes": p, "raw": t[:12]}
            except Exception as e:  # noqa: BLE001
                err = type(e).__name__
                time.sleep(5)
        return {"row": i, "p_yes": float("nan"), "raw": "ERR:" + err}

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for k in range(0, len(todo), 500):
            chunk = list(ex.map(job, todo[k:k + 500]))
            pd.DataFrame(chunk).to_csv(out, mode="a", header=not out.exists(), index=False)
            n = k + len(chunk)
            rate = n / (time.time() - t0)
            print(f"  {len(done) + n}/{len(d)}  {rate:.2f} rows/s  eta {(len(todo) - n) / rate / 3600:.1f} h", flush=True)
    s = pd.read_csv(out).drop_duplicates("row").set_index("row").reindex(range(len(d)))
    ok = s.p_yes.notna().to_numpy()
    agree = ((s.p_yes[ok] >= 0.5).astype(int).to_numpy() == d.label.to_numpy()[ok]).mean()
    print(f"done: {ok.sum()} rows with P(yes), {(~ok).sum()} failed (their soft label falls back to the "
          f"corpus label); verdict agreement with the corpus labels {agree:.3f}", flush=True)
    soft = s.p_yes.to_numpy().copy(); soft[~ok] = d.label.to_numpy()[~ok]
    d.assign(soft_label=soft).to_csv(REPO / a.out.replace("_softlabels.csv", "_soft.csv"), index=False)
    print(f"wrote {a.out.replace('_softlabels.csv', '_soft.csv')} (corpus + soft_label column)", flush=True)


if __name__ == "__main__":
    main()
