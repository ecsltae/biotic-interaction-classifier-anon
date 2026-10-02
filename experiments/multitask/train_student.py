#!/usr/bin/env python3
"""Distil the query-conditioned LLM verifier into a small cross-encoder.

Input is a genuine segment pair: segment A is the candidate triple, segment B is
the passage. This is the formulation the teacher was prompted with, and the one
the deployed sentence-level filter lacks.
"""
import argparse, json, hashlib, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup, set_seed)
from sklearn.metrics import average_precision_score, f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"src"))
sys.path.insert(0, str(REPO/"scripts"))
import xenc_format

class DS(Dataset):
    """Tokenises once, pads per batch in `collate`.

    Padding is masked out, so per-batch padding is numerically identical to padding
    every example to max_length -- it just stops the GPU multiplying zeros. The
    examples here average ~55 tokens against a 256 limit, so this is most of the
    forward pass.
    """
    def __init__(self, df, tok, max_len=256, fmt="triple"):
        self.q, self.p = xenc_format.build_many(
            fmt, df.source_species, df.interaction_type, df.target_species, df.text.astype(str))
        self.y = df.label.astype(int).tolist()
        self.tok, self.ml = tok, max_len
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        if self.p is None:
            e = self.tok(self.q[i], truncation=True, max_length=self.ml)
        else:
            e = self.tok(self.q[i], self.p[i], truncation="only_second", max_length=self.ml)
        return dict(e) | {"labels": self.y[i]}

def collate(tok):
    def f(batch):
        y = torch.tensor([b.pop("labels") for b in batch])
        return dict(tok.pad(batch, padding=True, return_tensors="pt")) | {"labels": y}
    return f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training/distill/student_train.csv")
    ap.add_argument("--encoder", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--dev-data", default=None,
                    help="external development set for epoch selection and the threshold; "
                         "default a pair-grouped --val-frac split of --data")
    ap.add_argument("--input-format", default="triple", choices=list(xenc_format.FORMATS))
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="split each batch into chunks of this size and accumulate gradients; "
                         "0 = off. Same gradient, less peak memory (the box is shared).")
    ap.add_argument("--pos-weight", type=float, default=None,
                    help="weight on the positive class in CE loss (control arm); default None = unweighted")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_float32_matmul_precision("high"); set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(REPO/a.data)
    # group split on taxon pair so a pair never spans train and dev
    df["_pk"] = [tuple(sorted([str(x).lower(), str(y).lower()]))
                 for x, y in zip(df.source_species, df.target_species)]
    if a.dev_data:
        tr, va = df, pd.read_csv(REPO/a.dev_data)
        print(f"train {len(tr)} / dev {len(va)}  (external dev set {a.dev_data})", flush=True)
    else:
        rng = np.random.RandomState(a.seed)
        pairs = df._pk.unique(); rng.shuffle(pairs)
        ndev = int(len(pairs)*a.val_frac)
        devp = set(pairs[:ndev])
        tr, va = df[~df._pk.isin(devp)], df[df._pk.isin(devp)]
        print(f"train {len(tr)} / dev {len(va)}  (pair-grouped split, {len(devp)} dev pairs)", flush=True)

    tok = AutoTokenizer.from_pretrained(a.encoder)
    model = AutoModelForSequenceClassification.from_pretrained(a.encoder, num_labels=2).to(dev)
    cf = collate(tok)
    tl = DataLoader(DS(tr, tok, a.max_len, a.input_format), batch_size=a.batch_size, shuffle=True,
                    num_workers=2, collate_fn=cf)
    vl = DataLoader(DS(va, tok, a.max_len, a.input_format), batch_size=a.batch_size*2,
                    num_workers=2, collate_fn=cf)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1*len(tl)*a.epochs), len(tl)*a.epochs)

    cw = None if a.pos_weight is None else torch.tensor([1.0, a.pos_weight], device=dev)
    if cw is not None:
        eff = a.pos_weight*tr.label.sum()/(a.pos_weight*tr.label.sum()+(1-tr.label).sum())
        print(f"class-weighted CE: pos_weight={a.pos_weight:.4f} -> effective pos rate {eff:.4f}", flush=True)
    best, best_state, hist = -1, None, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        for b in tl:
            b = {k: v.to(dev) for k, v in b.items()}
            n = b["labels"].shape[0]
            mb = a.micro_batch if a.micro_batch and a.micro_batch < n else n
            for i in range(0, n, mb):                      # exact same gradient as one pass
                c = {k: v[i:i+mb] for k, v in b.items()}
                w = c["labels"].shape[0]/n
                if cw is None:
                    loss = model(**c).loss*w
                else:
                    y = c.pop("labels")
                    # normalise by the FULL batch's weight sum so accumulating the chunks
                    # reproduces the full-batch weighted mean exactly
                    loss = torch.nn.functional.cross_entropy(
                        model(**c).logits, y, weight=cw, reduction="sum")/cw[b["labels"]].sum()
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for b in vl:
                y = b.pop("labels"); b = {k: v.to(dev) for k, v in b.items()}
                P.extend(torch.softmax(model(**b).logits.float(), -1)[:,1].cpu().numpy()); Y.extend(y.numpy())
        ap_ = average_precision_score(Y, P)
        hist.append({"epoch": ep+1, "dev_auprc": float(ap_)})
        print(f"  ep{ep+1} dev AUPRC {ap_:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if ap_ > best:
            best, best_state = ap_, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_P, best_Y = np.array(P), np.array(Y)     # the threshold must come from the saved epoch
    model.load_state_dict(best_state)
    out = REPO/a.out; out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    # dev-derived threshold, never from test
    grid = np.arange(0.01,1.0,0.01)
    t = float(grid[int(np.argmax([f1_score(best_Y,(best_P>=g).astype(int),zero_division=0) for g in grid]))])
    (out/"student_config.json").write_text(json.dumps({
        "threshold_dev": t, "best_dev_auprc": best, "seed": a.seed, "epochs": a.epochs,
        "input_format": a.input_format, "encoder": a.encoder, "max_len": a.max_len, "lr": a.lr,
        "micro_batch": a.micro_batch,
        "data": a.data, "dev_data": a.dev_data, "train_pos_rate": float(df.label.mean()), "pos_weight": a.pos_weight,
        "git": subprocess.run(["git","describe","--always","--dirty"],cwd=REPO,capture_output=True,text=True).stdout.strip(),
        "history": hist}, indent=2))
    print(f"saved {out}  dev_t={t:.3f}  dev AUPRC {best:.4f}", flush=True)

if __name__ == "__main__": main()
