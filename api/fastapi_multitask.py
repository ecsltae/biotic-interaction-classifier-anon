#!/usr/bin/env python3
"""
Biotic Interaction Classifier — Multi-task BiomedBERT (full_typed_a05_ner2)
===========================================================================
Joint classification + species NER (HOST/PATHOGEN/SPECIES + interaction terms).
EP-relax F1=0.868 | AUC=0.887 | beats BiomedBERT×FLAN-T5 ensemble (F1=0.857)

Endpoints:
  GET  /health          — status + model info
  POST /predict         — single sentence → label + probability
  POST /batch           — list of sentences → list of results (max 500)
"""

import sys
import time
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
import uvicorn

# Import multi-task model class
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments/multitask"))
from model import MultiTaskBiomedBERT

# ── Config ────────────────────────────────────────────────────────────────

MODEL_DIR  = Path("/path/to/MetaP/classifier/models/multitask/full_typed_a05_ner2")
THRESHOLD  = 0.13   # optimised on EP-relax (F1=0.868, AUC=0.887)
MAX_LENGTH = 256
DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Load model at startup ─────────────────────────────────────────────────

print(f"Loading MultiTask BiomedBERT (full_typed_a05_ner2) from {MODEL_DIR} ...", flush=True)
_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
_model     = MultiTaskBiomedBERT.load(str(MODEL_DIR), device=str(DEVICE))
_model.eval()
print(f"Model loaded on {DEVICE}. Threshold={THRESHOLD}", flush=True)

# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Biotic Interaction Classifier — Multi-task BiomedBERT",
    description=(
        "Detects whether a sentence describes a biotic interaction between two species. "
        "Multi-task BiomedBERT: joint classification + HOST/PATHOGEN/SPECIES NER. "
        "EP-relax F1=0.868 (beats BiomedBERT×FLAN-T5 ensemble F1=0.857)."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(examples=["Wolbachia pipientis infects Drosophila melanogaster."])
    threshold: float = Field(default=THRESHOLD, examples=[THRESHOLD])

class PredictResponse(BaseModel):
    text: str
    label: int          # 1 = biotic interaction, 0 = not
    probability: float  # P(biotic interaction)
    threshold_used: float

class BatchRequest(BaseModel):
    sentences: List[str] = Field(
        examples=[["Wolbachia pipientis infects Drosophila melanogaster.", "This gene was first cloned in 1998."]]
    )
    threshold: float = Field(default=THRESHOLD, examples=[THRESHOLD])

class BatchResponse(BaseModel):
    results: List[PredictResponse]
    n_positive: int
    n_total: int

# ── Inference ─────────────────────────────────────────────────────────────

def _predict_batch(texts: list[str], threshold: float) -> list[dict]:
    enc = _tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        out = _model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc.get("token_type_ids"),
        )
        probs = torch.softmax(out["cls_logits"], dim=-1)[:, 1].cpu().tolist()
    return [
        {
            "text": text,
            "label": int(p >= threshold),
            "probability": round(p, 4),
            "threshold_used": threshold,
        }
        for text, p in zip(texts, probs)
    ]

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "multitask_full_typed_a05_ner2",
        "architecture": "BiomedBERT-base + NER head (HOST/PATHOGEN/SPECIES/INT)",
        "ner_scheme": "full_typed",
        "alpha": 0.5,
        "pretrain_ner_epochs": 2,
        "ep_relax_f1": 0.868,
        "ep_relax_auc": 0.887,
        "synth_gold_f1": 0.930,
        "default_threshold": THRESHOLD,
        "device": str(DEVICE),
        "beats_ensemble": True,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=422, detail="text cannot be empty")
    return _predict_batch([req.text], req.threshold)[0]


@app.post("/batch", response_model=BatchResponse)
def batch_predict(req: BatchRequest):
    if not req.sentences:
        raise HTTPException(status_code=422, detail="sentences list cannot be empty")
    if len(req.sentences) > 500:
        raise HTTPException(status_code=422, detail="max 500 sentences per batch")

    results = []
    batch_size = 64
    for i in range(0, len(req.sentences), batch_size):
        results.extend(_predict_batch(req.sentences[i:i + batch_size], req.threshold))

    return {
        "results": results,
        "n_positive": sum(r["label"] for r in results),
        "n_total": len(results),
    }


if __name__ == "__main__":
    uvicorn.run("fastapi_multitask:app", host="0.0.0.0", port=8003, reload=False)
