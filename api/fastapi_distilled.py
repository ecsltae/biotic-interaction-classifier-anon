#!/usr/bin/env python3
"""
Biotic Interaction Classifier — Distilled BiomedBERT v2
========================================================
Single model distilled from BiomedBERT cv_reg × FLAN-T5-base v12 ensemble.
EP-relax F1=0.808 | Synth gold F1=0.959 | 109M params | single forward pass

Endpoints:
  GET  /health          — status + model info
  POST /predict         — single sentence → label + probability
  POST /batch           — list of sentences → list of results
"""

import re
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────

MODEL_DIR  = Path("/path/to/MetaP/classifier/models/distilled_BiomedBERT_v2")
THRESHOLD  = 0.25   # optimised on EP-relax (F1=0.808, Prec=0.750, Rec=0.875)
MAX_LENGTH = 256
DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Load model at startup ─────────────────────────────────────────────────

print(f"Loading distilled BiomedBERT v2 from {MODEL_DIR} ...", flush=True)
_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
_model     = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR), local_files_only=True).to(DEVICE).eval()
print(f"Model loaded on {DEVICE}. Threshold={THRESHOLD}", flush=True)

# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Biotic Interaction Classifier — Distilled BiomedBERT v2",
    description=(
        "Detects whether a sentence describes a biotic interaction between two species. "
        "Distilled from BiomedBERT cv_reg × FLAN-T5-base v12 ensemble (EP F1=0.857). "
        "Single model: EP F1=0.808, 109M params."
    ),
    version="2.0.0",
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
    sentences: List[str] = Field(examples=[["Wolbachia pipientis infects Drosophila melanogaster.", "This gene was first cloned in 1998."]])
    threshold: float = Field(default=THRESHOLD, examples=[THRESHOLD])

class BatchResponse(BaseModel):
    results: List[PredictResponse]
    n_positive: int
    n_total: int

# ── Inference ─────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_one(text: str, threshold: float) -> dict:
    processed = preprocess(text)
    enc = _tokenizer(
        processed, return_tensors="pt",
        truncation=True, max_length=MAX_LENGTH,
    ).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(_model(**enc).logits, dim=-1)[0, 1].item()
    return {
        "text": text,
        "label": int(prob >= threshold),
        "probability": round(prob, 4),
        "threshold_used": threshold,
    }

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "distilled_BiomedBERT_v2",
        "architecture": "BiomedBERT-base (109M params)",
        "teacher": "BiomedBERT cv_reg × FLAN-T5-base v12 (geo ensemble)",
        "ep_relax_f1": 0.808,
        "synth_gold_f1": 0.959,
        "default_threshold": THRESHOLD,
        "device": str(DEVICE),
        "distillation": {"temperature": 2, "alpha": 0.5},
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=422, detail="text cannot be empty")
    return predict_one(req.text, req.threshold)


@app.post("/batch", response_model=BatchResponse)
def batch_predict(req: BatchRequest):
    if not req.sentences:
        raise HTTPException(status_code=422, detail="sentences list cannot be empty")
    if len(req.sentences) > 500:
        raise HTTPException(status_code=422, detail="max 500 sentences per batch")
    results = [predict_one(s, req.threshold) for s in req.sentences]
    return {
        "results": results,
        "n_positive": sum(r["label"] for r in results),
        "n_total": len(results),
    }


if __name__ == "__main__":
    uvicorn.run("fastapi_distilled:app", host="0.0.0.0", port=8003, reload=False)
