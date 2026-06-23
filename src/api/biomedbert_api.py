#!/usr/bin/env python3
"""
FastAPI for Biotic Interaction Classification using a Fine-Tuned BiomedBERT Model
===============================================================================
This script sets up a FastAPI server to classify sentences as positive or negative
for biotic interaction using a pre-trained BiomedBERT model.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize FastAPI app
app = FastAPI(
    title="Biotic Interaction Classifier API",
    description="Classifies sentences as positive or negative for biotic interaction.",
    docs_url="/"
)

# Load model and tokenizer
model_path = "transformer_BiomedBERT_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Request and response schemas
class TextClassificationRequest(BaseModel):
    texts: list[str]

class SinglePrediction(BaseModel):
    prediction: str  # "Positive" or "Negative"
    negative_probability: float
    positive_probability: float

class TextClassificationResponse(BaseModel):
    results: list[SinglePrediction]

# Endpoint
@app.post("/classify/", response_model=TextClassificationResponse)
async def classify_text(request: TextClassificationRequest):
    try:
        texts = request.texts
        inputs = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        results = []
        for pred, prob in zip(preds, probs):
            prediction = "Positive" if pred == 1 else "Negative"
            results.append(SinglePrediction(
                prediction=prediction,
                negative_probability=float(prob[0]),
                positive_probability=float(prob[1])
            ))

        return TextClassificationResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
